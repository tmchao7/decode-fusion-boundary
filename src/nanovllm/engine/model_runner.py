import gc
import pickle
import warnings

import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.backends.megakernel import MegakernelBackend
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.metrics import nvtx_range
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.distributed import build_init_method
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(
        self,
        config: Config,
        rank: int,
        event: Event | list[Event],
        init_method: str | None = None,
    ):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        # Megakernel owns the decode path; capturing a second CUDA-Graph is
        # unnecessary overhead. Unsupported Megakernel requests still use the
        # eager torch path through this flag.
        self.use_cudagraph = (
            not self.enforce_eager and config.decode_backend != "megakernel"
        )
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group(
            "nccl",
            init_method or build_init_method(),
            world_size=self.world_size,
            rank=rank,
        )
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.megakernel = None
        self.last_backend_used = "torch"
        self.last_fallback_reason = None
        self._cuda_poisoned = False
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if config.decode_backend == "megakernel":
            self.megakernel = MegakernelBackend(config, self.model)
            self.megakernel.set_kv_cache(self.kv_cache)
            # Build the decoder (pack weights, allocate scratch) eagerly so the
            # first decode token has no cold-start spike. On failure disable the
            # megakernel and let the torch path remain the fallback.
            try:
                self.megakernel.ensure_ready()
            except Exception as exc:  # noqa: BLE001 - fall back gracefully
                warnings.warn(
                    "Megakernel disabled at startup; using the standard "
                    f"backend: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.megakernel.disable(str(exc))
        if self.use_cudagraph:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1: # 多卡场景
            if rank == 0: # 主进程
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else: # 非主进程
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop() # 死循环等待 主进程

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if self.use_cudagraph:
            del self.graphs, self.graph_pool
        # A device-side kernel fault poisons the CUDA stream. Synchronizing
        # again during cleanup would mask the original exception with a second
        # "illegal memory access" and make diagnosis harder.
        if not self._cuda_poisoned:
            torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        # 主卡执行
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        # 所有卡执行
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """Compile a small representative path without consuming KV-cache budget.

        The previous warmup used the full ``max_num_batched_tokens`` budget
        (16,384 tokens by default). FlashAttention/torch.compile can retain a
        very large temporary allocation from that shape, leaving no room for
        the actual KV pool. Graph capture happens separately after KV allocation.
        """

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        seq_len = min(self.config.max_model_len, 512)
        seqs = [Sequence([0] * seq_len)]
        seqs[0].num_scheduled_tokens = seq_len
        try:
            self.run(seqs, True)
            torch.cuda.synchronize()
        finally:
            reset_context()
            del seqs
            gc.collect()
            torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        available_for_kv = total * config.gpu_memory_utilization - used - peak + current
        config.num_kvcache_blocks = int(available_for_kv) // block_bytes
        if config.num_kvcache_blocks <= 0:
            raise RuntimeError(
                "Insufficient free GPU memory for KV cache: "
                f"GPU total={total / 2**30:.2f} GiB, "
                f"free={free / 2**30:.2f} GiB, "
                f"model/current allocation={used / 2**30:.2f} GiB, "
                f"peak allocation={peak / 2**30:.2f} GiB, "
                f"requested utilization={config.gpu_memory_utilization:.2f}. "
                "Select an idle GPU with CUDA_VISIBLE_DEVICES or stop other GPU jobs."
            )
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens # 过滤掉seq命中的kv cache 的计算起始位置
            seqlen_q = seq.num_scheduled_tokens # 本次seq要算的token数
            end = start + seqlen_q
            seqlen_k = end # 当前序列总共拥有的 KV Cache 长度
            input_ids.extend(seq[start:end]) # 抽出 seq tokens
            positions.extend(range(start, end)) #
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            # 本 Batch 中最长的 Q 长度和最长的 K 长度
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            # 计算kv cache 的分配的槽位的 逻辑位置->物理位置的映射 slot_mapping
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache 前缀缓存命中
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token) # 取最后一个token
            positions.append(len(seq) - 1)
            context_lens.append(len(seq)) # 记录当前上下文长度
            # 本次新算出的 KV 要写入显存的哪个绝对地址
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or not self.use_cudagraph or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        self.last_backend_used = "torch"
        self.last_fallback_reason = None
        # The first Megakernel milestone is intentionally a single-sequence,
        # greedy decode path. Unsupported requests fall back to the existing
        # PyTorch/eager/CUDA-Graph implementation below.
        if self.megakernel is not None and not self.megakernel.disabled_reason:
            decision = MegakernelBackend.compatibility(
                self.config, seqs, is_prefill
            )
            if not decision.supported and not is_prefill:
                self.last_fallback_reason = decision.reason
            if decision.supported:
                try:
                    with nvtx_range("megakernel_decode"):
                        if len(seqs) == 1:
                            # Single-seq decode reads seq.last_token/len(seq)
                            # itself; only the block table (device tensor) is
                            # required, so skip the full prepare_decode() which
                            # would otherwise build unused CUDA tensors each step.
                            block_tables = self.prepare_block_tables(seqs)
                            set_context(False, block_tables=block_tables)
                            token_ids = self.megakernel.decode(
                                seqs[0], block_tables[0]
                            )
                        else:
                            input_ids, positions = self.prepare_decode(seqs)
                            context = get_context()
                            token_ids = self.megakernel.decode_batch(
                                input_ids, positions, context.block_tables
                            )
                except Exception as exc:
                    # Once a kernel step has started, nano-vLLM's shared paged
                    # cache may have been updated. Do not silently produce
                    # incorrect tokens by falling back after that point.
                    if self.megakernel.decoder is not None:
                        self._cuda_poisoned = True
                        raise RuntimeError(
                            "Megakernel decode failed after initialization: "
                            f"{exc}; restart with decode_backend=\"torch\""
                        ) from exc
                    self.megakernel.disable(str(exc))
                    self.last_fallback_reason = str(exc)
                    warnings.warn(
                        "Megakernel unavailable; falling back to the standard "
                        f"backend: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    self.last_backend_used = "megakernel"
                    return token_ids
                finally:
                    reset_context()

        # 对输入seqs进行 p or d的处理 ,这里返回对seqs中的seq的id和位置
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    def run_burst(
        self, seqs: list[Sequence], max_steps: int
    ) -> list[list[int]]:
        """Run a no-sync megakernel burst for a batch of decode sequences.

        The scheduler has already pre-allocated burst blocks (R4) so the block
        table is fixed for the whole span. Returns ``[seq][max_steps]`` raw
        token log; the scheduler ``postprocess_burst`` truncates at EOS /
        max_tokens.
        """
        self.last_backend_used = "torch"
        self.last_fallback_reason = None
        try:
            with nvtx_range("megakernel_burst"):
                block_tables = self.prepare_block_tables(seqs)
                token_logs = self.megakernel.generate_paged_batch(
                    seqs,
                    block_tables,
                    max_steps,
                    self.config.eos,
                )
        except Exception as exc:
            # The kernel has already started (and may have written to the
            # shared paged cache), so a silent fallback would corrupt state.
            if self.megakernel.decoder is not None:
                self._cuda_poisoned = True
                raise RuntimeError(
                    "Megakernel burst failed after initialization: "
                    f"{exc}; restart with decode_backend=\"torch\""
                ) from exc
            self.megakernel.disable(str(exc))
            self.last_fallback_reason = str(exc)
            warnings.warn(
                "Megakernel burst unavailable; falling back to step-wise: "
                f"{exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        else:
            self.last_backend_used = "megakernel"
            return token_logs
        finally:
            reset_context()

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
