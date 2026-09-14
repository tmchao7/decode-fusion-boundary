import atexit
from dataclasses import fields
from time import perf_counter
import warnings
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.burst_policy import burst_batch_eligibility, clamp_burst_batch
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.metrics import CudaEventTimer, MetricsCollector, nvtx_range
from nanovllm.utils.distributed import build_init_method


class LLMEngine:


    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        Sequence.block_size = config.kvcache_block_size
        # The Megakernel contract fixes max_model_len at 2048 and the check is
        # inside compatibility(), which is only reached per step -- a user who
        # sets decode_backend="megakernel" but leaves the 4096 default gets a
        # silent full fallback.  Say it once, loudly.
        if config.decode_backend == "megakernel" and config.max_model_len > 2048:
            warnings.warn(
                "decode_backend='megakernel' requires max_model_len <= 2048 "
                f"(got {config.max_model_len}); every request will fall back to "
                "the torch path. Pass max_model_len=2048 to actually use the "
                "Megakernel.",
                RuntimeWarning,
            )
        self._exited = False
        # Burst path guards (see _step_burst): a burst that makes no progress
        # must not spin the caller's while-loop, and the burst selection must
        # respect the batch kernel's real launch limit.  The limit is queried
        # once and cached -- it costs CUDA driver calls.
        self._burst_disabled = False
        self._burst_batch_limit_cache: int | None = None
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        init_method = build_init_method()
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(
                target=ModelRunner,
                args=(config, i, event, init_method),
            )
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events, init_method)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        if self._exited:
            return
        self._exited = True
        try:
            self.model_runner.call("exit")
        except Exception as exc:
            # Preserve the original inference exception. CUDA cleanup can fail
            # again after a device-side fault and must not replace it.
            warnings.warn(f"nano-vLLM cleanup warning: {exc}", RuntimeWarning)
        del self.model_runner
        # Each benchmark backend is created in the same Python process. Release
        # PyTorch's cached CUDA blocks so the next backend sees real free memory.
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception as exc:
            warnings.warn(f"nano-vLLM CUDA cache cleanup warning: {exc}", RuntimeWarning)
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq

    def _should_burst(self) -> bool:
        """Whether the next ``step`` can use the no-sync megakernel burst path.

        Runtime preconditions (extension healthy, nothing waiting) plus the
        pure batch-shape policy in :func:`burst_batch_eligibility`.
        """
        mr = self.model_runner
        if mr.megakernel is None or mr.megakernel.disabled_reason:
            return False
        if mr._cuda_poisoned:
            return False
        if self._burst_disabled:
            return False
        if self.scheduler.waiting:
            return False
        eligible, _ = burst_batch_eligibility(
            self.config, self.scheduler.running
        )
        return eligible

    def _burst_batch_limit(self) -> int:
        """Cached burst batch limit, clamped to the batch kernel's real limit.

        ``burst_batch_eligibility`` is torch-free and cannot see the kernel's
        occupancy limit, so the clamp happens here.  Queried once: the limit
        costs CUDA driver calls and does not change for a given extension.
        """

        if self._burst_batch_limit_cache is None:
            configured = int(getattr(self.config, "megakernel_burst_batch", 8))
            kernel_limit: int | None = None
            try:
                from nanovllm.backends.megakernel.backend import wmma_batch_limit

                kernel_limit = int(wmma_batch_limit(configured))
            except Exception:
                kernel_limit = None
            self._burst_batch_limit_cache = clamp_burst_batch(
                configured, kernel_limit
            )
        return self._burst_batch_limit_cache

    def _step_burst(self, metrics: MetricsCollector | None):
        """Execute one no-sync burst and return ``(outputs, num_tokens)``."""
        seqs, max_steps = self.scheduler.schedule_burst(
            burst_cap=getattr(self.config, "megakernel_burst_cap", 64),
            max_batch=self._burst_batch_limit(),
        )
        if not seqs or max_steps <= 0:
            return self._step_normal(metrics)

        before_completion = {
            seq.seq_id: len(seq.completion_token_ids) for seq in seqs
        }
        model_start = perf_counter()
        cuda_timer = CudaEventTimer() if metrics is not None else None
        if cuda_timer is not None:
            cuda_timer.start()
        with nvtx_range("nano_vllm_burst"):
            token_logs = self.model_runner.call("run_burst", seqs, max_steps)
        if cuda_timer is not None:
            cuda_timer.stop()
        model_duration = perf_counter() - model_start

        if token_logs is None:
            # Burst failed before the kernel ran. Undo the burst block
            # allocations so the step-wise fallback's may_append / prepare_decode
            # see a table matching num_blocks (no double-allocate, no stale
            # block_table[-1]).
            for seq in seqs:
                self.scheduler.block_manager.trim_burst(seq)
            return self._step_normal(metrics)

        self.scheduler.postprocess_burst(seqs, token_logs)
        token_timestamp = perf_counter()
        total_new = 0
        if metrics is not None:
            metrics.record_step(
                seqs,
                False,
                model_duration,
                self.model_runner.last_backend_used,
                self.model_runner.last_fallback_reason,
                cuda_timer=cuda_timer,
            )
            for seq in seqs:
                new_tokens = (
                    len(seq.completion_token_ids) - before_completion[seq.seq_id]
                )
                total_new += max(0, new_tokens)
                for _ in range(max(0, new_tokens)):
                    metrics.record_token(seq.seq_id, token_timestamp)
        else:
            for seq in seqs:
                new_tokens = (
                    len(seq.completion_token_ids) - before_completion[seq.seq_id]
                )
                total_new += max(0, new_tokens)
        outputs = [
            (seq.seq_id, seq.completion_token_ids)
            for seq in seqs
            if seq.is_finished
        ]
        if total_new == 0:
            # A burst that appended nothing would spin the caller's
            # ``while not is_finished()`` loop forever: no sequence finished,
            # no token was produced, and _should_burst() would stay true.  This
            # happens when the launch aborts (all rows come back -1) rather
            # than raising.  postprocess_burst has already trimmed the burst
            # allocations, so rerun this step through the step-wise path and
            # stop using burst for this engine.
            self._burst_disabled = True
            warnings.warn(
                "Megakernel burst produced no tokens; disabling the burst path "
                "and falling back to the step-wise schedule for this engine.",
                RuntimeWarning,
            )
            return self._step_normal(metrics)
        return outputs, -total_new

    def _step_normal(self, metrics: MetricsCollector | None):
        """Execute one normal (step-wise) schedule/run/postprocess cycle."""
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = (
            sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        )
        before_completion = {
            seq.seq_id: len(seq.completion_token_ids) for seq in seqs
        }
        model_start = perf_counter()
        cuda_timer = CudaEventTimer() if metrics is not None else None
        if cuda_timer is not None:
            cuda_timer.start()
        range_name = "nano_vllm_prefill" if is_prefill else "nano_vllm_decode_step"
        with nvtx_range(range_name):
            token_ids = self.model_runner.call("run", seqs, is_prefill)
        if cuda_timer is not None:
            cuda_timer.stop()
        model_duration = perf_counter() - model_start
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        token_timestamp = perf_counter()
        if metrics is not None:
            metrics.record_step(
                seqs,
                is_prefill,
                model_duration,
                self.model_runner.last_backend_used,
                self.model_runner.last_fallback_reason,
                cuda_timer=cuda_timer,
            )
            for seq in seqs:
                new_tokens = len(seq.completion_token_ids) - before_completion[seq.seq_id]
                for _ in range(max(0, new_tokens)):
                    metrics.record_token(seq.seq_id, token_timestamp)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def step(self, metrics: MetricsCollector | None = None):
        if self._should_burst():
            return self._step_burst(metrics)
        return self._step_normal(metrics)

    def is_finished(self):
        return self.scheduler.is_finished()

    def _generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool,
        metrics: MetricsCollector | None,
    ):
        started_at = perf_counter()
        pbar = tqdm(
            total=len(prompts),
            desc="Generating",
            dynamic_ncols=True,
            disable=not use_tqdm,
        )
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            seq = self.add_request(prompt, sp)
            if metrics is not None:
                metrics.add_request(
                    seq.seq_id,
                    seq.num_prompt_tokens,
                    perf_counter(),
                )
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        try:
            with nvtx_range("nano_vllm_request"):
                while not self.is_finished():
                    t = perf_counter()
                    output, num_tokens = self.step(metrics=metrics)
                    elapsed = perf_counter() - t
                    if num_tokens > 0:
                        prefill_throughput = num_tokens / elapsed
                    else:
                        decode_throughput = -num_tokens / elapsed
                    pbar.set_postfix({
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    })
                    for seq_id, token_ids in output:
                        outputs[seq_id] = token_ids
                        pbar.update(1)
        finally:
            pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
        report = None
        if metrics is not None:
            report = metrics.report(wall_time=perf_counter() - started_at)
        return outputs, report

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        outputs, _ = self._generate(prompts, sampling_params, use_tqdm, None)
        return outputs

    def generate_with_metrics(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = False,
    ) -> dict:
        collector = MetricsCollector(self.config.decode_backend)
        outputs, report = self._generate(
            prompts,
            sampling_params,
            use_tqdm,
            collector,
        )
        return {"outputs": outputs, "metrics": report}
