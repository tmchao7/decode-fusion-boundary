import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.metrics import MetricsCollector, nvtx_range


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        Sequence.block_size = config.kvcache_block_size
        self._exited = False
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        if self._exited:
            return
        self._exited = True
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq

    def step(self, metrics: MetricsCollector | None = None):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        before_completion = {
            seq.seq_id: len(seq.completion_token_ids) for seq in seqs
        }
        model_start = perf_counter()
        range_name = "nano_vllm_prefill" if is_prefill else "nano_vllm_decode_step"
        with nvtx_range(range_name):
            token_ids = self.model_runner.call("run", seqs, is_prefill)
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
            )
            for seq in seqs:
                new_tokens = len(seq.completion_token_ids) - before_completion[seq.seq_id]
                for _ in range(max(0, new_tokens)):
                    metrics.record_token(seq.seq_id, token_timestamp)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

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
