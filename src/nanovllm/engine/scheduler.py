from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_prefill_tokens_per_step = (
            config.max_prefill_tokens_per_step or config.max_num_batched_tokens
        )
        self.prefill_decode_ratio = config.prefill_decode_ratio
        self._decode_steps_since_prefill = 0
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self._peak_waiting_sequences = 0
        self._peak_running_sequences = 0

    def _update_peaks(self) -> None:
        self._peak_waiting_sequences = max(
            self._peak_waiting_sequences, len(self.waiting)
        )
        self._peak_running_sequences = max(
            self._peak_running_sequences, len(self.running)
        )


    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)
        self._update_peaks()

    def _should_schedule_decode(self) -> bool:
        """Prefer decode when waiting prefill would otherwise starve ITL."""

        return bool(self.running) and (
            not self.waiting
            or self._decode_steps_since_prefill < self.prefill_decode_ratio
        )

    # schedule 收集 要处理的token序列 及其对应的flag(p or d)，貌似一次就反馈一组？就返回一个元组
    def schedule(self) -> tuple[list[Sequence], bool]:
        self._update_peaks()
        if self._should_schedule_decode():
            scheduled = self._schedule_decode()
            if scheduled:
                self._decode_steps_since_prefill += 1
                self._update_peaks()
                return scheduled, False

        scheduled = self._schedule_prefill()
        if scheduled:
            self._decode_steps_since_prefill = 0
            self._update_peaks()
            return scheduled, True

        scheduled = self._schedule_decode()
        if scheduled:
            self._decode_steps_since_prefill += 1
            self._update_peaks()
            return scheduled, False
        raise RuntimeError("scheduler made no progress")

    def _schedule_prefill(self) -> list[Sequence]:
        scheduled_seqs = []
        num_batched_tokens = 0
        prefill_budget = self.max_prefill_tokens_per_step

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = min(
                self.max_num_batched_tokens - num_batched_tokens,
                prefill_budget - num_batched_tokens,
            )
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        return scheduled_seqs

    def _schedule_decode(self) -> list[Sequence]:
        scheduled_seqs = []
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft() # 取出running队头的准备decode
            while not self.block_manager.can_append(seq): # 不能申请KV cache了
                if self.running: # 先优先踢出running的队末
                    self.preempt(self.running.pop())
                else: # 这个情况是 waiting队列很多 ，running空了，但是还是优先于prefill 导致 自己刚从running出来准备decode失败
                    self.preempt(seq) # 只能踢走自己了
                    break
            else:  # 推理
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        # KV pressure may preempt every running sequence. In that case the
        # caller must be allowed to schedule waiting prefill instead of
        # crashing on an empty decode batch.
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs

    # 用于将处于running队列或者即将decode的 提回等待队列
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    def stats(self) -> dict[str, int]:
        """Return scheduler and KV-cache counters for runtime observability."""

        return {
            "waiting_sequences": len(self.waiting),
            "running_sequences": len(self.running),
            "peak_waiting_sequences": self._peak_waiting_sequences,
            "peak_running_sequences": self._peak_running_sequences,
            "decode_steps_since_prefill": self._decode_steps_since_prefill,
            "prefill_decode_ratio": self.prefill_decode_ratio,
            **self.block_manager.stats(),
        }
