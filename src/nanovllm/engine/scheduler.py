from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # Observability for the mixed-length burst overrun tripwire in
        # schedule_burst() (consumed by tools/correctness_burst_mixed.py).
        # Counters are bounded; the last-burst fields describe the most recent
        # schedule_burst() call.
        self.bursts_observed = 0
        self.burst_mixed_budget_bursts = 0
        self.burst_max_steps_violations = 0
        self.burst_reservation_violations = 0
        self.last_burst_max_steps = 0
        self.last_burst_budgets: list[int] = []
        self.last_burst_reserved_tokens: list[int] = []

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    # schedule 收集 要处理的token序列 及其对应的flag(p or d)，貌似一次就反馈一组？就返回一个元组
    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
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

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
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
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    # 用于将处于running队列或者即将decode的 提回等待队列
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def schedule_burst(
        self, burst_cap: int = 64, max_batch: int = 8
    ) -> tuple[list[Sequence], int]:
        """Select running sequences for a fixed-table no-sync megakernel burst.

        Pre-allocates each sequence's decode blocks for ``max_steps`` new tokens
        so the block table stays fixed for the whole burst.  Returns ``(seqs,
        max_steps)``.

        ``max_steps`` is the **smallest** budget of the selected sequences, not
        the largest.  The device burst has no per-sequence cap: it deactivates a
        sequence only on EOS, and runs the *same* ``max_steps`` for every row of
        the batch.  Reserving only a sequence's own smaller budget while running
        the batch maximum would therefore make that sequence write K/V past the
        blocks it owns -- ``ldg_kv_ptr`` never bounds-checks, the padded
        block-table entries are another sequence's physical blocks, so the
        overrun silently corrupts a neighbour's KV (and then its attention).
        Clamping to the minimum keeps the reservation and the run length equal
        for every selected sequence, so no row can overrun.

        The cost is that one short sequence limits the whole burst; it finishes
        after this burst and the remainder is re-selected next step, so there is
        no starvation, only more (shorter) bursts.

        Only sequences whose blocks can be allocated for ``max_steps`` are
        included; if none qualify the caller should fall back to the normal
        step-wise schedule.  Selected sequences stay in ``self.running`` (no
        rotation); finished sequences are removed by :meth:`postprocess_burst`.
        """
        candidates: list[tuple[Sequence, int]] = []
        for seq in self.running:
            if len(candidates) >= max_batch:
                break
            budget = min(
                seq.max_tokens - seq.num_completion_tokens, burst_cap
            )
            if budget <= 0:
                continue
            candidates.append((seq, budget))
        if not candidates:
            return [], 0

        # Every selected sequence must be able to run the full burst length.
        max_steps = min(budget for _, budget in candidates)
        seqs: list[Sequence] = []
        selected_budgets: list[int] = []
        for seq, budget in candidates:
            if not self.block_manager.can_allocate_burst(seq, max_steps):
                continue
            self.block_manager.allocate_burst(seq, max_steps)
            seq.is_prefill = False
            seqs.append(seq)
            selected_budgets.append(budget)

        # Regression tripwire for the mixed-length overrun documented above.
        # The device burst runs ``max_steps`` for every row, so max_steps must
        # be the minimum budget and every selected sequence must be reserved
        # for that whole length; otherwise a shorter row writes past its blocks.
        # This is the same contract as ``tests/test_burst.py``, but checked on
        # real GPU runs from tools/correctness_burst_mixed.py -- a token-level
        # comparison cannot distinguish that corruption from the documented
        # bf16 ulp cascade, so the guard needs this oracle-free signal.
        # Cost is one pass over <= max_batch sequences per *burst* (not per
        # token), and the counters are bounded.
        self.bursts_observed += 1
        self.last_burst_max_steps = max_steps
        self.last_burst_budgets = selected_budgets
        self.last_burst_reserved_tokens = [
            len(seq.block_table) * self.block_size - len(seq) for seq in seqs
        ]
        if len(set(selected_budgets)) > 1:
            # A genuinely mixed-length burst.  The guard needs at least one of
            # these to be meaningful: with uniform budgets min == max, so the
            # contract below holds trivially and the overrun cannot occur.
            self.burst_mixed_budget_bursts += 1
        if max_steps != min((budget for _, budget in candidates), default=0):
            self.burst_max_steps_violations += 1
        if any(reserved < max_steps for reserved in self.last_burst_reserved_tokens):
            self.burst_reservation_violations += 1

        if not seqs:
            return [], 0
        return seqs, max_steps

    def postprocess_burst(
        self, seqs: list[Sequence], token_logs: list[list[int]]
    ) -> None:
        """Apply burst-generated tokens to sequences.

        ``token_logs[s]`` is the raw ``[max_steps]`` output for sequence ``s``.
        Entries past a per-seq EOS are -1 (sentinel). This method truncates at
        EOS (inclusive) and ``max_tokens``, appends the surviving tokens, hashes
        newly completed blocks, and deallocates finished sequences.
        """
        for seq, raw_tokens in zip(seqs, token_logs):
            remaining = seq.max_tokens - seq.num_completion_tokens
            tokens: list[int] = []
            for tok in raw_tokens:
                if tok < 0 or len(tokens) >= remaining:
                    break
                tokens.append(tok)
                if not seq.ignore_eos and tok == self.eos:
                    break
            for token_id in tokens:
                seq.append_token(token_id)
            seq.num_scheduled_tokens = len(tokens)
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            finished = bool(tokens) and (
                (not seq.ignore_eos and tokens[-1] == self.eos)
                or seq.num_completion_tokens == seq.max_tokens
            )
            if finished:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
            else:
                # Free burst-allocated blocks past the current length so the
                # next step-wise schedule/may_append sees a table that matches
                # num_blocks (no double-allocate, no stale block_table[-1]).
                self.block_manager.trim_burst(seq)

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
