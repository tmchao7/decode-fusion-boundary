"""Lock the schedule verifier: field decoding, the checks, and the mutants.

The verifier is the pre-server gate for anything that changes ``schedule.py``.
These tests pin the three things that make its verdicts meaningful: the opcode
field layout (which the CUDA header defines), the two detectors, and the fact
that the adversarial population contains no false accepts.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import unittest

from nanovllm.backends.megakernel import schedule as tg
from nanovllm.backends.megakernel import verifier as V
from taskgraph_fixtures import BALANCED_CHAIN, BALANCED_CYCLE, RawSchedule, instruction  # noqa: E402

class OpcodeDecodingTests(unittest.TestCase):
    """The field indices must match ``csrc/taskgraph_kernel.cu``."""

    def test_qkv_proj_waits_on_layer_and_arrives_on_qkv(self) -> None:
        decoded = V.decode_instruction(
            instruction(tg.TgOp.QKV_PROJ, f5=3, f6=2, f7=4, f9=5)
        )

        self.assertEqual(len(decoded.waits), 1)
        self.assertEqual(decoded.waits[0], V.Condition(V.BarKey("layer", 3), 2))
        self.assertEqual(decoded.arrivals, ((V.BarKey("qkv", 4), 5),))

    def test_attn_decode_combine_bit_retargets_the_arrival(self) -> None:
        plain = V.decode_instruction(
            instruction(tg.TgOp.ATTN_DECODE, f7=1, f8=1, f9=2)
        )
        combine = V.decode_instruction(
            instruction(tg.TgOp.ATTN_DECODE, f7=1, f8=1, f9=2 | (1 << 31))
        )

        self.assertEqual(plain.arrivals, ((V.BarKey("attn", 2), 1),))
        self.assertEqual(combine.arrivals, ((V.BarKey("combine", 2), 1),))
        self.assertEqual(plain.waits, combine.waits)

    def test_attn_combine_waits_on_combine_and_arrives_on_attn(self) -> None:
        decoded = V.decode_instruction(
            instruction(tg.TgOp.ATTN_COMBINE, f5=2, f6=7, f7=9)
        )

        self.assertEqual(decoded.waits, (V.Condition(V.BarKey("combine", 2), 7),))
        self.assertEqual(decoded.arrivals, ((V.BarKey("attn", 9), 1),))

    def test_add_rmsnorm_needs_both_branches_before_advancing_the_layer(self) -> None:
        decoded = V.decode_instruction(
            instruction(tg.TgOp.ADD_RMSNORM, f3=0, f4=1, f5=0, f6=1, f7=4)
        )

        self.assertEqual(
            decoded.waits,
            (
                V.Condition(V.BarKey("oproj", 0), 1),
                V.Condition(V.BarKey("ffn_down", 0), 1),
            ),
        )
        self.assertEqual(decoded.arrivals, ((V.BarKey("layer", 4), 1),))

    def test_ffn_and_lm_head_chains(self) -> None:
        o_proj = V.decode_instruction(instruction(tg.TgOp.O_PROJ, f6=1, f7=2, f8=3, f9=4))
        upgate = V.decode_instruction(
            instruction(tg.TgOp.FFN_UPGATE_ACT, f6=1, f7=2, f8=3, f9=4)
        )
        down = V.decode_instruction(instruction(tg.TgOp.FFN_DOWN, f6=1, f7=2, f8=3, f9=4))
        lm_head = V.decode_instruction(instruction(tg.TgOp.LM_HEAD, f6=1, f7=2, f8=3, f9=4))
        argmax = V.decode_instruction(instruction(tg.TgOp.LM_ARGMAX, f6=1, f7=2))

        self.assertEqual(o_proj.waits, (V.Condition(V.BarKey("attn", 1), 2),))
        self.assertEqual(o_proj.arrivals, ((V.BarKey("oproj", 3), 4),))
        self.assertEqual(upgate.waits, (V.Condition(V.BarKey("oproj", 1), 2),))
        self.assertEqual(upgate.arrivals, ((V.BarKey("silu", 3), 4),))
        self.assertEqual(down.waits, (V.Condition(V.BarKey("silu", 1), 2),))
        self.assertEqual(down.arrivals, ((V.BarKey("ffn_down", 3), 4),))
        self.assertEqual(lm_head.waits, (V.Condition(V.BarKey("layer", 1), 2),))
        self.assertEqual(lm_head.arrivals, ((V.BarKey("lm_head", 3), 4),))
        self.assertEqual(argmax.waits, (V.Condition(V.BarKey("lm_head", 1), 2),))
        self.assertEqual(argmax.arrivals, ())

    def test_nop_has_no_dependencies(self) -> None:
        decoded = V.decode_instruction(instruction(tg.TgOp.NOP))

        self.assertEqual(decoded.waits, ())
        self.assertEqual(decoded.arrivals, ())
        self.assertEqual(decoded.name, "NOP")

    def test_a_truncated_instruction_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            V.decode_instruction([tg.TgOp.NOP, 0, 0])


class BarKeyCompatibilityTests(unittest.TestCase):
    def test_barkeys_are_plain_tuple_keys(self) -> None:
        """The ABI tests index bar state as ``bars[("layer", 1)]``."""

        self.assertEqual(V.BarKey("layer", 1), ("layer", 1))
        self.assertEqual({V.BarKey("layer", 1): 5}[("layer", 1)], 5)


class ValidScheduleTests(unittest.TestCase):
    def test_every_grid_size_and_attn_split_verifies(self) -> None:
        for batch_size in (1, 2, 4, 8):
            for num_sms in (1, 4, 108, 170):
                for attn_splits in (1, 2):
                    with self.subTest(bs=batch_size, sms=num_sms, attn=attn_splits):
                        sched = tg.build_schedule(
                            tg.default_geometry(batch_size, attn_splits=attn_splits),
                            num_sms,
                        )
                        report = V.verify(sched, num_sms)
                        self.assertTrue(report.ok, report.describe())
                        self.assertIsNone(report.cycle)
                        self.assertTrue(report.arrival_audit.ok)

    def test_attn_splits_two_adds_instructions_but_stays_balanced(self) -> None:
        """The old 'attn_splits=2 deadlocks' record is settled here, not on the GPU."""

        single = V.verify(tg.build_schedule(tg.default_geometry(1, attn_splits=1), 108), 108)
        split = V.verify(tg.build_schedule(tg.default_geometry(1, attn_splits=2), 108), 108)

        self.assertTrue(single.ok and split.ok)
        self.assertGreater(split.instructions, single.instructions)


class SegmentedScheduleTests(unittest.TestCase):
    def _segments(self, groups: int, num_sms: int = 108, batch_size: int = 1):
        geo = tg.default_geometry(batch_size)
        return [
            tg.build_schedule(geo, num_sms, layer_range=layer_range)
            for layer_range in tg.split_layers(geo.num_layers, groups)
        ]

    def test_every_fusion_granularity_verifies(self) -> None:
        for groups in (1, 2, 4, 7, 14, 28):
            with self.subTest(groups=groups):
                report = V.verify_segmented(self._segments(groups), 108)
                self.assertTrue(report.ok, report.describe())

    def test_segmentation_conserves_instructions(self) -> None:
        reference = V.verify(self._segments(1)[0], 108).instructions

        for groups in (2, 4, 7, 14, 28):
            with self.subTest(groups=groups):
                self.assertEqual(
                    V.verify_segmented(self._segments(groups), 108).instructions,
                    reference,
                )

    def test_a_later_segment_alone_is_not_verifiable(self) -> None:
        """Why segments need carried counters: segment 2 waits on segment 1's work."""

        segments = self._segments(2)
        alone = V.verify(segments[1], 108)

        self.assertFalse(alone.ok)
        self.assertFalse(alone.arrival_audit.ok)
        self.assertGreater(alone.instructions, 0)


class ArrivalAuditTests(unittest.TestCase):
    def test_a_balanced_cycle_is_balanced(self) -> None:
        """Four instructions in a cycle whose arrival counts exactly cancel."""

        rows = BALANCED_CYCLE
        report = V.verify(RawSchedule(rows, [4], 4), 1)

        self.assertTrue(report.arrival_audit.ok)
        self.assertFalse(report.ok)

    def test_a_consistent_chain_balances(self) -> None:
        healthy = RawSchedule(BALANCED_CHAIN, [4], 4)

        self.assertTrue(V.audit_arrivals(healthy, 1).ok)

    def test_doubling_an_arrival_is_over_arrival(self) -> None:
        rows = BALANCED_CHAIN
        rows[0][9] = 4  # qkv arrives twice as often as anyone consumes

        audit = V.audit_arrivals(RawSchedule(rows, [4], 4), 1)

        self.assertFalse(audit.ok)
        self.assertIn(V.BarKey("qkv", 0), audit.over)

    def test_a_missing_arrival_is_under_arrival(self) -> None:
        rows = BALANCED_CHAIN
        rows[0][9] = 1  # qkv now arrives once, but ATTN waits for 2

        audit = V.audit_arrivals(RawSchedule(rows, [4], 4), 1)

        self.assertFalse(audit.ok)
        self.assertIn(V.BarKey("qkv", 0), audit.under)


class CycleLocalizerTests(unittest.TestCase):
    def test_names_a_cycle_whose_arrivals_balance(self) -> None:
        rows = BALANCED_CYCLE
        report = V.verify(RawSchedule(rows, [4], 4), 1)

        self.assertIsNotNone(report.cycle)
        assert report.cycle is not None
        # Closed walk over distinct instructions, first == last.
        self.assertEqual(report.cycle[0], report.cycle[-1])
        self.assertEqual(len(set(report.cycle)), len(report.cycle) - 1)
        self.assertTrue(all(sm == 0 for sm, _ in report.cycle))

    def test_real_schedules_report_no_cycle(self) -> None:
        for batch_size in (1, 8):
            for num_sms in (1, 108, 170):
                with self.subTest(bs=batch_size, sms=num_sms):
                    sched = tg.build_schedule(tg.default_geometry(batch_size), num_sms)
                    self.assertIsNone(V.wait_graph_cycle(sched, num_sms))

    def test_the_localizer_does_not_fire_on_a_healthy_prefix(self) -> None:
        """A partial stream with an unsatisfied wait is a deadlock, not a cycle."""

        rows = [instruction(tg.TgOp.QKV_PROJ, f5=0, f6=1, f7=0, f9=1)]
        report = V.verify(RawSchedule(rows, [1], 1), 1)

        self.assertFalse(report.ok)
        self.assertIsNone(report.cycle)


class MutantGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sched = tg.build_schedule(tg.default_geometry(1), 1)
        self.num_sms = 1

    # A tiny raw schedule keeps this exhaustive: materialising the population of
    # a full geometry (tens of thousands of 72k-int buffers) is minutes of copying.
    SMALL = BALANCED_CYCLE

    def test_the_population_is_not_empty_and_every_mutant_differs(self) -> None:
        for sched, num_sms in (
            (RawSchedule(self.SMALL, [4], 4), 1),
            (self.sched, self.num_sms),
        ):
            with self.subTest(vertices=len(sched.num_inst_per_sm)):
                population = V.mutant_count(sched, num_sms)
                self.assertGreater(population, 5)
                sample = list(V.mutants(sched, num_sms, sample=30))
                self.assertLessEqual(len(sample), population)
                for mutant in sample:
                    differs = (
                        mutant.flat_inst_buf != sched.flat_inst_buf
                        or mutant.num_inst_per_sm != sched.num_inst_per_sm
                    )
                    self.assertTrue(differs, f"no-op mutant: {mutant.label}")

    def test_the_whole_population_matches_its_count_on_a_small_schedule(self) -> None:
        sched = RawSchedule(self.SMALL, [4], 4)

        self.assertEqual(
            len(list(V.mutants(sched, 1))), V.mutant_count(sched, 1)
        )

    def test_a_nonsensical_sample_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            list(V.mutants(self.sched, self.num_sms, sample=0))

    def test_generation_is_deterministic(self) -> None:
        first = [m.label for m in V.mutants(self.sched, self.num_sms, sample=20)]
        second = [m.label for m in V.mutants(self.sched, self.num_sms, sample=20)]

        self.assertEqual(first, second)

    def test_sampling_returns_a_subset(self) -> None:
        everything = {m.label for m in V.mutants(self.sched, self.num_sms)}
        sampled = [m.label for m in V.mutants(self.sched, self.num_sms, sample=25)]

        self.assertEqual(len(sampled), 25)
        self.assertTrue(set(sampled).issubset(everything))

    def test_every_mutation_kind_is_represented(self) -> None:
        kinds = {
            mutant.label.split(":")[0]
            for mutant in V.mutants(self.sched, self.num_sms, sample=60)
        }

        self.assertEqual(
            kinds,
            {"drop_arrival", "double_arrival", "shift_bar", "raise_threshold",
             "truncate_stream"},
        )


class AdversarialReportTests(unittest.TestCase):
    def test_zero_false_accepts_on_a_sampled_population(self) -> None:
        sched = tg.build_schedule(tg.default_geometry(1), 1)
        report = V.adversarial_report([("BS1/sms1", sched, 1)], sample=24)

        self.assertEqual(report.valid_total, 1)
        self.assertEqual(report.accepted_valid, 1)
        self.assertEqual(report.mutant_total, 24)
        self.assertEqual(report.rejected_mutants, 24)
        self.assertEqual(report.false_accepts, [])
        self.assertTrue(report.ok)
        self.assertTrue(report.sampled)
        self.assertGreater(report.candidate_total, report.mutant_total)
        # Both detectors must contribute, or one of them is not pulling weight.
        self.assertTrue(report.rejected_checks)

    def test_a_rejected_valid_schedule_is_a_false_accept(self) -> None:
        """The report must flag a *valid* schedule it cannot verify."""

        # A QKV wait that nothing satisfies: the report must call this a false
        # accept, not silently count it as a rejected mutant.
        rows = [instruction(tg.TgOp.QKV_PROJ, f5=0, f6=1, f7=0, f9=1)]
        report = V.adversarial_report(
            [("broken", RawSchedule(rows, [1], 1), 1)], sample=None
        )

        self.assertEqual(report.accepted_valid, 0)
        self.assertTrue(report.false_accepts)
        self.assertFalse(report.ok)


if __name__ == "__main__":
    unittest.main()
