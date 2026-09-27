"""Lock the structural cost model and the predictions it pre-registers.

The model is only worth having if its two claims hold: the structure is derived
from the schedule rather than hand-counted, and it predicts a configuration it
was not calibrated on. The numbers below are the documented ones from docs/12
(10.8 ms/step at 108 SMs, 275 ms/step at 1 SM, and the Amdahl interpolation
T(4) = 75.05 ms); they must not be re-tuned without updating docs/18.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import unittest
from collections import Counter

from nanovllm.backends.megakernel import schedule as tg
from nanovllm.backends.megakernel import verifier as V
from taskgraph_fixtures import instruction  # noqa: E402
from nanovllm.cost_model import (
    GRID_SYNC_US,
    HOST_LAUNCH_US,
    dependency_levels,
    fit_k_step_ns,
    predict_ms,
    preregistered_granularity_prediction,
    segment_delta_ms,
    structure_from_schedule,
    structure_from_prepared,
    structure_from_segments,
)

GEO = tg.default_geometry(1)
DOCUMENTED_DEVICE_MS_108SM = 10.8
DOCUMENTED_DEVICE_MS_1SM = 275.0
DOCUMENTED_AMDAHL_MS_4SM = 75.05
#: The corrected K-step count (docs/12 used 197 waves x 64 = 12608).
DOCUMENTED_K_STEPS = 11612


def structure(batch_size: int = 1, num_sms: int = 108, attn_splits: int = 1):
    geo = tg.default_geometry(batch_size, attn_splits=attn_splits)
    return structure_from_schedule(tg.build_schedule(geo, num_sms), num_sms, geo), geo


class StructureTests(unittest.TestCase):
    def test_level_count_follows_the_documented_wave_rule(self) -> None:
        """6 emitting waves per layer + LM_HEAD + LM_ARGMAX; the combine wave is empty."""

        step, _ = structure()

        self.assertEqual(step.levels, 6 * 28 + 2)
        self.assertEqual(step.levels, 170)

    def test_attn_splits_two_adds_the_combine_wave(self) -> None:
        single, _ = structure(attn_splits=1)
        split, _ = structure(attn_splits=2)

        self.assertEqual(split.levels, 7 * 28 + 2)
        self.assertEqual(split.levels, 198)
        # The combine wave is a reduction with no serial K loop, so it must not
        # change the modeled time -- a falsifiable prediction, see docs/18.
        self.assertEqual(split.k_steps, single.k_steps)

    def test_every_level_is_exactly_one_wave(self) -> None:
        """The level-is-a-wave assumption is asserted, not assumed."""

        step, _ = structure()
        self.assertEqual(
            [wave.name for wave in step.waves[:6]],
            ["QKV_PROJ", "ATTN_DECODE", "O_PROJ", "FFN_UPGATE_ACT", "FFN_DOWN", "ADD_RMSNORM"],
        )
        self.assertEqual(step.waves[-2].name, "LM_HEAD")
        self.assertEqual(step.waves[-1].name, "LM_ARGMAX")

    def test_a_mixed_level_is_rejected(self) -> None:
        """Guard the assumption itself: two opcodes at one depth must raise."""

        # A NOP and a QKV whose wait threshold is 0 both have depth 1 and no
        # producers, so they land in the same level with different names.
        prepared = V.PreparedSchedule(
            streams=(
                (
                    V.decode_instruction(instruction(tg.TgOp.NOP)),
                    V.decode_instruction(
                        instruction(tg.TgOp.QKV_PROJ, f5=0, f6=0, f7=0, f9=1)
                    ),
                ),
            )
        )

        self.assertEqual(len(dependency_levels(prepared)), 1)
        with self.assertRaises(RuntimeError):
            structure_from_prepared(prepared, GEO)

    def test_a_cycle_raises_instead_of_mislabelling_levels(self) -> None:
        cycle = V.PreparedSchedule(
            streams=tuple(
                V.decode_instruction(row)
                for row in (
                    instruction(tg.TgOp.QKV_PROJ, f5=0, f6=1, f7=0, f9=1),
                    instruction(tg.TgOp.ATTN_DECODE, f7=0, f8=1, f9=0),
                    instruction(tg.TgOp.O_PROJ, f6=0, f7=1, f8=0, f9=1),
                    instruction(tg.TgOp.ADD_RMSNORM, f3=0, f4=1, f5=0, f6=0, f7=0),
                )
            )
        )
        # Each stream entry is one SM; a 4-cycle needs them in one stream, so
        # build that explicitly.
        cycle = V.PreparedSchedule(
            streams=(
                tuple(
                    V.decode_instruction(row)
                    for row in (
                        instruction(tg.TgOp.QKV_PROJ, f5=0, f6=1, f7=0, f9=1),
                        instruction(tg.TgOp.ATTN_DECODE, f7=0, f8=1, f9=0),
                        instruction(tg.TgOp.O_PROJ, f6=0, f7=1, f8=0, f9=1),
                        instruction(tg.TgOp.ADD_RMSNORM, f3=0, f4=1, f5=0, f6=0, f7=0),
                    )
                ),
            )
        )

        with self.assertRaises(RuntimeError):
            dependency_levels(cycle)

    def test_the_gemm_waves_carry_their_own_k(self) -> None:
        """FFN_DOWN contracts over intermediate (3072/16 = 192), not 64."""

        step, _ = structure()
        totals = step.k_steps_by_wave()

        self.assertEqual(totals["QKV_PROJ"], 28 * 64)
        self.assertEqual(totals["O_PROJ"], 28 * 64)
        self.assertEqual(totals["FFN_UPGATE_ACT"], 28 * 64)
        self.assertEqual(totals["FFN_DOWN"], 28 * 192)
        self.assertEqual(totals["ADD_RMSNORM"], 0)

    def test_the_lm_head_needs_thirteen_rounds(self) -> None:
        step, _ = structure()
        rounds = step.rounds_by_wave()

        self.assertEqual(rounds["LM_HEAD"], 13)  # 1357 instructions / 108 SMs
        self.assertEqual(rounds["LM_HEAD"] * 64, step.k_steps_by_wave()["LM_HEAD"])

    def test_total_k_steps_match_the_documented_count(self) -> None:
        step, _ = structure()

        self.assertEqual(step.k_steps, DOCUMENTED_K_STEPS)
        self.assertEqual(step.instructions, 4522)

    def test_grid_size_only_changes_the_rounds(self) -> None:
        one_sm, _ = structure(num_sms=1)

        # With a single SM every instruction is its own round.
        self.assertEqual(one_sm.rounds, one_sm.instructions)
        self.assertEqual(one_sm.levels, 170)

    def test_batch_size_does_not_change_the_structure(self) -> None:
        """A falsifiable prediction: the wave model is BS-invariant up to BS=8.

        The archived line's BS>=2 collapse is therefore *not* explained here, and
        docs/18 says so instead of extrapolating.
        """

        reference, _ = structure(batch_size=1)
        for batch_size in (2, 4, 8):
            with self.subTest(batch_size=batch_size):
                step, _ = structure(batch_size=batch_size)
                self.assertEqual(step.levels, reference.levels)
                self.assertEqual(step.k_steps, reference.k_steps)

    def test_segmentation_does_not_change_the_structure(self) -> None:
        """The verifier's instruction conservation, seen through the cost model."""

        geo = tg.default_geometry(1)
        reference = structure_from_segments(
            [tg.build_schedule(geo, 108)], 108, geo
        )
        for groups in (2, 4, 7, 14, 28):
            with self.subTest(groups=groups):
                segments = [
                    tg.build_schedule(geo, 108, layer_range=layer_range)
                    for layer_range in tg.split_layers(28, groups)
                ]
                step = structure_from_segments(segments, 108, geo)
                self.assertEqual(step.levels, reference.levels)
                self.assertEqual(step.k_steps, reference.k_steps)
                self.assertEqual(step.instructions, reference.instructions)


class CalibrationTests(unittest.TestCase):
    def test_calibration_reproduces_the_documented_baseline(self) -> None:
        step, _ = structure()
        k_step_ns = fit_k_step_ns(step, DOCUMENTED_DEVICE_MS_108SM)

        self.assertAlmostEqual(
            predict_ms(step, k_step_ns), DOCUMENTED_DEVICE_MS_108SM, places=6
        )
        self.assertEqual(round(k_step_ns), 930)

    def test_out_of_sample_single_sm_prediction_is_within_five_percent(self) -> None:
        """Calibrated on 108 SMs; the 1-SM number is a previously documented measurement."""

        trained, _ = structure(num_sms=108)
        k_step_ns = fit_k_step_ns(trained, DOCUMENTED_DEVICE_MS_108SM)
        one_sm, _ = structure(num_sms=1)

        predicted = predict_ms(one_sm, k_step_ns)
        error = (predicted - DOCUMENTED_DEVICE_MS_1SM) / DOCUMENTED_DEVICE_MS_1SM
        self.assertLess(abs(error), 0.05, f"predicted {predicted:.1f} ms")

    def test_agrees_with_the_documented_amdahl_interpolation(self) -> None:
        """docs/12 §1.1 fitted T(n) = 8.3 + 267/n from the 1-SM and 108-SM points."""

        trained, _ = structure(num_sms=108)
        k_step_ns = fit_k_step_ns(trained, DOCUMENTED_DEVICE_MS_108SM)
        four_sm, _ = structure(num_sms=4)

        predicted = predict_ms(four_sm, k_step_ns)
        self.assertLess(abs(predicted - DOCUMENTED_AMDAHL_MS_4SM) / DOCUMENTED_AMDAHL_MS_4SM, 0.01)

    def test_predict_uses_the_calibrated_constant_linearly(self) -> None:
        step, _ = structure()

        self.assertAlmostEqual(predict_ms(step, 1000.0), step.k_steps / 1000.0, places=9)

    def test_bad_inputs_are_rejected(self) -> None:
        step, _ = structure()

        with self.assertRaises(ValueError):
            predict_ms(step, 0.0)
        with self.assertRaises(ValueError):
            fit_k_step_ns(step, 0.0)
        with self.assertRaises(ValueError):
            structure_from_schedule(tg.build_schedule(GEO, 108), 0, GEO)


class PreRegisteredPredictionTests(unittest.TestCase):
    def test_granularity_effect_is_bounded_and_monotone(self) -> None:
        prediction = preregistered_granularity_prediction(base_device_ms=1.89)

        groups = prediction["groups"]
        lower = prediction["device_ms_lower_bound"]
        upper = prediction["device_ms_upper_bound"]

        self.assertEqual(groups[0], 1)
        self.assertAlmostEqual(lower[1], 1.89)
        self.assertLess(lower[28], upper[28])
        for previous, current in zip(groups, groups[1:]):
            self.assertLess(lower[previous], lower[current])
            self.assertLess(upper[previous], upper[current])

    def test_worst_case_is_the_documented_bound(self) -> None:
        prediction = preregistered_granularity_prediction(base_device_ms=1.89)
        expected = (27 * HOST_LAUNCH_US / 1000.0) / 1.89

        self.assertAlmostEqual(prediction["worst_case_relative_change"], expected, places=9)
        self.assertLess(prediction["worst_case_relative_change"], 0.16)
        self.assertGreater(
            prediction["device_ms_lower_bound"][28],
            1.89 + 26 * 2 * GRID_SYNC_US / 1000.0 - 1e-9,
        )

    def test_segment_delta_is_linear_in_g(self) -> None:
        delta = segment_delta_ms([1, 2, 3], base_ms=2.0, per_segment_us=10.0)

        self.assertEqual(delta, {1: 2.0, 2: 2.01, 3: 2.02})

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            segment_delta_ms([1], base_ms=0.0, per_segment_us=1.0)
        with self.assertRaises(ValueError):
            segment_delta_ms([1], base_ms=1.0, per_segment_us=-1.0)
        with self.assertRaises(ValueError):
            preregistered_granularity_prediction(base_device_ms=1.0, groups=())


if __name__ == "__main__":
    unittest.main()


class PackingConsistencyTests(unittest.TestCase):
    """The model assumes `rounds = ceil(instructions / num_sms)` per wave.

    That only holds if the schedule builder really packs `_max_per_inst` tiles
    into each instruction, so check the observed per-layer instruction counts
    against the tile counts the geometry implies. QKV is excluded on purpose: it
    uses `_pack_qkv`'s opcode-specific layout, and is pinned to its observed
    count instead of a formula that does not describe it.
    """

    def test_packed_gemm_waves_match_their_tile_counts(self) -> None:
        step, geo = structure()
        waves_per_name = Counter(wave.name for wave in step.waves)
        # One wave per layer for the six layer waves, plus the two tail waves.
        self.assertEqual(waves_per_name["O_PROJ"], 28)
        self.assertEqual(waves_per_name["LM_HEAD"], 1)

        expectations = {
            "O_PROJ": tg._oproj_tile_count(geo),
            "FFN_DOWN": tg._ffn_down_tile_count(geo),
            "FFN_UPGATE_ACT": tg._ffn_upgate_tile_count(geo),
        }
        for name, tiles in expectations.items():
            per_inst = tg._max_per_inst(tiles, 108)
            with self.subTest(wave=name, tiles=tiles, per_inst=per_inst):
                self.assertEqual(per_inst, 7)
                self.assertEqual(
                    -(-tiles // per_inst),
                    self._observed_instructions(step, name),
                )

    def test_qkv_uses_its_own_layout(self) -> None:
        """`_pack_qkv` packs q/k/v separately: 19 + 10 + 10 = 39, but the observed
        count is 56. Pinned so a future change to that layout is noticed."""

        step, _ = structure()

        self.assertEqual(self._observed_instructions(step, "QKV_PROJ"), 56)
        tiles = tg._qkv_tile_count(tg.default_geometry(1))
        self.assertEqual(tiles, 256)
        self.assertNotEqual(-(-tiles // 7), 56)

    def test_lm_head_is_the_only_multi_round_wave(self) -> None:
        step, geo = structure()

        multi = {wave.name for wave in step.waves if wave.rounds > 1}
        self.assertEqual(multi, {"LM_HEAD"})
        tiles = tg._lm_head_tile_count(geo)
        self.assertEqual(-(-tiles // 7), 1357)
        self.assertEqual(len([w for w in step.waves if w.name == "LM_HEAD"]), 1)

    @staticmethod
    def _observed_instructions(step, name: str) -> int:
        for wave in step.waves:
            if wave.name == name:
                return wave.instructions
        raise AssertionError(f"no {name} wave in the structure")
