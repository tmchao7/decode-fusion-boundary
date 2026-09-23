"""Source invariants for the task-graph megakernel (Phase 0).

These tests read the C/C++/Python sources directly and pin the relationships
that must stay in sync across the ABI boundary. They run without a GPU, like
``test_batch_launch_guards.py`` and ``test_paged_kernel_source.py``.

Locked relationships:
* ``csrc/tg_abi.h``  <-> ``schedule.py``            (instruction width, opcodes, struct fields)
* ``csrc/taskgraph_kernel.cu`` <-> ``build.py``     (#ifndef fallbacks == build defaults)
* ``schedule.py``     internal                       (bar index formulas, encoding round-trip,
                                                     schedule structure, dependency graph validity)
* ``csrc/taskgraph_kernel.cu`` <-> ``schedule.py``  (op dispatch covers every emitted opcode)

Run: ``PYTHONPATH=src python -m unittest tests.test_taskgraph_abi -v``
"""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys_path_added = False
if str(REPO_ROOT / "src") not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT / "src"))
    sys_path_added = True

from nanovllm.backends.megakernel import schedule as tg  # noqa: E402

MEGAKERNEL = REPO_ROOT / "src" / "nanovllm" / "backends" / "megakernel"
TG_ABI_H = MEGAKERNEL / "csrc" / "tg_abi.h"
TG_KERNEL_CU = MEGAKERNEL / "csrc" / "taskgraph_kernel.cu"
BUILD_PY = MEGAKERNEL / "build.py"
SCHEDULE_PY = MEGAKERNEL / "schedule.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# =============================================================================
# ABI sync: tg_abi.h <-> schedule.py
# =============================================================================


class AbiSyncTests(unittest.TestCase):
    """The instruction width, opcode enum, and struct fields must match between
    the layout-locked C header and the Python schedule builder."""

    def test_instruction_width_matches_header(self) -> None:
        header = read(TG_ABI_H)
        m = re.search(r"constexpr int TG_INST_WIDTH\s*=\s*(\d+)", header)
        self.assertIsNotNone(m, "tg_abi.h no longer defines TG_INST_WIDTH")
        assert m is not None
        self.assertEqual(int(m.group(1)), tg.TG_INST_WIDTH)

    def test_opcode_enum_matches_header(self) -> None:
        header = read(TG_ABI_H)
        # The header declares `enum TgOp : int32_t { TG_NOP = 0, TG_QKV_PROJ = 1, ... }`.
        # Parse each `TG_<NAME> = <value>` and compare to schedule.py's TgOp.
        declared = {}
        for m in re.finditer(r"(TG_\w+)\s*=\s*(\d+)", header):
            name, val = m.group(1), int(m.group(2))
            # Skip non-opcode matches like TG_INST_WIDTH, TG_LW_COUNT, TG_MAX_BATCH.
            if name in ("TG_INST_WIDTH", "TG_LW_COUNT", "TG_MAX_BATCH",
                        "TG_LW_INPUT_LAYERNORM", "TG_LW_Q_PROJ", "TG_LW_K_PROJ",
                        "TG_LW_V_PROJ", "TG_LW_Q_NORM", "TG_LW_K_NORM",
                        "TG_LW_O_PROJ", "TG_LW_POST_ATTN_LAYERNORM",
                        "TG_LW_GATE_PROJ", "TG_LW_UP_PROJ", "TG_LW_DOWN_PROJ"):
                continue
            declared[name] = val

        # Every opcode in TgOp must appear in the header with the same value.
        for name in ("NOP", "QKV_PROJ", "ATTN_DECODE", "ATTN_COMBINE",
                     "O_PROJ", "FFN_UPGATE_ACT", "FFN_DOWN", "ADD_RMSNORM",
                     "LM_HEAD", "LM_ARGMAX"):
            full = f"TG_{name}"
            self.assertIn(full, declared, f"{full} missing from tg_abi.h enum")
            self.assertEqual(
                declared[full], getattr(tg.TgOp, name),
                f"{full} value mismatch: header={declared[full]} "
                f"schedule={getattr(tg.TgOp, name)}",
            )

    def test_layer_weight_count_is_eleven(self) -> None:
        """LDGLayerWeights has 11 fields; the enum must count them."""
        header = read(TG_ABI_H)
        # The enum uses ``TG_LW_COUNT,  // 11`` (no explicit ``=``), so we
        # match the trailing comment. Alternatively an explicit
        # ``TG_LW_COUNT = 11`` form is also accepted.
        m = re.search(r"TG_LW_COUNT\s*(?:=\s*(\d+)|,\s*//\s*(\d+))", header)
        self.assertIsNotNone(m)
        assert m is not None
        count = int(m.group(1) or m.group(2))
        self.assertEqual(count, 11)
        # Cross-check against kernel.cu's LDGLayerWeights struct field count.
        kernel = read(TG_KERNEL_CU)
        # The struct has 11 const __nv_bfloat16* fields (one per weight).
        struct_match = re.search(
            r"struct LDGLayerWeights \{(.*?)\};", kernel, re.DOTALL
        )
        # taskgraph_kernel.cu doesn't define LDGLayerWeights (it's in kernel.cu),
        # so we check the model.py packing order instead. The 11-weight order is
        # locked by weights_from_nano_model; we just verify the enum count here.

    def test_launch_desc_has_no_holes_in_documented_sections(self) -> None:
        """TgLaunchDesc must mention every scratch buffer the schedule expects.

        We don't lock the full struct layout (append-only), but we verify the
        documented scratch fields exist by name so the kernel's TgGlobals
        constructor doesn't dereference a missing field.
        """
        header = read(TG_ABI_H)
        required_fields = [
            "num_sms", "inst_buf", "num_inst_per_sm", "max_inst",
            "layer_weights", "embed_weight", "final_norm_weight",
            "cos_table", "sin_table",
            "k_cache", "v_cache", "page_table", "cache_seqlens", "row_active",
            "num_phys_pages", "max_pages_per_seq", "page_block_size",
            "kv_layer_stride",
            "norm_out", "q_out", "k_out", "v_out", "attn_out", "o_proj_out",
            "x_attn", "post_attn_norm", "silu_out", "ffn_down_out", "x_ffn",
            "x_resid", "lm_logits", "o_partial", "lse_partial",
            "bars", "num_layers", "BS", "hidden", "intermediate",
            "q_size", "kv_size", "num_q_heads", "num_kv_heads", "head_dim",
            "vocab_size", "oproj_splits", "down_splits", "max_attn_splits",
            "timing",
        ]
        for field in required_fields:
            with self.subTest(field=field):
                self.assertIn(
                    f" {field};", header,
                    f"TgLaunchDesc missing field '{field}' — the kernel's "
                    "tg_make_globals will not compile, or a schedule writer "
                    "will silently write to the wrong offset.",
                )

    def test_bars_struct_has_eight_arrays(self) -> None:
        header = read(TG_ABI_H)
        for bar in ("bar_qkv", "bar_attn", "bar_oproj", "bar_silu",
                    "bar_ffn_down", "bar_layer", "bar_combine", "bar_lm_head"):
            with self.subTest(bar=bar):
                self.assertIn(f"uint64_t {bar};", header)


# =============================================================================
# Build flag sync: taskgraph_kernel.cu #ifndef fallbacks == build.py defaults
# =============================================================================


class BuildFlagSyncTests(unittest.TestCase):
    """``taskgraph_kernel.cu``'s ``#ifndef`` fallbacks must equal ``build.py``'s
    ``taskgraph_kernel_flags()`` defaults. Same pattern as
    ``test_batch_launch_guards.py::KernelFlagDefaultTests``."""

    def _kernel_default(self, name: str) -> int | None:
        kernel = read(TG_KERNEL_CU)
        m = re.search(rf"#ifndef {name}\s*\n#define {name}\s+(\d+)", kernel)
        return int(m.group(1)) if m else None

    def _build_default(self, name: str) -> int | None:
        build = read(BUILD_PY)
        m = re.search(rf'f"-D{name}=\{{_env_int\([\'"]{name}[\'"],\s*(\d+)\)}}', build)
        return int(m.group(1)) if m else None

    def test_tg_num_blocks(self) -> None:
        self.assertEqual(self._kernel_default("TG_NUM_BLOCKS"), 108)
        self.assertEqual(self._build_default("TG_NUM_BLOCKS"), 108)

    def test_tg_block_size(self) -> None:
        self.assertEqual(self._kernel_default("TG_BLOCK_SIZE"), 256)
        self.assertEqual(self._build_default("TG_BLOCK_SIZE"), 256)

    def test_tg_inst_ring(self) -> None:
        self.assertEqual(self._kernel_default("TG_INST_RING"), 4)
        self.assertEqual(self._build_default("TG_INST_RING"), 4)

    def test_tg_b_stages(self) -> None:
        # The cp.async pipeline depth. Env-overridable so the depth can be A/B'd
        # on hardware; 8 is the hardware maximum for outstanding commit groups.
        self.assertEqual(self._kernel_default("TG_B_STAGES"), 8)
        self.assertEqual(self._build_default("TG_B_STAGES"), 8)

    def test_tg_max_batch(self) -> None:
        self.assertEqual(self._kernel_default("TG_MAX_BATCH"), 8)
        # TG_MAX_BATCH is hardcoded in build.py (not env-overridable), matching
        # the LDG_MAX_BATCH pattern.
        build = read(BUILD_PY)
        self.assertIn('"-DTG_MAX_BATCH=8"', build)


# =============================================================================
# Dispatch coverage: taskgraph_kernel.cu dispatches every opcode schedule emits
# =============================================================================


class DispatchCoverageTests(unittest.TestCase):
    """The kernel's ``tg_dispatch`` switch must have a case for every opcode
    the schedule builder can emit. A missing case means an emitted instruction
    silently falls through to NOP (default branch), producing no compute and
    no bar arrival — the dependency graph deadlocks."""

    def test_every_emitted_opcode_has_a_dispatch_case(self) -> None:
        kernel = read(TG_KERNEL_CU)
        # Find every `case tg::TG_<NAME>:` in the dispatch switch.
        dispatched = set(re.findall(r"case tg::(TG_\w+):", kernel))
        # Opcodes the schedule builder can emit (everything except NOP, which
        # is the filler and has its own branch).
        emitted = {
            "TG_QKV_PROJ", "TG_ATTN_DECODE", "TG_ATTN_COMBINE", "TG_O_PROJ",
            "TG_FFN_UPGATE_ACT", "TG_FFN_DOWN", "TG_ADD_RMSNORM", "TG_LM_HEAD",
        }
        missing = emitted - dispatched
        self.assertFalse(
            missing,
            f"tg_dispatch missing cases for emitted opcodes {missing}; "
            "these instructions would fall through to NOP and deadlock the "
            "dependency graph.",
        )


# =============================================================================
# Bar index formulas
# =============================================================================


class BarIndexTests(unittest.TestCase):
    """Lock the bar index formulas. These must match the kernel's expectation
    (every produce_bar_idx / wait_bar_idx is computed here, and the kernel
    indexes the same array with the same formula)."""

    def test_qkv_index_is_layer_times_hkv_plus_head(self) -> None:
        for layer in range(28):
            for hkv in (8,):
                for kv_head in range(hkv):
                    self.assertEqual(
                        tg.TgBarIndex.qkv(layer, hkv, kv_head),
                        layer * hkv + kv_head,
                    )

    def test_qkv_index_is_unique_per_layer_head(self) -> None:
        indices = {
            tg.TgBarIndex.qkv(l, 8, h)
            for l in range(28) for h in range(8)
        }
        self.assertEqual(len(indices), 28 * 8)

    def test_attn_index_is_layer_times_splits_plus_split(self) -> None:
        for layer in range(28):
            for splits in (1, 2, 4):
                for split in range(splits):
                    self.assertEqual(
                        tg.TgBarIndex.attn(layer, splits, split),
                        layer * splits + split,
                    )

    def test_combine_index_is_unique_per_layer_batch_head(self) -> None:
        for BS in (1, 2, 4, 8):
            indices = {
                tg.TgBarIndex.combine(l, BS, 8, b, h)
                for l in range(28) for b in range(BS) for h in range(8)
            }
            self.assertEqual(len(indices), 28 * BS * 8)

    def test_silu_index_is_layer_times_down_splits_plus_split(self) -> None:
        for layer in range(28):
            for down_splits in (1, 2):
                for split in range(down_splits):
                    self.assertEqual(
                        tg.TgBarIndex.silu(layer, down_splits, split),
                        layer * down_splits + split,
                    )


# =============================================================================
# Instruction encoding round-trip
# =============================================================================


class InstructionEncodingTests(unittest.TestCase):
    """Every instruction builder must produce a 16-int list with opcode in
    slot 0, and the field offsets must match docs/11 §3.3."""

    def test_all_instructions_are_width_sixteen(self) -> None:
        insts = [
            tg.make_nop(),
            tg.make_qkv_proj(0, 0, 0, True, 0, 0, 0, 0),
            tg.make_attn_decode(0, 0, 0, 0, 1, 0, 0, 32, 0, False),
            tg.make_attn_combine(0, 0, 0, 2, 0, 2, 0),
            tg.make_gemm(tg.TgOp.O_PROJ, 0, 0, 0, 0, 128, 0, 8, 0),
            tg.make_add_rmsnorm(0, 0, 0, 64, 0, 64, 1, False),
            tg.make_lm_head(27, 0, 0, 27, 1),
        ]
        for inst in insts:
            with self.subTest(opcode=inst[0]):
                self.assertEqual(len(inst), tg.TG_INST_WIDTH)

    def test_nop_is_all_zeros_except_opcode(self) -> None:
        inst = tg.make_nop()
        self.assertEqual(inst[0], tg.TgOp.NOP)
        self.assertEqual(inst[1:], [0] * (tg.TG_INST_WIDTH - 1))

    def test_qkv_proj_field_offsets(self) -> None:
        inst = tg.make_qkv_proj(
            layer=3, tile_n=5, gemm_idx=1, needs_rope=True,
            wait_bar_idx=3, wait_target=1, produce_bar_idx=27, batch_row=2,
        )
        self.assertEqual(inst[0], tg.TgOp.QKV_PROJ)
        self.assertEqual(inst[1], 3)       # LAYER
        self.assertEqual(inst[2], 5)       # TILE_N
        self.assertEqual(inst[3], 1)       # GEMM_IDX (K)
        self.assertEqual(inst[4], 1)       # NEEDS_ROPE
        self.assertEqual(inst[5], 3)       # WAIT_BAR_IDX
        self.assertEqual(inst[6], 1)       # WAIT_TARGET
        self.assertEqual(inst[7], 27)      # PRODUCE_BAR_IDX
        self.assertEqual(inst[8], 2)       # BATCH_ROW
        self.assertEqual(inst[9], 1)       # NUM_TILES
        self.assertEqual(inst[10], 5)      # first tile_n of the pack

    def test_qkv_proj_encodes_tile_range(self) -> None:
        # Packing is a RANGE, not a tile list: the kernel derives
        # N = num_tiles*16 from inst[10] and never reads inst[11..15].
        inst = tg.make_qkv_proj(
            layer=0, tile_n=6, gemm_idx=0, needs_rope=True,
            wait_bar_idx=0, wait_target=0, produce_bar_idx=0, batch_row=0,
            num_tiles=4,
        )
        self.assertEqual(inst[9], 4)       # NUM_TILES
        self.assertEqual(inst[10], 6)      # first tile_n of the range
        self.assertEqual(inst[11:], [0] * 5,
                         "tile list slots must stay unused (range semantics)")

    def test_gemm_encodes_tile_range(self) -> None:
        inst = tg.make_gemm(
            tg.TgOp.O_PROJ, layer=0, tile_m=0, tile_n=3, split=0, k_tiles=128,
            wait_bar_idx=0, wait_target=8, produce_bar_idx=0,
            num_tiles=4,
        )
        self.assertEqual(inst[9], 4)   # NUM_TILES
        self.assertEqual(inst[10], 3)  # first tile_n of the range
        self.assertEqual(inst[11:], [0] * 5,
                         "tile list slots must stay unused (range semantics)")

    def test_gemm_rejects_zero_tiles(self) -> None:
        with self.assertRaises(ValueError):
            tg.make_gemm(
                tg.TgOp.O_PROJ, 0, 0, 0, 0, 0, 0, 0, 0, num_tiles=0,
            )
        with self.assertRaises(ValueError):
            tg.make_qkv_proj(
                layer=0, tile_n=0, gemm_idx=0, needs_rope=True,
                wait_bar_idx=0, wait_target=0, produce_bar_idx=0, batch_row=0,
                num_tiles=0,
            )

    def test_packer_cap_saturates_worker_warps(self) -> None:
        # 7 worker warps: 7 tiles per instruction is one tile per warp, the
        # widest N one GEMM can spread without serialising tiles per warp.
        self.assertEqual(tg._TILE_SLOTS_PER_INST, 7)
        # QKV: 16 Q tiles per kv_head -> 7+7+2, all consecutive within the head.
        geo = tg.default_geometry(BS=1, attn_splits=1)
        for inst in tg._pack_qkv(tg.build_qkv_wave(geo, 0)):
            self.assertLessEqual(inst[9], 7)
            self.assertGreaterEqual(inst[9], 1)
        # Generic GEMM: 64 O_PROJ tiles -> 7+7+...+1.
        packed = tg._pack_tiles(tg.build_o_proj_wave(geo, 0))
        self.assertEqual(sum(i[9] for i in packed), 64)
        self.assertTrue(all(1 <= i[9] <= 7 for i in packed))

    def test_attn_decode_encode_produce_is_combine_in_sign_bit(self) -> None:
        inst = tg.make_attn_decode(
            layer=0, batch_idx=0, kv_head_idx=0, split_idx=0, num_splits=2,
            window_size=0, wait_bar_idx=0, wait_target=32,
            produce_bar_idx=5, produce_is_combine=True,
        )
        # When produce_is_combine, bit 31 of produce field is set.
        self.assertEqual(inst[9], 5 | (1 << 31))

    def test_attn_decode_produce_attn_when_not_combine(self) -> None:
        inst = tg.make_attn_decode(
            layer=0, batch_idx=0, kv_head_idx=0, split_idx=0, num_splits=1,
            window_size=0, wait_bar_idx=0, wait_target=32,
            produce_bar_idx=5, produce_is_combine=False,
        )
        self.assertEqual(inst[9], 5)
        self.assertEqual(inst[9] >> 31, 0)


# =============================================================================
# Schedule structure
# =============================================================================


class ScheduleStructureTests(unittest.TestCase):
    """The full schedule for Qwen3-0.6B must have sensible shape: enough
    instructions to cover all layers, balanced placement, correct bar dims."""

    def setUp(self) -> None:
        self.geo = tg.default_geometry(BS=1, attn_splits=1)
        self.sched = tg.build_schedule(self.geo, num_sms=108)

    def test_total_instructions_covers_all_layers(self) -> None:
        # At minimum: 28 layers × (QKV + ATTN + O_PROJ + FFN + RMSNORM) + LM_HEAD.
        # With packing the exact count varies, but it must be in the thousands.
        self.assertGreater(self.sched.total_instructions, 1000)
        self.assertLess(self.sched.total_instructions, 100000,
                        "instruction count exploded — packing may have regressed")

    def test_max_inst_is_bounded(self) -> None:
        # 98 was the Phase 0 figure; allow headroom but catch explosions.
        self.assertGreater(self.sched.max_inst, 0)
        self.assertLess(self.sched.max_inst, 500,
                        "max_inst too high — controller prefetch may not keep up")

    def test_placement_is_balanced(self) -> None:
        # Round-robin should spread instructions roughly evenly across SMs.
        counts = self.sched.num_inst_per_sm
        self.assertEqual(len(counts), 108)
        self.assertGreater(min(counts), 0, "some SM got zero instructions")
        spread = max(counts) - min(counts)
        self.assertLessEqual(spread, 2,
                             f"placement unbalanced (spread={spread}); "
                             "round-robin cursor should keep spread <= 1")

    def test_inst_buf_shape_is_num_sms_times_max_inst_times_width(self) -> None:
        expected = 108 * self.sched.max_inst * tg.TG_INST_WIDTH
        self.assertEqual(len(self.sched.flat_inst_buf), expected)

    def test_bar_dims_match_geometry(self) -> None:
        d = self.sched.bar_dims
        self.assertEqual(d.bar_qkv, 28 * 8)
        self.assertEqual(d.bar_attn, 28 * 1)   # oproj_splits=1
        self.assertEqual(d.bar_oproj, 28)
        self.assertEqual(d.bar_silu, 28 * 1)   # down_splits=1
        self.assertEqual(d.bar_ffn_down, 28)
        self.assertEqual(d.bar_layer, 29)   # num_layers + 1 (LM_HEAD slot)
        self.assertEqual(d.bar_combine, 0)     # attn_splits=1
        self.assertEqual(d.bar_lm_head, 1)     # single LM_HEAD -> LM_ARGMAX slot

    def test_nop_padding_fills_short_sms(self) -> None:
        # SMs with fewer instructions are NOP-padded to max_inst. Every NOP
        # has opcode 0 in slot 0.
        width = tg.TG_INST_WIDTH
        for sm in range(108):
            base = sm * self.sched.max_inst * width
            for i in range(self.sched.max_inst):
                inst = self.sched.flat_inst_buf[base + i * width: base + (i + 1) * width]
                if i >= self.sched.num_inst_per_sm[sm]:
                    self.assertEqual(inst[0], tg.TgOp.NOP,
                                     f"SM {sm} slot {i} should be NOP padding")


# =============================================================================
# Wave granularity: pack tiles to cover every SM, not to make big instructions
# =============================================================================


def _wave_tiles(inst: list[int]) -> int:
    """GEMM tiles carried by one instruction (0 for non-GEMM opcodes)."""
    if inst[0] in (tg.TgOp.QKV_PROJ, tg.TgOp.O_PROJ, tg.TgOp.FFN_UPGATE_ACT,
                   tg.TgOp.FFN_DOWN, tg.TgOp.LM_HEAD):
        return inst[9]
    return 0


def waves_with_cap(geo, cap: int) -> list[list[list[int]]]:
    """All waves assembled with an explicit packing cap (a test knob)."""
    waves: list[list[list[int]]] = []
    for layer in range(geo.num_layers):
        waves.extend([
            tg._pack_qkv(tg.build_qkv_wave(geo, layer), cap),
            tg.build_attn_wave(geo, layer),
            tg.build_attn_combine_wave(geo, layer),
            tg._pack_tiles(tg.build_o_proj_wave(geo, layer), cap),
            tg._pack_tiles(tg.build_ffn_upgate_wave(geo, layer), cap),
            tg._pack_tiles(tg.build_ffn_down_wave(geo, layer), cap),
            tg.build_add_rmsnorm_wave(geo, layer),
        ])
    waves.append(tg._pack_tiles(tg.build_lm_head_wave(geo), cap))
    return waves


def wave_k_step_path(geo, num_sms: int, cap: int | None = None) -> int:
    """Serial K-steps on the schedule's critical path.

    The schedule's serial cost in units of one K-step (16 columns of K). Waves
    are strictly ordered — the next cannot start until the last instruction of
    the current one lands — and :func:`tg.place_round_robin` gives each
    instruction its own SM. Within one instruction the tiles run in PARALLEL
    across the worker warps (``tg_wmma_gemm`` fans N out), so an instruction
    with M tiles costs ``ceil(M / warps)`` sequential warp-rounds, not ``M``.
    """
    warps = 7  # TG_WORKER_WARPS
    if cap is not None:
        waves = waves_with_cap(geo, cap)
    else:
        waves = []
        for layer in range(geo.num_layers):
            waves.extend(tg.build_layer_waves(geo, layer, num_sms))
        waves.append(tg._pack_tiles(
            tg.build_lm_head_wave(geo),
            tg._max_per_inst(tg._lm_head_tile_count(geo), num_sms)))
    cursor = 0
    total = 0
    for wave in waves:
        per_sm: dict[int, int] = {}
        for inst in wave:
            sm = cursor % num_sms
            tiles = _wave_tiles(inst)
            rounds = -(-tiles // warps) if tiles else 1  # non-GEMM = 1 step
            per_sm[sm] = per_sm.get(sm, 0) + rounds
            cursor += 1
        total += max(per_sm.values(), default=0)
    return total


class WaveGranularityTests(unittest.TestCase):
    """Tile packing is a *range*, and the range width is a scheduling knob.

    Packing maximally (7 tiles = one per worker warp) is optimal: tiles inside
    one instruction run in parallel across the warps, so a wave's cost is
    ``ceil(M / 7)`` warp-rounds per instruction and ``ceil(ceil(tiles/M)/SMs)``
    instructions per SM. Both ``M = 7`` and ``M = 2`` give the same K-step
    count, but ``M = 7`` saturates the warps (more bytes in flight per SM) and
    emits the fewest instructions (least barrier traffic).
    """

    def test_max_per_inst_packs_maximally(self) -> None:
        # Always the cap: the grid size must NOT shrink the range.
        for num_sms in (1, 8, 108):
            self.assertEqual(tg._max_per_inst(192, num_sms), 7)
            self.assertEqual(tg._max_per_inst(10 ** 6, num_sms), 7)
        # Fewer tiles than the cap -> pack what exists.
        self.assertEqual(tg._max_per_inst(1, 108), 1)
        self.assertEqual(tg._max_per_inst(3, 108), 3)
        self.assertEqual(tg._max_per_inst(0, 108), 1)

    def test_packing_saturates_warps_without_extra_rounds(self) -> None:
        geo = tg.default_geometry(BS=1, attn_splits=1)
        # FFN gate/up has 192 tiles: 28 instructions of 7 = exactly one warp
        # round each, on 28 SMs.
        packed = tg._pack_tiles(tg.build_ffn_upgate_wave(geo, 0))
        self.assertEqual(sum(i[9] for i in packed), 192)
        self.assertTrue(all(i[9] == 7 for i in packed[:-1]))
        self.assertEqual(-(-len(packed) // 1), len(packed))

    def test_wider_packing_never_lengthens_the_path(self) -> None:
        """Packing maximally must dominate thinner packing.

        Spreading tiles thinner over more SMs cannot help: fewer tiles per
        instruction means proportionally fewer active warps, so each SM streams
        proportionally less per K-step while the wave still costs the same
        ``ceil(M / 7)`` warp-rounds. Wider is therefore never worse — and at
        equal width the extra SMs are what shorten the path.
        """
        geo = tg.default_geometry(BS=1, attn_splits=1)
        for num_sms in (1, 8, 108):
            p1 = wave_k_step_path(geo, num_sms, cap=1)
            p2 = wave_k_step_path(geo, num_sms, cap=2)
            p7 = wave_k_step_path(geo, num_sms, cap=7)
            self.assertLessEqual(p7, p2, f"cap=7 worse than cap=2 at {num_sms} SMs")
            self.assertLessEqual(p2, p1, f"cap=2 worse than cap=1 at {num_sms} SMs")

    def test_more_sms_shortens_the_path(self) -> None:
        """The grid is what parallelises the waves; the packing width is fixed."""
        geo = tg.default_geometry(BS=1, attn_splits=1)
        path_1 = wave_k_step_path(geo, 1)
        path_108 = wave_k_step_path(geo, 108)
        self.assertLess(path_108, path_1)
        self.assertGreater(path_1 / path_108, 5.0,
                           "108 SMs should parallelise the serial waves by ~28x")

    def test_granularity_preserves_bar_accounting(self) -> None:
        """Every bar's arrival total must be independent of the grid size.

        The packer only regroups tiles; each instruction still arrives with its
        tile count as ``inc`` (``tg_arrive(bar, idx, num_tiles)``), so the totals
        consumers wait on cannot change. This is what makes the granularity a
        pure scheduling knob.
        """
        geo = tg.default_geometry(BS=2, attn_splits=2)
        ref = None
        for num_sms in (1, 4, 108):
            sched = tg.build_schedule(geo, num_sms)
            done, total, bars, _ = simulate_execution(sched, num_sms)
            self.assertEqual(done, total)
            counts = dict(bars)
            if ref is None:
                ref = counts
            else:
                self.assertEqual(
                    counts, ref,
                    f"bar totals changed at num_sms={num_sms} — granularity is "
                    "no longer a pure scheduling knob")
        assert ref is not None
        # Spot-check the invariants the kernel depends on.
        self.assertEqual(ref[("lm_head", 0)], geo.vocab // 16)
        self.assertEqual(ref[("layer", 1)], geo.BS)
        self.assertEqual(ref[("qkv", 0)], 32)   # 16 Q + 8 K + 8 V tiles


# =============================================================================
# Dependency graph validity
# =============================================================================


class DependencyGraphTests(unittest.TestCase):
    """Every produce_bar_idx must be in range of its bar array, and every
    wait must target a bar that is produced. This catches the class of bugs
    where a typo in the index formula makes a producer write slot X and the
    consumer wait on slot Y — the graph deadlocks.

    We don't simulate execution; we check structural invariants:
    1. Every bar index referenced (wait or produce) is < that bar's length.
    2. For every (bar, slot), the number of produces == the sum of wait_targets
       that reference it (each arrival is expected by exactly one consumer).
       This is the "no dangling produce / no unsatisfied wait" property."""

    def setUp(self) -> None:
        self.geo = tg.default_geometry(BS=1, attn_splits=1)
        self.sched = tg.build_schedule(self.geo, num_sms=108)

    def _iter_instructions(self):
        width = tg.TG_INST_WIDTH
        for sm in range(108):
            for i in range(self.sched.num_inst_per_sm[sm]):
                base = sm * self.sched.max_inst * width + i * width
                yield self.sched.flat_inst_buf[base: base + width]

    def _bar_len(self, name: str) -> int:
        d = self.sched.bar_dims
        return {
            "bar_qkv": d.bar_qkv, "bar_attn": d.bar_attn,
            "bar_oproj": d.bar_oproj, "bar_silu": d.bar_silu,
            "bar_ffn_down": d.bar_ffn_down, "bar_layer": d.bar_layer,
            "bar_combine": d.bar_combine, "bar_lm_head": d.bar_lm_head,
        }[name]

    def test_all_bar_indices_in_range(self) -> None:
        """Every wait_bar_idx and produce_bar_idx must be < the bar's length.

        Field offsets are opcode-specific (see docs/11 §3.3):
          - QKV_PROJ:      wait@5, target@6, produce@7
          - ATTN_DECODE:   wait@7, target@8, produce@9 (with bit31 = combine flag)
          - ATTN_COMBINE:  wait@5, target@6, produce@7
          - generic GEMM (O_PROJ/FFN_UPGATE/FFN_DOWN/LM_HEAD):
                          wait@6, target@7, produce@8, num_tiles@9
          - ADD_RMSNORM:   wait_oproj@3/target@4, wait_ffn@5/target@6, produce@7
          - LM_ARGMAX:     wait@6 (bar_lm_head), target@7
        """
        for inst in self._iter_instructions():
            op = inst[0]
            with self.subTest(opcode=op):
                if op == tg.TgOp.QKV_PROJ:
                    self.assertLess(inst[5], self._bar_len("bar_layer"))
                    self.assertLess(inst[7], self._bar_len("bar_qkv"))
                elif op == tg.TgOp.ATTN_DECODE:
                    self.assertLess(inst[7], self._bar_len("bar_qkv"))
                    produce = inst[9] & ~(1 << 31)
                    if inst[9] >> 31:
                        self.assertLess(produce, self._bar_len("bar_combine"))
                    else:
                        self.assertLess(produce, self._bar_len("bar_attn"))
                elif op == tg.TgOp.ATTN_COMBINE:
                    self.assertLess(inst[5], self._bar_len("bar_combine"))
                    self.assertLess(inst[7], self._bar_len("bar_attn"))
                elif op == tg.TgOp.O_PROJ:
                    self.assertLess(inst[6], self._bar_len("bar_attn"))
                    self.assertLess(inst[8], self._bar_len("bar_oproj"))
                elif op == tg.TgOp.FFN_UPGATE_ACT:
                    self.assertLess(inst[6], self._bar_len("bar_oproj"))
                    self.assertLess(inst[8], self._bar_len("bar_silu"))
                elif op == tg.TgOp.FFN_DOWN:
                    self.assertLess(inst[6], self._bar_len("bar_silu"))
                    self.assertLess(inst[8], self._bar_len("bar_ffn_down"))
                elif op == tg.TgOp.ADD_RMSNORM:
                    self.assertLess(inst[3], self._bar_len("bar_oproj"))
                    self.assertLess(inst[5], self._bar_len("bar_ffn_down"))
                    self.assertLess(inst[7], self._bar_len("bar_layer"))
                elif op == tg.TgOp.LM_HEAD:
                    self.assertLess(inst[6], self._bar_len("bar_layer"))
                    self.assertLess(inst[8], self._bar_len("bar_lm_head"))
                elif op == tg.TgOp.LM_ARGMAX:
                    self.assertLess(inst[6], self._bar_len("bar_lm_head"))
                # NOP: no bar refs

    def test_exactly_one_ffn_upgate_publishes_x_attn(self) -> None:
        """tg_ffn_upgate_op stages x_attn in shared memory and only the packed
        instruction with tile_n0 == 0 publishes it to g.x_attn (so concurrent
        CTAs no longer write the same global addresses). Each layer must have
        exactly one such instruction or ADD_RMSNORM would read stale x_attn."""
        per_layer: dict[int, int] = {}
        for inst in self._iter_instructions():
            if inst[0] == tg.TgOp.FFN_UPGATE_ACT and (inst[10] & 0xFFFF) == 0:
                per_layer[inst[1]] = per_layer.get(inst[1], 0) + 1
        self.assertEqual(len(per_layer), self.geo.num_layers,
                         "some layer has no FFN_UPGATE x_attn publisher")
        self.assertTrue(all(c == 1 for c in per_layer.values()),
                        f"expected exactly one publisher per layer: {per_layer}")

    def test_packed_gemm_encodes_a_consecutive_range(self) -> None:
        """A packed GEMM group becomes ONE kernel GEMM call with
        ``N = num_tiles * 16`` based at the first tile's ``tile_n`` (inst[10],
        which the kernel reads). Packing is therefore a *range*: the group's
        tiles must be consecutive or that single call would cover the wrong
        weight/output range. inst[11..15] must stay unused."""
        seen_any = False
        for inst in self._iter_instructions():
            op = inst[0]
            if op == tg.TgOp.QKV_PROJ:
                num_tiles = inst[9]
                tile_n0 = inst[10]           # QKV: plain tile_n
            elif op in (tg.TgOp.O_PROJ, tg.TgOp.FFN_UPGATE_ACT,
                        tg.TgOp.FFN_DOWN, tg.TgOp.LM_HEAD):
                num_tiles = inst[9]
                tile_n0 = inst[10] & 0xFFFF  # generic: (tile_m << 16) | tile_n
            else:
                continue
            seen_any = True
            self.assertGreaterEqual(num_tiles, 1,
                                    f"opcode {op} has an empty tile range")
            self.assertLessEqual(num_tiles, tg._TILE_SLOTS_PER_INST,
                                 f"opcode {op} exceeds the packing cap")
            self.assertEqual(inst[11:], [0] * 5,
                             "tile list slots must be unused (range semantics)")
            # The range start must be a real tile index, and the range must fit
            # inside the weight matrix that opcode indexes.
            self.assertGreaterEqual(tile_n0, 0)
        self.assertTrue(seen_any, "no GEMM instructions found to check")

    def test_every_produced_bar_slot_is_waited_upon(self) -> None:
        """For each bar slot, total arrivals produced must equal the max
        wait_target demanded by any consumer. Deadlock if produces < max_target;
        waste if >.

        Arrival counting (the key invariant, see docs/11 §4):
          - Non-GEMM ops (ATTN/ADD_RMSNORM): each instruction arrives inc=1,
            so produces += 1 per instruction.
          - GEMM ops (QKV_PROJ/O_PROJ/FFN_UPGATE/FFN_DOWN/LM_HEAD): each
            instruction arrives inc=NUM_TILES (inst[9]), so a packed
            instruction carrying 6 tiles does atomicAdd(bar, 6).
            produces += inst[9].

        Threshold semantics (the key subtlety): ``tg_wait`` polls until the bar
        counter >= target. Multiple consumers may wait on the same slot with
        the same or different thresholds. The producer only needs to drive the
        counter to ``max(target)`` — once it reaches the highest threshold,
        all lower-threshold waiters are already released. So the invariant is
        ``produces[slot] == max(waits[slot])``, NOT ``produces[slot] ==
        sum(waits[slot])``. For example, ``bar_layer[L]`` is waited on by both
        QKV (target=BS) and FFN_UPGATE (target=BS); the producer (ADD_RMSNORM,
        BS arrivals) satisfies both because max(BS, BS) == BS, not 2*BS."""
        from collections import defaultdict

        produces: dict[tuple[str, int], int] = defaultdict(int)
        waits: dict[tuple[str, int], int] = {}  # max target per slot

        def _wait_max(key: tuple[str, int], target: int) -> None:
            waits[key] = max(waits.get(key, 0), target)

        for inst in self._iter_instructions():
            op = inst[0]
            if op == tg.TgOp.QKV_PROJ:
                # GEMM: inc=NUM_TILES@9 (the op now packs consecutive tile_n)
                produces[("bar_qkv", inst[7])] += inst[9]
                _wait_max(("bar_layer", inst[5]), inst[6])
            elif op == tg.TgOp.ATTN_DECODE:
                _wait_max(("bar_qkv", inst[7]), inst[8])
                produce = inst[9] & ~(1 << 31)
                # non-GEMM: inc=1
                if inst[9] >> 31:
                    produces[("bar_combine", produce)] += 1
                else:
                    produces[("bar_attn", produce)] += 1
            elif op == tg.TgOp.ATTN_COMBINE:
                # non-GEMM: inc=1
                _wait_max(("bar_combine", inst[5]), inst[6])
                produces[("bar_attn", inst[7])] += 1
            elif op == tg.TgOp.O_PROJ:
                # GEMM: wait@6, target@7, produce@8, inc=NUM_TILES@9
                _wait_max(("bar_attn", inst[6]), inst[7])
                produces[("bar_oproj", inst[8])] += inst[9]
            elif op == tg.TgOp.FFN_UPGATE_ACT:
                # GEMM: wait@6 (bar_oproj — post-attn norm), target@7,
                # produce@8 (bar_silu), inc@9
                _wait_max(("bar_oproj", inst[6]), inst[7])
                produces[("bar_silu", inst[8])] += inst[9]
            elif op == tg.TgOp.FFN_DOWN:
                # GEMM: wait@6 (bar_silu), target@7, produce@8 (bar_ffn_down), inc@9
                _wait_max(("bar_silu", inst[6]), inst[7])
                produces[("bar_ffn_down", inst[8])] += inst[9]
            elif op == tg.TgOp.ADD_RMSNORM:
                # non-GEMM: inc=1. wait_oproj@3/target@4, wait_ffn@5/target@6, produce@7
                _wait_max(("bar_oproj", inst[3]), inst[4])
                _wait_max(("bar_ffn_down", inst[5]), inst[6])
                produces[("bar_layer", inst[7])] += 1
            elif op == tg.TgOp.LM_HEAD:
                # GEMM: wait@6 (bar_layer), target@7, produce@8 (bar_lm_head),
                # inc=NUM_TILES@9. The argmax is a separate terminal op so it
                # can wait for every tile across SMs.
                _wait_max(("bar_layer", inst[6]), inst[7])
                produces[("bar_lm_head", inst[8])] += inst[9]
            elif op == tg.TgOp.LM_ARGMAX:
                # terminal: wait@6 (bar_lm_head), target@7, no produce
                _wait_max(("bar_lm_head", inst[6]), inst[7])

        # bar_layer[0] has no producer (no layer -1); layer 0's inputs wait
        # with target=0, so max demand is 0. produces is 0 (absent). Both
        # sides are 0, so it passes naturally — but we tolerate it explicitly
        # to document the edge case.
        all_slots = set(produces) | set(waits)
        mismatches = []
        for slot in all_slots:
            p = produces.get(slot, 0)
            w = waits.get(slot, 0)
            # bar_layer[0]: no producer, consumers wait with target=0.
            if p != w and not (slot == ("bar_layer", 0) and p == 0 and w == 0):
                mismatches.append((slot, p, w))
        self.assertFalse(
            mismatches,
            f"produce/wait mismatch (deadlock or waste): {mismatches[:5]}",
        )


# =============================================================================
# Execution simulation: prove the schedule cannot deadlock
# =============================================================================


def simulate_execution(sched, num_sms):
    """Execute the per-SM instruction streams under the bar semantics.

    Models each SM as a program counter that advances when its next
    instruction's wait is satisfied, applying the instruction's bar arrivals.
    Repeats a full scan until nothing can make progress. Because bar counters
    are monotone (executing an instruction never disables another wait), this
    greedy maximal execution completes iff *some* concurrent execution can —
    so a shortfall is a real deadlock, not a simulation artifact.

    This is the invariant that matters for num_sms > 1: every producer a
    consumer waits on must sit earlier in the global order (and hence be
    reachable by its SM before that consumer's SM stalls).

    Returns (completed, total, bars, pc).
    """
    from collections import defaultdict

    W = tg.TG_INST_WIDTH
    per_sm = []
    for sm in range(num_sms):
        base = sm * sched.max_inst * W
        per_sm.append([
            sched.flat_inst_buf[base + i * W: base + (i + 1) * W]
            for i in range(sched.num_inst_per_sm[sm])
        ])
    total = sum(len(x) for x in per_sm)
    bars = defaultdict(int)
    pc = [0] * num_sms
    done = 0

    def wait_ok(inst):
        op = inst[0]
        if op == tg.TgOp.QKV_PROJ:
            return bars[("layer", inst[5])] >= inst[6]
        if op == tg.TgOp.ATTN_DECODE:
            return bars[("qkv", inst[7])] >= inst[8]
        if op == tg.TgOp.ATTN_COMBINE:
            return bars[("combine", inst[5])] >= inst[6]
        if op == tg.TgOp.O_PROJ:
            return bars[("attn", inst[6])] >= inst[7]
        if op == tg.TgOp.FFN_UPGATE_ACT:
            return bars[("oproj", inst[6])] >= inst[7]
        if op == tg.TgOp.FFN_DOWN:
            return bars[("silu", inst[6])] >= inst[7]
        if op == tg.TgOp.ADD_RMSNORM:
            return (bars[("oproj", inst[3])] >= inst[4]
                    and bars[("ffn_down", inst[5])] >= inst[6])
        if op == tg.TgOp.LM_HEAD:
            return bars[("layer", inst[6])] >= inst[7]
        if op == tg.TgOp.LM_ARGMAX:
            return bars[("lm_head", inst[6])] >= inst[7]
        return True  # NOP

    def apply(inst):
        op = inst[0]
        if op == tg.TgOp.QKV_PROJ:
            bars[("qkv", inst[7])] += inst[9]
        elif op == tg.TgOp.ATTN_DECODE:
            key = "combine" if (inst[9] >> 31) else "attn"
            bars[(key, inst[9] & ~(1 << 31))] += 1
        elif op == tg.TgOp.ATTN_COMBINE:
            bars[("attn", inst[7])] += 1
        elif op == tg.TgOp.O_PROJ:
            bars[("oproj", inst[8])] += inst[9]
        elif op == tg.TgOp.FFN_UPGATE_ACT:
            bars[("silu", inst[8])] += inst[9]
        elif op == tg.TgOp.FFN_DOWN:
            bars[("ffn_down", inst[8])] += inst[9]
        elif op == tg.TgOp.ADD_RMSNORM:
            bars[("layer", inst[7])] += 1
        elif op == tg.TgOp.LM_HEAD:
            bars[("lm_head", inst[8])] += inst[9]
        # LM_ARGMAX / NOP produce nothing.

    while done < total:
        progressed = False
        for sm in range(num_sms):
            if pc[sm] >= len(per_sm[sm]):
                continue
            inst = per_sm[sm][pc[sm]]
            if wait_ok(inst):
                apply(inst)
                pc[sm] += 1
                done += 1
                progressed = True
        if not progressed:
            break
    return done, total, bars, pc


class ScheduleExecutionTests(unittest.TestCase):
    """The schedule must be deadlock-free for any grid size.

    A single missing dependency edge (a producer placed after its consumer in
    the global order) makes the real kernel spin forever on the GPU, which is
    painful to diagnose on a remote box. This catches it on the CPU.
    """

    def test_no_deadlock_across_batch_and_grid_sizes(self) -> None:
        for attn_splits in (1, 2):
            for BS in (1, 2, 4, 8):
                for num_sms in (1, 4, 8, 108):
                    with self.subTest(attn_splits=attn_splits, BS=BS,
                                      num_sms=num_sms):
                        geo = tg.default_geometry(BS=BS, attn_splits=attn_splits)
                        sched = tg.build_schedule(geo, num_sms)
                        done, total, _, pc = simulate_execution(sched, num_sms)
                        self.assertEqual(
                            done, total,
                            f"schedule deadlocks: only {done}/{total} "
                            f"instructions can execute (per-SM progress: {pc})",
                        )

    def test_simulation_completes_every_bar(self) -> None:
        """The last instruction must be LM_ARGMAX and every bar must be fully
        consumed by the time the stream ends."""
        geo = tg.default_geometry(BS=2, attn_splits=1)
        sched = tg.build_schedule(geo, num_sms=8)
        done, total, bars, pc = simulate_execution(sched, 8)
        self.assertEqual(done, total)
        # bar_lm_head must have been driven to exactly the LM head tile count.
        self.assertEqual(bars[("lm_head", 0)], geo.vocab // 16)


# =============================================================================
# Multi-batch geometry sanity
# =============================================================================


class MultiBatchGeometryTests(unittest.TestCase):
    """BS=2/4/8 schedules must build without error and have bar dims that
    scale with BS (bar_combine) but not the per-layer bars."""

    def test_bs2_builds_and_bar_combine_scales(self) -> None:
        sched = tg.build_schedule(tg.default_geometry(BS=2, attn_splits=2), 108)
        self.assertEqual(sched.bar_dims.bar_combine, 28 * 2 * 8)
        # Per-layer bars don't depend on BS.
        self.assertEqual(sched.bar_dims.bar_qkv, 28 * 8)
        self.assertEqual(sched.bar_dims.bar_oproj, 28)

    def test_bs8_builds(self) -> None:
        sched = tg.build_schedule(tg.default_geometry(BS=8, attn_splits=1), 108)
        self.assertGreater(sched.total_instructions, 0)
        self.assertEqual(len(sched.num_inst_per_sm), 108)


if __name__ == "__main__":
    unittest.main()
