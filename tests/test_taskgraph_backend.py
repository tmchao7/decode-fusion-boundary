"""Task-graph decode backend plumbing (``decode_backend="megakernel"``).

Everything here is CUDA-free: the task-graph *policy* (admission, batch ceiling)
and the *plumbing* that decides which decoder an engine builds. The kernel
itself is exercised by ``tests/test_taskgraph_*.py`` on a GPU box.

These tests exist because the failure mode of a missed gate is silent: an
unrecognised ``decode_backend`` string makes the engine skip the megakernel
entirely and run every row on torch, which a benchmark then reports as a
megakernel result (docs/05 §6, the "fallback row is not a megakernel result"
rule). Each gate is pinned individually.
"""

import ast
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nanovllm.backends.megakernel import (  # noqa: E402
    TG_MAX_BATCH,
    MegakernelBackend,
    is_megakernel_backend,
    taskgraph_compatibility,
)
from nanovllm.backends.megakernel.taskgraph import (  # noqa: E402
    MAX_ATTN_SPLITS,
    max_pages_per_seq,
    resolve_num_sms,
)


def make_hf_config(**overrides):
    values = dict(
        model_type="qwen3",
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        vocab_size=151936,
        head_dim=128,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_config(**overrides):
    values = dict(
        decode_backend="megakernel",
        tensor_parallel_size=1,
        max_num_seqs=8,
        max_model_len=2048,
        hf_config=make_hf_config(),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def greedy_seq():
    return SimpleNamespace(is_greedy=True, temperature=0.0)


class TaskGraphPolicyTests(unittest.TestCase):
    def test_accepts_batch_1_through_the_compile_time_ceiling(self):
        config = make_config(max_num_seqs=TG_MAX_BATCH)
        for batch in range(1, TG_MAX_BATCH + 1):
            with self.subTest(batch=batch):
                decision = taskgraph_compatibility(
                    config, [greedy_seq()] * batch, is_prefill=False
                )
                self.assertTrue(decision.supported, decision.reason)

    def test_rejects_batch_above_the_compile_time_ceiling(self):
        """TG_MAX_BATCH sizes scratch and bars, so an over-batch launch would
        index past both. It must be refused before the kernel runs."""

        config = make_config(max_num_seqs=64)
        decision = taskgraph_compatibility(
            config, [greedy_seq()] * (TG_MAX_BATCH + 1), is_prefill=False
        )
        self.assertFalse(decision.supported)
        self.assertIn(str(TG_MAX_BATCH), decision.reason)

    def test_rejects_prefill(self):
        config = make_config()
        self.assertFalse(
            taskgraph_compatibility(config, [greedy_seq()], is_prefill=True).supported
        )

    def test_rejects_sampling_sequences(self):
        config = make_config()
        seq = SimpleNamespace(is_greedy=False, temperature=0.6)
        decision = taskgraph_compatibility(config, [seq], is_prefill=False)
        self.assertFalse(decision.supported)
        self.assertIn("greedy", decision.reason)

    def test_rejects_tensor_parallel(self):
        config = make_config(tensor_parallel_size=2)
        self.assertFalse(
            taskgraph_compatibility(config, [greedy_seq()], is_prefill=False).supported
        )

    def test_rejects_empty_batch(self):
        config = make_config()
        self.assertFalse(
            taskgraph_compatibility(config, [], is_prefill=False).supported
        )

    def test_does_not_require_a_legacy_batch_decode_opt_in(self):
        """The public Megakernel is task-graph and accepts whole batches."""

        config = make_config(max_num_seqs=8)
        decision = taskgraph_compatibility(
            config, [greedy_seq()] * 4, is_prefill=False
        )
        self.assertTrue(decision.supported, decision.reason)

    def test_batch_limit_is_the_documented_constant(self):
        self.assertEqual(TG_MAX_BATCH, 8)


class BackendDispatchTests(unittest.TestCase):
    """``MegakernelBackend.compatibility`` must route on decode_backend."""

    def test_megakernel_uses_the_taskgraph_policy(self):
        config = make_config(max_num_seqs=64)
        decision = MegakernelBackend.compatibility(
            config, [greedy_seq()] * (TG_MAX_BATCH + 1), is_prefill=False
        )
        self.assertFalse(decision.supported)
        self.assertIn("Task-graph", decision.reason)

    def test_taskgraph_checks_the_qwen_shape_contract(self):
        config = make_config(hf_config=make_hf_config(vocab_size=32000))
        decision = MegakernelBackend.compatibility(
            config, [greedy_seq()], is_prefill=False
        )
        self.assertFalse(decision.supported)
        self.assertIn("unsupported Qwen config", decision.reason)

    def test_taskgraph_rejects_untied_embeddings_and_attention_bias(self):
        for override, expected in (
            ({"tie_word_embeddings": False}, "tie_word_embeddings"),
            ({"attention_bias": True}, "attention_bias"),
        ):
            with self.subTest(field=expected):
                config = make_config(hf_config=make_hf_config(**override))
                decision = MegakernelBackend.compatibility(
                    config, [greedy_seq()], is_prefill=False
                )
                self.assertFalse(decision.supported)
                self.assertIn(expected, decision.reason)

    def test_is_megakernel_backend_covers_only_the_mainline_path(self):
        self.assertTrue(is_megakernel_backend("megakernel"))
        self.assertFalse(is_megakernel_backend("megakernel-tg"))
        self.assertFalse(is_megakernel_backend("torch"))
        self.assertFalse(is_megakernel_backend(None))


class DecoderConstructionTests(unittest.TestCase):
    """The task-graph decoder is the only mainline Megakernel implementation."""

    def test_config_validates_the_taskgraph_backend_name(self):
        source = (REPO_ROOT / "src" / "nanovllm" / "config.py").read_text(
            encoding="utf-8"
        )
        match = re.search(r"VALID_DECODE_BACKENDS\s*=\s*\{([^}]*)\}", source)
        self.assertIsNotNone(match)
        assert match is not None
        values = {item.strip().strip('"\'') for item in match.group(1).split(",")}
        self.assertEqual(values, {"torch", "megakernel"})

    def test_engine_gates_use_is_megakernel_backend(self):
        """``model_runner`` has two gates keyed on the backend string (graph
        capture and decoder construction). Both must go through the shared
        predicate, otherwise adding a backend silently disables the kernel."""

        source = (REPO_ROOT / "src" / "nanovllm" / "engine" / "model_runner.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("is_megakernel_backend(config.decode_backend)", source)
        self.assertNotIn("supports_no_sync_burst", source)
        self.assertNotIn('config.decode_backend != "megakernel"', source)
        self.assertNotIn('config.decode_backend == "megakernel"', source)

    def test_backend_builds_the_taskgraph_decoder_for_the_new_string(self):
        source = (
            REPO_ROOT / "src" / "nanovllm" / "backends" / "megakernel" / "backend.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('== "megakernel-tg"', source)
        self.assertIn("TaskGraphDecoder", source)



class GeometryHelperTests(unittest.TestCase):
    def test_max_pages_matches_the_model_runner_block_table_stride(self):
        """The task-graph page_table stride must be exactly what
        ``prepare_block_tables`` pads to, or the launch writes out of bounds."""

        # ModelRunner.prepare_block_tables pads to max(len(block_table)), and
        # with max_model_len=2048 / block_size=256 that is 8 columns.
        self.assertEqual(max_pages_per_seq(2048, 256), 8)
        self.assertEqual(max_pages_per_seq(2048, 16), 128)
        self.assertEqual(max_pages_per_seq(1, 256), 1)

    def test_max_pages_rejects_a_zero_block_size(self):
        with self.assertRaisesRegex(ValueError, "block_size"):
            max_pages_per_seq(2048, 0)

    def test_attn_split_capacity_matches_default_geometry(self):
        """The kernel indexes o_partial/lse_partial with the max_attn_splits the
        schedule was built with; a mismatch silently corrupts the combine."""

        from nanovllm.backends.megakernel import schedule as tg_schedule

        geo = tg_schedule.default_geometry(BS=1, attn_splits=1)
        self.assertEqual(geo.max_attn_splits, MAX_ATTN_SPLITS)

    def test_attn_splits_above_capacity_is_rejected(self):
        """``MAX_ATTN_SPLITS`` is the o_partial/lse_partial capacity. A schedule
        with more splits than that writes past the buffers."""

        source = (
            REPO_ROOT / "src" / "nanovllm" / "backends" / "megakernel" / "taskgraph.py"
        ).read_text(encoding="utf-8")
        self.assertIn("1 <= self._attn_splits <= MAX_ATTN_SPLITS", source)
        # The schedule builder bakes in its own capacity; the two must agree.
        from nanovllm.backends.megakernel import schedule as tg_schedule

        for attn_splits in (1, 2, MAX_ATTN_SPLITS):
            geo = tg_schedule.default_geometry(BS=1, attn_splits=attn_splits)
            with self.subTest(attn_splits=attn_splits):
                self.assertEqual(geo.max_attn_splits, MAX_ATTN_SPLITS)

    def test_num_sms_defaults_to_the_device_count(self):
        import os

        saved = os.environ.pop("TG_NUM_SMS", None)
        try:
            self.assertEqual(resolve_num_sms(108), 108)
            self.assertEqual(resolve_num_sms(0), 1)
        finally:
            if saved is not None:
                os.environ["TG_NUM_SMS"] = saved

    def test_num_sms_env_override(self):
        import os

        saved = os.environ.get("TG_NUM_SMS")
        os.environ["TG_NUM_SMS"] = "4"
        try:
            self.assertEqual(resolve_num_sms(108), 4)
        finally:
            if saved is None:
                os.environ.pop("TG_NUM_SMS", None)
            else:
                os.environ["TG_NUM_SMS"] = saved


class CompareBackendPlumbingTests(unittest.TestCase):
    """``benchmarks/compare.py`` must know about the task-graph backend.

    compare.py imports torch, so this parses the module instead of importing
    it. The failure this guards is a benchmark table that claims to compare the
    task-graph kernel while every row silently ran torch.
    """

    @classmethod
    def setUpClass(cls):
        cls.source = (REPO_ROOT / "benchmarks" / "compare.py").read_text(
            encoding="utf-8"
        )
        cls.tree = ast.parse(cls.source)

    def _dict_value(self, name):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        return ast.literal_eval(node.value)
        raise AssertionError(f"{name} not found in compare.py")

    def test_backend_configs_include_the_single_mainline_megakernel(self):
        configs = self._dict_value("BACKEND_CONFIGS")
        self.assertEqual(set(configs), {"torch-eager", "torch-cudagraph", "megakernel"})
        entry = configs["megakernel"]
        self.assertEqual(entry["decode_backend"], "megakernel")
        # run_case() passes max_model_len=2048 to LLM() itself; a copy here would
        # be a duplicate keyword argument at engine construction.
        self.assertNotIn("max_model_len", entry)

    def test_megakernel_family_contains_only_the_mainline_backend(self):
        family = self._dict_value("MEGAKERNEL_FAMILY")
        self.assertEqual(family, ("megakernel",))

    def test_default_backends_are_the_docs_13_main_table(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                func = getattr(node.func, "attr", None)
                if func != "add_argument":
                    continue
                names = [
                    arg.value
                    for arg in node.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                ]
                if "--backends" not in names:
                    continue
                default = next(
                    kw.value
                    for kw in node.keywords
                    if kw.arg == "default"
                )
                self.assertEqual(
                    ast.literal_eval(default),
                    "torch-eager,torch-cudagraph,megakernel",
                )
                return
        self.fail("compare.py has no --backends argument")

    def test_batch_ceiling_matches_the_compile_time_constant(self):
        self.assertEqual(
            self._dict_value("MEGAKERNEL_MAX_BATCH"), TG_MAX_BATCH
        )

    def test_grid_mode_exists_for_the_batch_by_context_table(self):
        """docs/13 §1.3 asks for a batch x context grid; without --grid the tool
        can only emit two fixed shapes and the grid is unreachable."""

        self.assertIn("--grid", self.source)
        self.assertIn("--grid-lengths", self.source)


class ServerCommandPlumbingTests(unittest.TestCase):
    """SERVER_COMMANDS.txt must be able to drive the new backend and must
    export both model-path variable names."""

    @classmethod
    def setUpClass(cls):
        cls.text = (REPO_ROOT / "SERVER_COMMANDS.txt").read_text(encoding="utf-8")

    def test_both_model_env_names_are_exported_from_model(self):
        """tools/tests read QWEN3_MODEL_PATH, the engine reads
        NANOVLLM_MODEL_PATH. Only exporting one leaves the other on a stale
        default (or the HF hub with HF_HUB_OFFLINE=1)."""

        self.assertIn('export NANOVLLM_MODEL_PATH="$MODEL"', self.text)
        self.assertIn('export QWEN3_MODEL_PATH="$MODEL"', self.text)

    def test_context_len_is_passed_as_a_list(self):
        """bench_taskgraph.py's --context-len accepts a comma list; passing a
        bare int silently benchmarks one context instead of the grid."""

        for match in re.finditer(r"--context-len\s+([0-9,]+)", self.text):
            with self.subTest(value=match.group(1)):
                # Either a single value or a comma list is fine; what must not
                # happen is a comma list reaching a parser that only takes int.
                self.assertRegex(match.group(1), r"^[0-9]+(,[0-9]+)*$")

    def test_no_ncu_invocation_on_the_taskgraph_modes(self):
        """ncu on a cooperative kernel hangs the GPU (docs/13 §7). Every ncu
        mention must be a comment or a warning, never a live command."""

        for line in self.text.splitlines():
            stripped = line.strip()
            if "ncu" not in stripped:
                continue
            with self.subTest(line=stripped):
                self.assertTrue(
                    stripped.startswith("#") or "echo" in stripped,
                    f"live ncu invocation: {stripped}",
                )


class LaunchArgumentOrderTests(unittest.TestCase):
    """``TaskGraphDecoder.decode_rows`` builds one positional call into
    ``tg_decode_step``. A swapped scratch buffer is not a crash: the kernel
    writes attention output into ``post_attn_norm`` and every token after the
    first is wrong. Parse the binding's declared signature and compare it to the
    order the decoder actually passes.
    """

    @classmethod
    def setUpClass(cls):
        cls.decoder_src = (
            REPO_ROOT / "src" / "nanovllm" / "backends" / "megakernel" / "taskgraph.py"
        ).read_text(encoding="utf-8")
        cls.binding_src = (
            REPO_ROOT
            / "src"
            / "nanovllm"
            / "backends"
            / "megakernel"
            / "csrc"
            / "torch_bindings.cpp"
        ).read_text(encoding="utf-8")

    def _binding_params(self):
        start = self.binding_src.index('"tg_decode_step("')
        end = self.binding_src.index("-> ()", start)
        block = self.binding_src[start:end]
        names = re.findall(r"(?:Tensor|int)\s+(\w+)", block)
        return names

    def _decoder_scratch_and_bars(self):
        start = self.decoder_src.index("self._scratch = (")
        end = self.decoder_src.index(")", start)
        # The first match is the assignment target itself (``self._scratch``).
        scratch = re.findall(r"self\._(\w+)", self.decoder_src[start:end])[1:]
        start = self.decoder_src.index("self._bar_names = (")
        end = self.decoder_src.index(")", start)
        bars = re.findall(r'"(\w+)"', self.decoder_src[start:end])
        return scratch, bars

    def test_scratch_buffers_are_passed_in_the_binding_order(self):
        scratch, _ = self._decoder_scratch_and_bars()
        params = self._binding_params()
        declared = [
            name
            for name in params
            if name
            in (
                "norm_out", "q_out", "k_out", "v_out", "attn_out",
                "o_proj_out", "x_attn", "post_attn_norm", "silu_out",
                "ffn_down_out", "up_out", "x_ffn", "x_resid", "lm_logits",
                "o_partial", "lse_partial",
            )
        ]
        self.assertEqual(scratch, declared)

    def test_bars_are_passed_in_the_binding_order(self):
        _, bars = self._decoder_scratch_and_bars()
        params = self._binding_params()
        declared = [
            name for name in params if name.startswith("bar_")
        ]
        self.assertEqual(bars, declared)

    def test_geometry_tail_matches_the_binding(self):
        """The trailing geometry ints must stay in the declared order."""

        params = self._binding_params()
        tail = params[-13:]
        self.assertEqual(
            tail,
            [
                "num_layers", "BS", "hidden", "intermediate", "q_size",
                "kv_size", "num_q_heads", "num_kv_heads", "head_dim",
                "vocab_size", "oproj_splits", "down_splits", "max_attn_splits",
            ],
        )
        # oproj_splits / down_splits > 1 are still NotImplementedError in the
        # schedule builder, so the decoder must pass 1/1.
        self.assertIn("1, 1, MAX_ATTN_SPLITS,", self.decoder_src)

    def test_bars_are_passed_as_a_dict_in_declared_order(self):
        """``*self._bars.values()`` relies on insertion order, which is
        ``_bar_names``. Building the dict from a set would silently reorder it."""

        self.assertIn("*self._bars.values()", self.decoder_src)
        self.assertIn("for name, size in dims.items()", self.decoder_src)


    def test_no_backend_config_duplicates_an_explicit_engine_keyword(self):
        """Every module that runs the task-graph backend passes
        ``max_model_len=2048`` to ``LLM()`` explicitly. A ``max_model_len`` key
        in its BACKENDS entry becomes a duplicate keyword and raises TypeError
        before the engine is even built -- which looks like "the backend does
        not work" rather than a config bug."""

        for relative in ("benchmarks/compare.py", "tools/correctness_smoke.py"):
            source = (REPO_ROOT / relative).read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                target = node.targets[0]
                if not (
                    isinstance(target, ast.Name) and target.id == "BACKENDS"
                ) and not (
                    isinstance(target, ast.Name)
                    and target.id == "BACKEND_CONFIGS"
                ):
                    continue
                configs = ast.literal_eval(node.value)
                for name, config in configs.items():
                    for duplicate in ("max_model_len", "max_num_seqs"):
                        with self.subTest(
                            module=relative, backend=name, key=duplicate
                        ):
                            self.assertNotIn(duplicate, config)


class BarAllocationTests(unittest.TestCase):
    """The decoder allocates bars from the bucket geometries (BS 1/2/4/8) and
    then reuses them for any batch in 1..TG_MAX_BATCH. That is only safe if the
    bucket max dominates every batch size, for every bar and split count."""

    def test_bucket_max_dominates_every_supported_batch(self):
        from nanovllm.backends.megakernel import schedule as tg_schedule

        bar_names = (
            "bar_qkv", "bar_attn", "bar_oproj", "bar_silu",
            "bar_ffn_down", "bar_layer", "bar_combine", "bar_lm_head",
        )
        for attn_splits in (1, 2, MAX_ATTN_SPLITS):
            buckets = [
                tg_schedule.build_schedule(
                    tg_schedule.default_geometry(BS=bs, attn_splits=attn_splits),
                    num_sms=108,
                ).bar_dims
                for bs in (1, 2, 4, 8)
            ]
            allocated = {
                name: max(getattr(dims, name) for dims in buckets)
                for name in bar_names
            }
            for bs in range(1, TG_MAX_BATCH + 1):
                dims = tg_schedule.build_schedule(
                    tg_schedule.default_geometry(BS=bs, attn_splits=attn_splits),
                    num_sms=108,
                ).bar_dims
                for name in bar_names:
                    with self.subTest(attn_splits=attn_splits, BS=bs, bar=name):
                        self.assertLessEqual(
                            getattr(dims, name),
                            allocated[name],
                            f"{name} for BS={bs} exceeds the bucket allocation",
                        )

    def test_every_supported_geometry_is_deadlock_free(self):
        """The simulator's deadlock proof must cover the batches the decoder can
        actually request, including the non-bucket sizes 3/5/6/7 that the engine
        can hand it."""

        from nanovllm.backends.megakernel import schedule as tg_schedule

        try:
            from tests.test_taskgraph_abi import simulate_execution
        except Exception:  # pragma: no cover - direct execution
            import sys as _sys

            _sys.path.insert(0, str(REPO_ROOT / "tests"))
            from test_taskgraph_abi import simulate_execution  # type: ignore

        for attn_splits in (1, 2):
            for bs in range(1, TG_MAX_BATCH + 1):
                for num_sms in (1, 4, 108, 170):
                    with self.subTest(attn_splits=attn_splits, BS=bs, num_sms=num_sms):
                        sched = tg_schedule.build_schedule(
                            tg_schedule.default_geometry(BS=bs, attn_splits=attn_splits),
                            num_sms=num_sms,
                        )
                        done, total, _, progress = simulate_execution(sched, num_sms)
                        self.assertEqual(
                            done,
                            total,
                            f"schedule deadlocks for BS={bs} attn_splits="
                            f"{attn_splits} num_sms={num_sms}: {done}/{total} "
                            f"(per-SM progress: {progress})",
                        )


class TaskGraphDebugContractTests(unittest.TestCase):
    def test_debug_bounds_is_explicitly_wired(self):
        taskgraph = (
            REPO_ROOT / "src/nanovllm/backends/megakernel/taskgraph.py"
        ).read_text(encoding="utf-8")
        backend = (
            REPO_ROOT / "src/nanovllm/backends/megakernel/backend.py"
        ).read_text(encoding="utf-8")
        config = (REPO_ROOT / "src/nanovllm/config.py").read_text(encoding="utf-8")
        build = (REPO_ROOT / "src/nanovllm/backends/megakernel/build.py").read_text(encoding="utf-8")
        self.assertIn("debug_bounds: bool = False", taskgraph)
        self.assertIn("megakernel_debug_bounds", backend)
        self.assertIn("megakernel_debug_bounds: bool = False", config)
        self.assertIn("NANOVLLM_DEBUG_BOUNDS", build)
        self.assertIn("self._audit_geometry(geo)", taskgraph)


if __name__ == "__main__":
    unittest.main()

class WeightContractSourceTests(unittest.TestCase):
    def test_weight_loader_checks_tied_storage_identity(self):
        source = (
            REPO_ROOT / "src" / "nanovllm" / "backends" / "megakernel" / "weights.py"
        ).read_text(encoding="utf-8")
        self.assertIn("lm_head.weight.data_ptr()", source)
        self.assertIn("embed_tokens.weight.data_ptr()", source)
        self.assertIn("tied lm_head and embed_tokens storage", source)
