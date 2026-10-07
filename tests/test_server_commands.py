"""Validate the commands in SERVER_COMMANDS.txt before they reach the server.

The server command file is the only interface to the A100, and a typo in a
module name or a flag costs a whole session (the model load alone is minutes).
This parses the file, resolves every ``python -m <module>`` invocation, and
checks that the flags it passes actually exist in that module's argparse
definition.  It also checks the docs' promised output filenames, since a rename
in the tool silently breaks the "send back results/a100_verdict.json"
instruction.
"""

import ast
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SERVER_COMMANDS = REPO_ROOT / "SERVER_COMMANDS.txt"
TOOLS_ROOT = REPO_ROOT / "tools"

# A file whose first lines carry one of these markers is a historical record
# (docs/06, docs/07, docs/09) and may reference the archive branch. A single
# line can also opt out by naming the reference as archived/deprecated.
HISTORY_MARKERS = (
    "归档", "历史", "已废弃", "未包含",
    "archive", "archived", "historical", "deprecated",
)
_BANNER_LINES = 12
_TOOL_REFERENCE = re.compile(r"tools/([a-z_0-9]+)\.py|tools\.([a-z_0-9]+)")


def find_dangling_tool_references(
    paths: Iterable[Path],
    *,
    tools_root: Path = TOOLS_ROOT,
    history_markers: tuple[str, ...] = HISTORY_MARKERS,
) -> list[str]:
    """Return ``path:line cites tools.<module>`` for modules this branch lacks.

    Only module-shaped references count: a ``tools/<module>.py`` path or the
    dotted ``tools.<module>`` form. A bare directory mention
    (``tools/benchmarks``), a glob and a struck-through retraction are not
    module references, and a historical banner exempts a whole file.
    """

    dangling: list[str] = []
    for path in paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        if any(marker in "\n".join(lines[:_BANNER_LINES]) for marker in history_markers):
            continue
        for number, line in enumerate(lines, start=1):
            # Prose wraps, so a marker may sit on the line above the citation.
            window = line if number == 1 else lines[number - 2] + line
            if any(marker in window for marker in history_markers):
                continue
            if "~~" in line:  # struck-through: an explicit retraction
                continue
            for match in _TOOL_REFERENCE.finditer(line):
                module = match.group(1) or match.group(2)
                if line[match.end() :].startswith("*"):  # glob, not a module
                    continue
                if not (tools_root / f"{module}.py").is_file():
                    dangling.append(f"{path.name}:{number} cites tools.{module}")
    return dangling

DOCS_09 = REPO_ROOT / "docs" / "08_服务器运行手册.md"

# Modules we own; everything else in the file (pip, nsys, nvidia-smi) is not
# ours to validate.
OWNED_PREFIXES = ("tools.", "benchmarks.", "nanovllm.")


def logical_lines(text: str) -> list[str]:
    """Join backslash continuations so each command is one string."""

    joined: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.endswith("\\"):
            buffer += line[:-1] + " "
            continue
        joined.append((buffer + line).strip())
        buffer = ""
    if buffer:
        joined.append(buffer.strip())
    return joined


def invocations(text: str) -> list[tuple[str, list[str]]]:
    """Return (module, args) for every ``python -m <module>`` command."""

    found: list[tuple[str, list[str]]] = []
    for line in logical_lines(text):
        match = re.search(r"python(?:3(?:\.\d+)?)?\s+-m\s+([A-Za-z_][\w.]*)", line)
        if not match:
            continue
        module = match.group(1)
        if not module.startswith(OWNED_PREFIXES):
            continue
        after = line[match.end():]
        args = [token for token in re.split(r"\s+", after.strip()) if token]
        found.append((module, args))
    return found


def module_source(module: str) -> Path:
    relative = module.replace(".", "/")
    # The installable package lives under src/, the tools/benchmarks packages at
    # the repository root.
    for root in (REPO_ROOT / "src", REPO_ROOT):
        for candidate in (
            root / f"{relative}.py",
            root / relative / "__main__.py",
            root / relative / "__init__.py",
        ):
            if candidate.exists():
                return candidate
    raise AssertionError(f"module {module} referenced by SERVER_COMMANDS has no source")


def module_flags(module: str) -> set[str]:
    """Collect every add_argument destination/flag from a module's AST."""

    tree = ast.parse(module_source(module).read_text(encoding="utf-8"))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "attr", None)
        if name not in ("add_argument", "add_parser"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.startswith("-"):
                    flags.add(arg.value)
    return flags


def external_help_flags(module: str) -> set[str]:
    """Flags accepted by -m <module> itself (e.g. unittest), via --help."""

    proc = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}",
            "PATH": "/usr/bin:/bin",
        },
    )
    if proc.returncode != 0:
        return set()
    return set(re.findall(r"(--[a-z][a-z0-9-]*)", proc.stdout))


class ServerCommandReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = SERVER_COMMANDS.read_text(encoding="utf-8")
        self.calls = invocations(self.text)

    def test_every_referenced_module_has_a_source_file(self) -> None:
        modules = {module for module, _ in self.calls}
        self.assertIn("tools.account_decode", modules)
        for module in sorted(modules):
            if module == "unittest":
                continue
            with self.subTest(module=module):
                self.assertTrue(module_source(module).exists())

    def test_every_flag_exists_on_its_module(self) -> None:
        cache: dict[str, set[str]] = {}
        for module, args in self.calls:
            if module == "unittest":
                continue
            if module not in cache:
                cache[module] = module_flags(module) | external_help_flags(module)
            allowed = cache[module]
            for token in args:
                if not token.startswith("--"):
                    continue
                flag = token.split("=", 1)[0]
                with self.subTest(module=module, flag=flag):
                    self.assertIn(
                        flag,
                        allowed,
                        f"{flag} is not accepted by {module}",
                    )

    def test_short_flags_are_not_invented(self) -> None:
        for module, args in self.calls:
            if module == "unittest":
                continue
            allowed = module_flags(module) | external_help_flags(module)
            for token in args:
                if re.fullmatch(r"-[a-zA-Z]", token):
                    with self.subTest(module=module, flag=token):
                        self.assertIn(token, allowed)

    def test_new_run_modes_are_dispatched(self) -> None:
        for mode in ("taskgraph", "taskgraph-baseline", "taskgraph-fusion"):
            with self.subTest(mode=mode):
                self.assertTrue(
                    re.search(rf"^\s*{mode}\)", self.text, re.MULTILINE),
                    f"RUN_MODE={mode} has no case branch",
                )
                self.assertIn(f"run_{mode.replace('-', '_')}()", self.text)

    def test_documented_run_modes_are_all_dispatched(self) -> None:
        # Mode names may contain hyphens. The old ``[a-z]+`` pattern stopped at
        # the hyphen, so ``RUN_MODE=taskgraph-perf`` never entered this check and
        # a stale docs reference survived the removal of that mode.
        advertised = set(
            re.findall(
                r"RUN_MODE=([a-z][a-z-]*)\s+bash SERVER_COMMANDS", self.text
            )
        )
        self.assertTrue(advertised)
        for mode in advertised:
            with self.subTest(mode=mode):
                self.assertTrue(
                    re.search(rf"^\s*{mode}\)", self.text, re.MULTILINE),
                    f"advertised RUN_MODE={mode} has no dispatch branch",
                )

    # README.md declares these two as historical: they document the archived
    # ``kernel.cu`` line, including a command surface that no longer exists.
    # They are excluded from the "current docs" scan below on purpose.
    HISTORICAL_DOCS = {
        "06_实际运行数据.md",
        "07_融合收益边界与根因.md",
    }

    def test_current_docs_only_advertise_existing_run_modes(self) -> None:
        """A stale ``RUN_MODE=...`` in a doc costs a whole server session."""

        stale: list[str] = []
        docs = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").glob("*.md"))]
        for path in docs:
            if path.name in self.HISTORICAL_DOCS:
                continue
            text = path.read_text(encoding="utf-8")
            for number, line in enumerate(text.splitlines(), start=1):
                # Prose about *removing* a mode is not an instruction to run it.
                if any(
                    marker in line
                    for marker in ("已移除", "已删除", "已过时", "不再", "历史", "归档",
                                   "removed", "stale", "archive")
                ):
                    continue
                for mode in re.findall(r"RUN_MODE=([a-z][a-z-]*)", line):
                    if not re.search(rf"^\s*{mode}\)", self.text, re.MULTILINE):
                        stale.append(f"{path.name}:{number} RUN_MODE={mode}")
        self.assertFalse(
            stale,
            "documented RUN_MODE(s) have no dispatch branch in "
            f"SERVER_COMMANDS.txt: {stale}",
        )

    def test_every_gpu_test_module_is_invoked_with_a_working_runner(self) -> None:
        """The kernel's correctness modules must actually run on the server.

        They were previously invoked by *nothing* -- RUN_MODE=taskgraph ran only
        the three CPU-side modules, so the seven kernel validation modules would
        have been run only if an operator remembered to. A script-style module
        invoked through unittest is worse than skipped: it reports zero tests and
        exits 0.
        """

        tests_dir = REPO_ROOT / "tests"
        modules = sorted(
            path.stem
            for path in tests_dir.glob("test_taskgraph_*.py")
        ) + sorted(path.stem for path in tests_dir.glob("test_tg_*.py"))
        self.assertTrue(modules)

        for module in modules:
            source = (tests_dir / f"{module}.py").read_text(encoding="utf-8")
            unittest_style = "unittest.TestCase" in source
            with self.subTest(module=module, style="unittest" if unittest_style else "script"):
                if unittest_style:
                    self.assertIn(f"tests.{module}", self.text)
                    self.assertNotIn(f"tests/{module}.py", self.text)
                else:
                    self.assertIn(f"tests/{module}.py", self.text)
                    self.assertNotIn(f"tests.{module}", self.text)

    def test_the_build_arch_is_not_forced_to_one_device(self) -> None:
        """nvcc's -arch=sm_XX emits SASS for that arch only, with no PTX.

        A forced default therefore cannot run on any other device: an sm_80
        default made the whole command file unrunnable on a rented RTX 5090
        (sm_120), failing at the first launch with "no kernel image is
        available for execution on the device".
        """

        self.assertNotIn("NANOVLLM_CUDA_ARCH:-sm_", self.text)
        self.assertIn("auto-detected", self.text)

    def test_the_offline_schedule_gate_runs_before_the_build(self) -> None:
        """docs/17 makes the verifier a precondition for touching schedule.py."""

        body = self._function_body("run_build")
        gate = body.index("tools.verify_schedule")
        build = body.index("nanovllm.backends.megakernel.build")
        self.assertLess(gate, build)
        self.assertIn("--sample", body)

    def test_the_attn_splits_comparison_is_wired(self) -> None:
        body = self._function_body("run_taskgraph_baseline")

        self.assertIn("--attn-splits 2", body)
        self.assertIn("taskgraph_attn_correctness", body)

    def test_documented_commands_use_flags_that_exist(self) -> None:
        """A wrong flag in a doc costs the same session as one in the command file.

        The command file itself is checked flag-by-flag elsewhere; this catches
        the copies inside the docs, which is where a tool rename leaves stale
        flags behind (``--sections bandwidth`` survived a rename until now).
        """

        cache: dict[str, set[str]] = {}
        problems: list[str] = []
        for path in (REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").glob("*.md"))):
            if path.name in self.HISTORICAL_DOCS:
                continue
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                for match in re.finditer(r"python -m ([a-zA-Z_0-9.]+)([^\n]*)", line):
                    module, tail = match.group(1), match.group(2)
                    if module in cache:
                        allowed = cache[module]
                    else:
                        try:
                            allowed = module_flags(module) | external_help_flags(module)
                        except (AssertionError, FileNotFoundError, SyntaxError, SystemExit):
                            # Not a local module (e.g. ``python -m pip install``).
                            continue
                        cache[module] = allowed
                    for token in tail.split():
                        if not token.startswith("--"):
                            continue
                        flag = token.split("=", 1)[0].strip("`）,;")
                        if flag not in allowed:
                            problems.append(f"{path.name}:{number} {module} {flag}")
        self.assertFalse(
            problems,
            f"documented commands pass flags their module does not accept: {problems}",
        )

    def test_mainline_commands_do_not_reference_archived_backends(self) -> None:
        for name in (
            "megakernel-tg",
            "megakernel-wmma",
            "megakernel-burst",
            "megakernel_batch_decode",
            "megakernel_burst",
        ):
            self.assertNotIn(name, self.text)

    # Sources that must never point an operator at a module this branch does not
    # ship: the checked-in docs, the command file, the live tools and the library.
    REFERENCE_TARGETS = (
        "README.md",
        "SERVER_COMMANDS.txt",
        "docs",
        "tools",
        "src/nanovllm",
        "tests",
    )

    def test_referenced_tools_exist_on_this_branch(self) -> None:
        """No doc, message or comment may cite a tool this branch does not ship.

        ``tools/bench_roofline_a100.py`` and ``tools/verdict.py`` (archive branch
        only) live on ``codex/archive/frozen-kernel``, but the docs and a tool's
        own hint text kept recommending them -- the same failure mode as the
        removed ``RUN_MODE=taskgraph-perf``. Historical docs are exempt, so a
        rename is caught here instead of on a paid server session.

        Only module-shaped references are checked: a ``tools/<module>.py`` path
        or the dotted ``tools.<module>`` form. A bare directory mention
        (``tools/benchmarks``), a glob and a struck-through retraction are not
        module references.
        """

        paths: list[Path] = []
        for target in self.REFERENCE_TARGETS:
            path = REPO_ROOT / target
            if path.is_dir():
                paths.extend(sorted(path.rglob("*.py")))
                paths.extend(sorted(path.glob("*.md")))
            else:
                paths.append(path)

        self.assertFalse(
            find_dangling_tool_references(paths),
            "references to tools that do not exist on this branch: "
            f"{find_dangling_tool_references(paths)}",
        )

    def test_the_scanner_catches_a_dangling_reference(self) -> None:
        """Negative validation: the guard must fail on a real dangling citation."""

        def cite(module: str) -> str:
            # Built by concatenation so this test's own source does not contain a
            # literal citation for the scanner to find.
            return "`tools/" + module + ".py` and `tools." + module + "`"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = root / "bad.md"
            bad.write_text("见 " + cite("no_such_tool") + "。\n", encoding="utf-8")
            self.assertEqual(
                find_dangling_tool_references([bad]),
                ["bad.md:1 cites tools." + "no_such_tool"] * 2,
            )

            historic = root / "historic.md"
            historic.write_text(
                "> 本文属于历史记录（归档）。\n\n见 " + cite("no_such_tool") + "。\n",
                encoding="utf-8",
            )
            self.assertEqual(find_dangling_tool_references([historic]), [])

            retracted = root / "retracted.md"
            retracted.write_text(
                "1. ~~写 " + cite("no_such_tool") + "~~ → 改为扩 compare.py（已完成）\n",
                encoding="utf-8",
            )
            self.assertEqual(find_dangling_tool_references([retracted]), [])

            globbed = root / "globbed.md"
            globbed.write_text(
                "| correctness 门 | `tools/correctness_*.py` |\n", encoding="utf-8"
            )
            self.assertEqual(find_dangling_tool_references([globbed]), [])

            directory_mention = root / "prose.md"
            directory_mention.write_text(
                "the tools/benchmarks packages live at the repository root\n",
                encoding="utf-8",
            )
            self.assertEqual(find_dangling_tool_references([directory_mention]), [])

            shipped = root / "shipped.md"
            shipped.write_text("见 `tools/bench_taskgraph.py`。\n", encoding="utf-8")
            self.assertEqual(find_dangling_tool_references([shipped]), [])

    def _function_body(self, name: str) -> str:
        match = re.search(
            rf"^{name}\(\) \{{(.*?)^\}}", self.text, re.MULTILINE | re.DOTALL
        )
        self.assertIsNotNone(match, f"{name} not found in SERVER_COMMANDS.txt")
        assert match is not None
        return match.group(1)

    def test_no_second_command_file_exists(self) -> None:
        """CONTRIBUTING.md: exactly one checked-in server command file."""

        extras = [
            path.name
            for path in REPO_ROOT.glob("*COMMANDS*.txt")
            if path.name != "SERVER_COMMANDS.txt"
        ]
        self.assertEqual(extras, [])

    def test_docs_promise_files_the_commands_actually_write(self) -> None:
        """docs/08 tells the user which files to send back; they must be real.

        Artifact names are machine-neutral on purpose (``results/performance.json``
        rather than ``results/a100_performance.json``): the same command file must
        run on an A100 and on an RTX 5090 without implying one machine's numbers
        are the other's.
        """

        promised = set(
            re.findall(r"results/([a-z0-9_]+\.(?:json|md|log))", DOCS_09.read_text(encoding="utf-8"))
        )
        self.assertTrue(promised)
        for filename in sorted(promised):
            with self.subTest(filename=filename):
                self.assertIn(
                    filename,
                    self.text,
                    f"docs/08 promises {filename} but SERVER_COMMANDS never writes it",
                )

    def test_artifact_names_are_not_tied_to_one_architecture(self) -> None:
        """A machine-specific prefix would make the other machine's run lie."""

        for line in self.text.splitlines():
            if line.lstrip().startswith("#"):
                continue
            self.assertNotRegex(
                line,
                r"results/(?:a100|h100|v100|rtx)_",
                "artifact names must be machine-neutral (docs/08 §4)",
            )


class CitationTests(unittest.TestCase):
    """docs/06 section 9.9 is the external-evidence record; keep it usable."""

    def setUp(self) -> None:
        self.text = (REPO_ROOT / "docs" / "07_融合收益边界与根因.md").read_text(
            encoding="utf-8"
        )

    def test_megaqwen_owner_is_corrected(self) -> None:
        """The v5 doc credited the wrong GitHub owner."""

        self.assertIn("https://github.com/Infatoshi/MegaQwen", self.text)
        section = self.text.split("### 9.9")[1]
        # The wrong owner may only appear next to an explicit correction.
        markers = ("不是", "错误", "修正")
        for match in re.finditer(r"NoneBone/megaQwen", section):
            window = section[max(0, match.start() - 160) : match.end() + 160]
            self.assertTrue(
                any(marker in window for marker in markers),
                "wrong owner appears without a correction marker",
            )

    def test_every_arxiv_id_is_inside_a_link(self) -> None:
        """An identifier the reader must paste is not a citation."""

        ids = set(re.findall(r"arXiv:(\d{4}\.\d{4,5})", self.text))
        self.assertGreaterEqual(len(ids), 4)
        linked = set(
            re.findall(r"\[(?:arXiv:)?(\d{4}\.\d{4,5})\]\(https?://", self.text)
        )
        for arxiv_id in sorted(ids):
            with self.subTest(arxiv_id=arxiv_id):
                self.assertIn(
                    arxiv_id, linked, f"{arxiv_id} is named but not linked"
                )

    def test_primary_sources_are_linked(self) -> None:
        for url in (
            "https://github.com/Infatoshi/MegaQwen",
            "https://arxiv.org/abs/2605.11581",
            "https://arxiv.org/abs/2512.22219",
            "https://www.usenix.org/conference/osdi26/presentation/cheng",
            "https://cohere.com/blog/megakernels",
        ):
            with self.subTest(url=url):
                self.assertIn(url, self.text)


if __name__ == "__main__":
    unittest.main()
