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
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SERVER_COMMANDS = REPO_ROOT / "SERVER_COMMANDS.txt"
DOCS_09 = REPO_ROOT / "docs" / "09_高并发判定任务书.md"

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
        self.assertIn("tools.verdict", modules)
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
        for mode in ("account", "roofline"):
            with self.subTest(mode=mode):
                self.assertTrue(
                    re.search(rf"^\s*{mode}\)", self.text, re.MULTILINE),
                    f"RUN_MODE={mode} has no case branch",
                )
                self.assertIn(f"run_{mode}()", self.text)

    def test_documented_run_modes_are_all_dispatched(self) -> None:
        advertised = set(
            re.findall(r"RUN_MODE=([a-z]+)\s+bash SERVER_COMMANDS", self.text)
        )
        self.assertTrue(advertised)
        for mode in advertised:
            with self.subTest(mode=mode):
                self.assertTrue(
                    re.search(rf"^\s*{mode}\)", self.text, re.MULTILINE),
                    f"advertised RUN_MODE={mode} has no dispatch branch",
                )

    def test_verdict_is_computed_by_both_probe_phases(self) -> None:
        for function in ("run_account", "run_roofline"):
            block = self._function_body(function)
            with self.subTest(function=function):
                self.assertIn("tools.verdict", block)

    def _function_body(self, name: str) -> str:
        match = re.search(
            rf"^{name}\(\) \{{(.*?)^\}}", self.text, re.MULTILINE | re.DOTALL
        )
        self.assertIsNotNone(match, f"{name} not found in SERVER_COMMANDS.txt")
        assert match is not None
        return match.group(1)

    def test_no_second_command_file_exists(self) -> None:
        """AGENTS.md: exactly one checked-in server command file."""

        extras = [
            path.name
            for path in REPO_ROOT.glob("*COMMANDS*.txt")
            if path.name != "SERVER_COMMANDS.txt"
        ]
        self.assertEqual(extras, [])

    def test_docs_promise_files_the_commands_actually_write(self) -> None:
        """docs/09 tells the user which files to send back; they must be real."""

        promised = set(
            re.findall(r"(a100_[a-z_]+\.(?:json|md|log))", DOCS_09.read_text(encoding="utf-8"))
        )
        self.assertTrue(promised)
        for filename in sorted(promised):
            with self.subTest(filename=filename):
                self.assertIn(
                    filename,
                    self.text,
                    f"docs/09 promises {filename} but SERVER_COMMANDS never writes it",
                )


class CitationTests(unittest.TestCase):
    """docs/07 section 9.9 is the external-evidence record; keep it usable."""

    def setUp(self) -> None:
        self.text = (REPO_ROOT / "docs" / "07_多batch为何打不过cudagraph.md").read_text(
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
