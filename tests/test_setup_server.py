import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "setup_server.sh"


class SetupServerScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = SCRIPT.read_text(encoding="utf-8")

    def test_shell_syntax_is_valid(self):
        result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_installer_is_explicit_and_defaults_to_conda_when_available(self):
        self.assertIn('INSTALLER="${NANOVLLM_INSTALLER:-}"', self.body)
        self.assertIn('command -v conda', self.body)
        self.assertIn('INSTALLER=conda', self.body)
        self.assertIn('INSTALLER=uv', self.body)
        self.assertRegex(self.body, r'NANOVLLM_INSTALLER must be conda or uv')

    def test_mirror_and_flash_wheel_are_configurable(self):
        self.assertIn("NANOVLLM_PYPI_INDEX_URL", self.body)
        self.assertIn("NANOVLLM_FLASH_ATTN_WHEEL", self.body)
        self.assertIn('pip_install --no-deps "${FLASH_ATTN_WHEEL}"', self.body)
        self.assertIn('pip_install "flash-attn>=2.7,<3" --no-build-isolation', self.body)

    def test_every_install_path_has_a_package_index(self):
        logical = []
        current = ""
        for raw in self.body.splitlines():
            current += raw.strip() + " "
            if not raw.rstrip().endswith("\\"):
                logical.append(current)
                current = ""
        for line in logical:
            if "pip install" in line and "--index-url" not in line and "pip_install" not in line:
                self.assertIn("--index-url", line, line)
        self.assertIn('uv pip install --python "${PYTHON}" --index-url "${PYPI_INDEX_URL}"', self.body)
        self.assertIn('"${PYTHON}" -m pip install \\\n    --index-url "${PYPI_INDEX_URL}"', self.body)

    def test_post_install_gates_are_present(self):
        self.assertIn('"${PYTHON}" -m pip check', self.body)
        self.assertIn('tools.verify_server_env', self.body)
        self.assertIn('nanovllm.backends.megakernel.build', self.body)
        self.assertIn('NANOVLLM_VERIFY_MODEL', self.body)


if __name__ == "__main__":
    unittest.main()
