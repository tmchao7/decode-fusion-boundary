import os
import socket
import unittest
from unittest.mock import patch

from nanovllm.utils.distributed import build_init_method


class DistributedInitTests(unittest.TestCase):
    def test_uses_explicit_port_from_environment(self):
        with patch.dict(os.environ, {"NANOVLLM_DIST_PORT": "24567"}, clear=False):
            self.assertEqual(build_init_method(), "tcp://127.0.0.1:24567")

    def test_rejects_invalid_explicit_port(self):
        with patch.dict(os.environ, {"NANOVLLM_DIST_PORT": "not-a-port"}, clear=False):
            with self.assertRaisesRegex(ValueError, "NANOVLLM_DIST_PORT"):
                build_init_method()

    def test_auto_selected_port_is_bindable(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NANOVLLM_DIST_PORT", None)
            init_method = build_init_method()
        port = int(init_method.rsplit(":", 1)[1])
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", port))


if __name__ == "__main__":
    unittest.main()
