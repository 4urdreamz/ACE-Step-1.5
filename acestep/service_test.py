"""Tests for the ACE-Step service lifecycle contract."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from acestep import service


class ServiceTest(unittest.TestCase):
    """Validate platform selection and process ownership safeguards."""

    @patch.object(service.platform, "machine", return_value="arm64")
    @patch.object(service.sys, "platform", "darwin")
    def test_platform_environment_selects_mlx(self, _machine: object) -> None:
        """Apple Silicon selects MLX without relying on a shell launcher."""
        with patch.dict(service.os.environ, {}, clear=True):
            self.assertEqual(service.platform_environment()["ACESTEP_LM_BACKEND"], "mlx")

    def test_stop_refuses_unowned_listener(self) -> None:
        """A listener without repository ownership is never terminated."""
        with TemporaryDirectory() as folder, patch.object(service, "port_open", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "no managed PID"):
                service.stop(Path(folder), 8001, 1)
    def test_stop_refuses_reused_pid(self) -> None:
        """A stale PID file cannot terminate a process from another checkout."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "api.pid").write_text("42", encoding="ascii")
            with (
                patch.object(service, "_process_exists", return_value=True),
                patch.object(service, "_process_identity", return_value="unrelated"),
                self.assertRaisesRegex(RuntimeError, "not owned"),
            ):
                service.stop(root, 8001, 1)



if __name__ == "__main__":
    unittest.main()
