"""Cross-platform lifecycle command for the ACE-Step API service."""

from __future__ import annotations

import argparse
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / ".runtime"


def platform_environment() -> dict[str, str]:
    """Return environment overrides required by the current hardware platform."""
    values = os.environ.copy()
    values.setdefault("TOKENIZERS_PARALLELISM", "false")
    if sys.platform == "darwin" and platform.machine() == "arm64":
        values.setdefault("ACESTEP_LM_BACKEND", "mlx")
    return values


def port_open(port: int) -> bool:
    """Return whether the local API port is accepting connections."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _wait(port: int, expected: bool, timeout: float, process: subprocess.Popen[bytes] | None = None) -> bool:
    """Wait for the API port to reach the requested state."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_open(port) is expected:
            return True
        if process is not None and process.poll() is not None:
            return False
        time.sleep(0.5)
    return port_open(port) is expected


def _process_exists(process_id: int) -> bool:
    """Return whether a process identifier currently exists."""
    try:
        os.kill(process_id, 0)
        return True
    except OSError:
        return False


def _process_identity(process_id: int) -> str:
    """Return the command line used to validate repository ownership."""
    if os.name != "nt":
        completed = subprocess.run(
            ["ps", "-p", str(process_id), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=10,
        )
    else:
        command = f"(Get-CimInstance Win32_Process -Filter 'ProcessId = {process_id}').CommandLine"
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=10,
        )
    return completed.stdout.strip()


def _terminate(process_id: int) -> None:
    """Terminate a service process group on the current platform."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process_id), "/T", "/F"],
            check=False, capture_output=True,
        )
        return
    try:
        os.killpg(process_id, signal.SIGTERM)
    except ProcessLookupError:
        return


def start(runtime_root: Path, host: str, port: int, timeout: float) -> int:
    """Start the API with the active environment's Python and record ownership."""
    if port_open(port):
        print(f"ACE-Step is already listening on port {port}.")
        return 0
    runtime_root.mkdir(parents=True, exist_ok=True)
    stdout_path = runtime_root / "api.stdout.log"
    stderr_path = runtime_root / "api.stderr.log"
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        process = subprocess.Popen(
            [sys.executable, "-m", "acestep.api_server", "--host", host, "--port", str(port)],
            cwd=PROJECT_ROOT, env=platform_environment(), stdout=stdout, stderr=stderr,
            creationflags=creationflags, start_new_session=os.name != "nt",
        )
    (runtime_root / "api.pid").write_text(str(process.pid), encoding="ascii")
    if not _wait(port, True, timeout, process):
        _terminate(process.pid)
        (runtime_root / "api.pid").unlink(missing_ok=True)
        raise RuntimeError(f"ACE-Step failed to start; inspect {stderr_path}")
    print(f"ACE-Step is healthy on port {port} (PID {process.pid}).")
    return 0


def stop(runtime_root: Path, port: int, timeout: float) -> int:
    """Stop only the API process identified by the managed PID file."""
    pid_file = runtime_root / "api.pid"
    if not pid_file.is_file():
        if port_open(port):
            raise RuntimeError(f"Port {port} is active but no managed PID exists; refusing to stop it")
        print("ACE-Step is already stopped.")
        return 0
    try:
        process_id = int(pid_file.read_text(encoding="ascii").strip())
    except ValueError as exc:
        raise RuntimeError("ACE-Step PID file is invalid; refusing to stop a process") from exc
    if _process_exists(process_id):
        identity = _process_identity(process_id).casefold()
        if str(PROJECT_ROOT).casefold() not in identity:
            raise RuntimeError(f"PID {process_id} is not owned by this ACE-Step checkout")
        _terminate(process_id)
    if not _wait(port, False, timeout):
        raise RuntimeError(f"ACE-Step did not stop within {timeout:g} seconds")
    pid_file.unlink(missing_ok=True)
    print("ACE-Step stopped.")
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the service lifecycle argument parser."""
    parser = argparse.ArgumentParser(prog="acestep-service")
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("start", "stop", "status"):
        child = subparsers.add_parser(command)
        child.add_argument("--port", type=int, default=8001)
        if command == "start":
            child.add_argument("--host", default="127.0.0.1")
        if command != "status":
            child.add_argument("--timeout", type=float, default=300 if command == "start" else 15)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ACE-Step service lifecycle CLI."""
    args = _parser().parse_args(argv)
    runtime_root = args.runtime_root.expanduser().resolve()
    try:
        if args.command == "start":
            return start(runtime_root, args.host, args.port, args.timeout)
        if args.command == "stop":
            return stop(runtime_root, args.port, args.timeout)
        running = port_open(args.port)
        print("running" if running else "stopped")
        return 0 if running else 1
    except (OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
