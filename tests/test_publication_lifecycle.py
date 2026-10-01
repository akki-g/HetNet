"""The separate launcher owns its descendants and rejects stale provenance."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from publication_reconstruction import __main__ as launcher


ROOT = Path(__file__).resolve().parents[1]


def terminated(pid):
    """An orphan can briefly remain a zombie before the OS reaps it."""
    result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                            capture_output=True, text=True)
    return result.returncode != 0 or result.stdout.strip().startswith("Z")


@pytest.mark.skipif(os.name != "posix", reason="launcher uses POSIX process groups")
@pytest.mark.parametrize("failure,expected", [("SIGINT", 130), ("SIGTERM", 143), ("output-error", 1)])
def test_launcher_interrupt_or_output_failure_stops_child_group(tmp_path, failure, expected):
    ready = tmp_path / "ready.json"
    output = tmp_path / "run"
    # The grandchild is in the child's process group; neither runs training.
    child_code = "\n".join([
        "import json, os, subprocess, sys, time",
        "from pathlib import Path",
        "grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])",
        f"Path({str(ready)!r}).write_text(json.dumps([os.getpid(), grandchild.pid]))",
        "print('ready', flush=True)",
        "time.sleep(120)",
    ])
    harness = "\n".join([
        "import sys",
        "from publication_reconstruction import __main__ as m",
        "original_resolve = m.resolve",
        "def resolve(args):",
        "    plan = original_resolve(args)",
        f"    plan['command'] = [sys.executable, '-u', '-c', {child_code!r}]",
        "    return plan",
        "m.resolve = resolve",
        "def fail_output(*args, **kwargs):",
        "    raise OSError('intentional log forwarding failure')",
        "m.print = fail_output" if failure == "output-error" else "# normal forwarding",
        "raise SystemExit(m.main())",
    ])
    proc = subprocess.Popen([sys.executable, "-c", harness, "train", "--task", "pp", "--seed", "0",
                             "--output", str(output)], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    pids = []
    try:
        deadline = time.monotonic() + 20
        while not ready.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert ready.exists(), proc.communicate(timeout=5)[0]
        pids = json.loads(ready.read_text())
        if failure != "output-error":
            proc.send_signal(getattr(signal, failure))
        stdout, _ = proc.communicate(timeout=15)
        assert proc.returncode == expected, stdout
        assert (output / "exit_code.txt").read_text().strip() == str(expected)
        deadline = time.monotonic() + 3
        while not all(terminated(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(.02)
        assert all(terminated(pid) for pid in pids), (pids, stdout)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_origin_validation_rejects_changed_or_unrecorded_runtime(tmp_path, monkeypatch):
    (tmp_path / "runtime").mkdir()
    source = tmp_path / "runtime/main.py"
    source.write_text("print('original')\n")
    origins = {"files": {"main.py": {"current_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}}
    (tmp_path / "ORIGINS.json").write_text(json.dumps(origins))
    monkeypatch.setattr(launcher, "HERE", tmp_path)
    payloads, _ = launcher.validate_source_origins()
    assert payloads["main.py"] == source.read_bytes()
    source.write_text("print('changed')\n")
    with pytest.raises(RuntimeError, match="Runtime source differs"):
        launcher.validate_source_origins()
    source.write_bytes(payloads["main.py"])
    (tmp_path / "runtime/unrecorded.py").write_text("pass\n")
    with pytest.raises(RuntimeError, match="inventory differs"):
        launcher.validate_source_origins()
