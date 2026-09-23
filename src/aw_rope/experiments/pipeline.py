"""One-allocation, serial, resumable AW-RoPE experiment pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import torch


PROXY_VARIABLES = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _command_hash(command: list[str]) -> str:
    return hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()


def _run_step(name: str, command: list[str], *, run_root: Path, project_root: Path) -> None:
    state = run_root / ".state"
    logs = run_root / "logs"
    state.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    done_path = state / f"{name}.done.json"
    failed_path = state / f"{name}.failed.json"
    command_digest = _command_hash(command)
    if done_path.exists():
        completed = json.loads(done_path.read_text(encoding="utf-8"))
        if completed.get("command_sha256") == command_digest:
            print(json.dumps({"step": name, "action": "skip-completed"}), flush=True)
            return
        raise RuntimeError(f"{name} completion marker belongs to a different command")
    if failed_path.exists():
        failed_path.unlink()
    started_at = _now()
    started = time.monotonic()
    log_path = logs / f"{name}.log"
    print(json.dumps({"step": name, "action": "start", "log": str(log_path)}), flush=True)
    environment = os.environ.copy()
    for name_of_proxy in PROXY_VARIABLES:
        environment.pop(name_of_proxy, None)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"event": "start", "time": started_at, "command": command}) + "\n")
        log.flush()
        process = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    payload = {
        "step": name,
        "command": command,
        "command_sha256": command_digest,
        "started_at": started_at,
        "finished_at": _now(),
        "elapsed_seconds": time.monotonic() - started,
        "returncode": process.returncode,
        "log": str(log_path),
    }
    if process.returncode:
        _atomic_json(failed_path, payload)
        raise RuntimeError(f"{name} failed with return code {process.returncode}; see {log_path}")
    _atomic_json(done_path, payload)
    print(json.dumps({"step": name, "action": "completed", "seconds": payload["elapsed_seconds"]}), flush=True)


def run_pipeline(project_root: Path, run_root: Path, workers: int | None) -> None:
    project_root = project_root.resolve()
    run_root = run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    active_proxies = {name: os.environ[name] for name in PROXY_VARIABLES if os.environ.get(name)}
    if active_proxies:
        raise RuntimeError(f"proxy variables must be unset: {sorted(active_proxies)}")
    runtime = {
        "started_at": _now(),
        "project_root": str(project_root),
        "run_root": str(run_root),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "proxy_mode": "direct-no-proxy",
        "workflow": ["stage1", "stage2", "stage3", "aggregate", "verify"],
    }
    _atomic_json(run_root / "runtime.json", runtime)
    common = ["--data-root", str(project_root / "data"), "--output-root", str(run_root)]
    if workers is not None:
        common.extend(["--workers", str(workers)])
    for stage in ("stage1", "stage2", "stage3"):
        _run_step(
            stage,
            [sys.executable, "-m", "aw_rope.experiments.stage", "--stage", stage, *common],
            run_root=run_root,
            project_root=project_root,
        )
    _run_step(
        "aggregate",
        [sys.executable, "-m", "aw_rope.experiments.report", "aggregate", "--run-root", str(run_root)],
        run_root=run_root,
        project_root=project_root,
    )
    _run_step(
        "verify",
        [sys.executable, "-m", "aw_rope.experiments.report", "full-verify", "--run-root", str(run_root)],
        run_root=run_root,
        project_root=project_root,
    )
    _atomic_json(run_root / ".state" / "pipeline.done.json", {"completed_at": _now(), "valid": True})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_pipeline(args.project_root, args.run_root, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
