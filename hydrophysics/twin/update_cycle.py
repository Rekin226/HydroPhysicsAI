"""Run a configured local update cycle, preserving the last accepted research release.

Commands are explicit argument lists (never shell strings). A production connection
requires supplied new measurements and a configured, validated assimilation/simulation
command. This module does not pretend that refreshing an old page assimilates new data.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import time
from pathlib import Path

from .release import atomic_json, publish, read_payload, sha256
from .validation import observations


def run(config: dict, status_path: Path) -> dict:
    config = dict(config)
    cwd = Path(config.get("cwd", ".")).resolve()
    for key in ("release_root", "observations", "page"):
        path = Path(config[key])
        config[key] = str(path if path.is_absolute() else cwd / path)
    root = Path(config["release_root"])
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".update.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another update is already running") from exc
        report = {"schema": 1, "state": "running", "started_unix": time.time(), "steps": []}
        atomic_json(status_path, report)
        try:
            if "fetch" in config:
                _command(config["fetch"], config, report)
            qc = observations(config["observations"], as_of=config.get("as_of"),
                              max_age_days=config.get("max_age_days", 62))
            report["observations"] = qc
            if not qc["fresh"]:
                raise ValueError("Observation batch is stale; current release retained")
            if not config.get("steps"):
                raise ValueError("An explicit assimilation/simulation and render workflow is required")
            for command in config["steps"]:
                _command(command, config, report)
                atomic_json(status_path, report)
            if not config.get("acceptance_reports"):
                raise ValueError("At least one explicit acceptance report is required")
            report["acceptance"] = []
            for filename in config["acceptance_reports"]:
                path = Path(filename)
                path = path if path.is_absolute() else cwd / path
                acceptance = json.loads(path.read_text())
                if acceptance.get("passed") is not True:
                    raise ValueError("Acceptance check failed; current release retained")
                report["acceptance"].append({"name": path.name, "sha256": sha256(path)})
            payload = read_payload(config["page"])
            source = payload.get("release", {}).get("sources", {}).get("observations", {})
            if source.get("sha256") != qc["input_sha256"]:
                raise ValueError("Rendered run does not identify the ingested observation batch")
            if payload["meta"]["vintage"] < qc["last_date"][:7]:
                raise ValueError("Model origin predates the observation batch")
            manifest = publish(config["page"], root)
            report.update(state="published", release_sha256=manifest["page_sha256"])
        except (ValueError, OSError, KeyError, TypeError) as exc:
            report.update(state="failed", error=str(exc))
        report["finished_unix"] = time.time()
        atomic_json(status_path, report)
        return report


def _command(command: list[str], config: dict, report: dict) -> None:
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise ValueError("Each workflow command must be a non-empty argument list")
    start = time.perf_counter()
    # Inherit environment credentials; never copy them into status files or command logs.
    try:
        result = subprocess.run(command, cwd=config.get("cwd"),
                                timeout=config.get("timeout_seconds", 3600),
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        # Exception strings can contain command arguments, including connection details.
        raise ValueError(f"Workflow step {len(report['steps']) + 1}: {type(exc).__name__}") from None
    report["steps"].append({"index": len(report["steps"]), "exit_code": result.returncode,
                            "seconds": round(time.perf_counter() - start, 3)})
    if result.returncode:
        raise ValueError(f"Workflow step {len(report['steps'])} failed (exit {result.returncode})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument("--status", type=Path, default=Path("results/update-status.json"))
    args = ap.parse_args(argv)
    config = json.loads(args.config.read_text())
    result = run(config, args.status)
    print(json.dumps(result, indent=2))
    if result["state"] != "published":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
