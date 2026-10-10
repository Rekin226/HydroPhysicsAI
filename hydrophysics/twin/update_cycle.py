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
from .validation import evaluate, observations


def _acceptance(entry: dict, cwd: Path, release: dict, observation_hash: str) -> dict:
    """Recompute the supported gate and bind its inputs to the rendered release.

    This checks reproducibility/identity, not authenticity of supplied field records
    or proof that a holdout was untouched. Only the head-validation task is supported.
    """
    required = {"report", "protocol", "predictions"}
    if not isinstance(entry, dict) or set(entry) != required:
        raise ValueError("Each acceptance entry needs report, protocol and predictions paths")
    paths = {key: Path(value) if Path(value).is_absolute() else cwd / value
             for key, value in entry.items()}
    hashes = {key: sha256(path) for key, path in paths.items()}
    supplied = json.loads(paths["report"].read_text())
    if not isinstance(supplied, dict) or supplied.get("passed") is not True:
        raise ValueError("Acceptance check failed; current release retained")
    checked = evaluate(paths["predictions"], paths["protocol"])
    if supplied != checked or not checked["passed"]:
        raise ValueError("Acceptance report does not reproduce from its protocol and predictions")
    if hashes != {key: sha256(path) for key, path in paths.items()}:
        raise ValueError("Acceptance artifacts changed during validation")
    sources = release.get("sources", {})
    expected = {
        "model_sha256": sources.get("forward", {}).get("sha256"),
        "code_sha256": release.get("code_sha256"),
        "observations_sha256": observation_hash,
        "protocol_sha256": sources.get("validation_protocol", {}).get("sha256"),
        "predictions_sha256": sources.get("validation_predictions", {}).get("sha256"),
    }
    for key, value in expected.items():
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
                or checked.get(key) != value):
            raise ValueError(f"Acceptance {key} does not match the rendered release")
    if checked["task"] != "free-running-head-validation":
        raise ValueError("Unsupported acceptance task")
    return {"name": paths["report"].name, "sha256": hashes["report"],
            "task": checked["task"], "period": checked["period"], **expected}


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
            page_hash = sha256(config["page"])
            payload = read_payload(config["page"])
            source = payload.get("release", {}).get("sources", {}).get("observations", {})
            if source.get("sha256") != qc["input_sha256"]:
                raise ValueError("Rendered run does not identify the ingested observation batch")
            if payload["meta"]["vintage"] < qc["last_date"][:7]:
                raise ValueError("Model origin predates the observation batch")
            report["acceptance"] = [_acceptance(entry, cwd, payload["release"], qc["input_sha256"])
                                    for entry in config["acceptance_reports"]]
            if sha256(config["observations"]) != qc["input_sha256"]:
                raise ValueError("Observation batch changed during the update")
            manifest = publish(config["page"], root, expected_sha256=page_hash,
                               acceptance=report["acceptance"])
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
