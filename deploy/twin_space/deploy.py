"""Create or update the twin analyst Space (chat + decision page) on Hugging Face.

One-time auth in your terminal, then deploy:

    hf auth login                                   # a write token from hf.co/settings/tokens
    python deploy/twin_space/deploy.py --set-key    # first time: also stores the API key secret
    python deploy/twin_space/deploy.py              # later: push a new pinned commit

The Space builds one commit of the public repo (default: origin/main), so the page and the
claims ledger it serves are that commit's, bound strictly. ``--set-key`` copies
ANTHROPIC_API_KEY (environment, else the repo-root .env) into a Space secret; it is never
written to the uploaded files or printed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_SPACE = "Rekin226/choushui-twin"


def remote_main_sha() -> str:
    subprocess.run(["git", "fetch", "-q", "origin", "main"], cwd=REPO, check=True)
    return subprocess.run(["git", "rev-parse", "origin/main"], cwd=REPO, check=True,
                          capture_output=True, text=True).stdout.strip()


def api_key() -> str | None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    env = REPO / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", default=DEFAULT_SPACE)
    ap.add_argument("--sha", default=None, help="repo commit to build (default: origin/main)")
    ap.add_argument("--set-key", action="store_true", help="store ANTHROPIC_API_KEY as a secret")
    ap.add_argument("--daily-cap", type=int, default=None,
                    help="set the ANALYST_DAILY_LLM_CAP Space variable")
    a = ap.parse_args(argv)

    from huggingface_hub import HfApi

    sha = a.sha or remote_main_sha()
    api = HfApi()
    api.create_repo(a.space, repo_type="space", space_sdk="docker", exist_ok=True)
    if a.set_key:
        key = api_key()
        if not key:
            raise SystemExit("no ANTHROPIC_API_KEY in the environment or .env")
        api.add_space_secret(a.space, "ANTHROPIC_API_KEY", key)
        print("secret ANTHROPIC_API_KEY set")
    if a.daily_cap is not None:
        api.add_space_variable(a.space, "ANALYST_DAILY_LLM_CAP", str(a.daily_cap))

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for name in ("launch.py", "README.md"):
            shutil.copy(HERE / name, out / name)
        docker = (HERE / "Dockerfile").read_text(encoding="utf-8")
        if "ARG REPO_SHA=main" not in docker:
            raise SystemExit("Dockerfile has no 'ARG REPO_SHA=main' line to pin")
        (out / "Dockerfile").write_text(docker.replace("ARG REPO_SHA=main", f"ARG REPO_SHA={sha}"),
                                        encoding="utf-8")
        api.upload_folder(folder_path=str(out), repo_id=a.space, repo_type="space",
                          commit_message=f"build HydroPhysicsAI {sha[:12]}")
    sub = a.space.lower().replace("/", "-")
    print(f"pushed {sha[:12]} -> https://huggingface.co/spaces/{a.space}  (app: https://{sub}.hf.space/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
