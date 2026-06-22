"""Create and push the Hugging Face Space for the demo.

One-time auth (in your terminal), then deploy:

    hf auth login                      # paste a write token from hf.co/settings/tokens
    python app/deploy_space.py         # creates + uploads the Space

Re-run it any time to push updates. Override the Space id with --space.
The token is read from the standard HF cache (or the HF_TOKEN env var).
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

DEFAULT_SPACE = "Rekin226/HydroPhysicsAI-demo"
APP_DIR = Path(__file__).resolve().parent
_REQ_RE = re.compile(
    r"(hydrophysics-ai @ git\+https://github\.com/Rekin226/HydroPhysicsAI\.git@)\S+")


def pin_requirements_to_head() -> str:
    """Pin the hydrophysics-ai git dependency to the current commit SHA.

    The package version is static (0.0.1), so pip treats an unchanged ``@main`` git
    requirement as already-satisfied and serves a STALE cached build -- which silently
    leaves the Space running old library code (e.g. a PhysicsUDE without ``rain_memory``).
    Pinning to the exact SHA changes the requirement each release, forcing a clean
    reinstall of the deployed commit. Rewrites app/requirements.txt in place so the
    committed file records exactly what the Space installs.
    """
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=APP_DIR,
                         capture_output=True, text=True, check=True).stdout.strip()
    req = APP_DIR / "requirements.txt"
    text = req.read_text()
    pinned, n = _REQ_RE.subn(rf"\g<1>{sha}", text)
    if n and pinned != text:
        req.write_text(pinned)
        print(f"pinned hydrophysics-ai -> {sha}")
    return sha


def main() -> None:
    ap = argparse.ArgumentParser(description="Deploy the demo to HF Spaces")
    ap.add_argument("--space", default=DEFAULT_SPACE, help="HF Space id (user/name)")
    ap.add_argument("--private", action="store_true", help="create the Space private")
    args = ap.parse_args()

    from huggingface_hub import HfApi, whoami

    api = HfApi()
    try:
        who = whoami()["name"]
    except Exception as exc:
        raise SystemExit("Not logged in. Run `hf auth login` with a *write* token first.") from exc
    print(f"authenticated as: {who}")

    pin_requirements_to_head()

    api.create_repo(repo_id=args.space, repo_type="space", space_sdk="gradio",
                    private=args.private, exist_ok=True)
    print(f"space ready: https://huggingface.co/spaces/{args.space}")

    # Upload app.py, requirements.txt and the Space README (the card frontmatter).
    api.upload_folder(
        folder_path=str(APP_DIR),
        repo_id=args.space,
        repo_type="space",
        ignore_patterns=["deploy_space.py", "__pycache__/*", "*.pyc"],
        commit_message="Deploy HydroPhysicsAI demo",
    )
    print(f"\ndeployed -> https://huggingface.co/spaces/{args.space}")
    print("first build takes a few minutes (installs torch + trains at startup).")


if __name__ == "__main__":
    main()
