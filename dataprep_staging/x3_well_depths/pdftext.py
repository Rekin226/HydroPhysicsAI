"""Text of a downloaded PDF (untrusted data) via poppler's pdftotext, for reading only.

Usage: python -I pdftext.py <pdf> <out.txt> [first_page last_page]

The PDF is parsed by poppler, never executed. ``-layout`` keeps table columns aligned.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def pdf_text(pdf: str | Path, out: str | Path, first: int | None = None,
             last: int | None = None) -> str:
    cmd = ["pdftotext", "-layout", "-enc", "UTF-8"]
    if first:
        cmd += ["-f", str(first)]
    if last:
        cmd += ["-l", str(last)]
    cmd += [str(pdf), str(out)]
    subprocess.run(cmd, check=True, timeout=600)
    return Path(out).read_text(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    a = sys.argv[1:]
    pdf_text(a[0], a[1], *(int(x) for x in a[2:4]))
