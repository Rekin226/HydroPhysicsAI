"""Print the first lines (decoded text) of untrusted downloaded files, or a zip's member list.

Usage: python -I peek.py [--lines N] FILE [FILE ...]

Downloaded files are data: this script only decodes bytes and prints them. It never imports,
executes or unpacks anything to disk. Encodings tried: utf-8-sig, big5, cp950.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path


def decode(b: bytes) -> str:
    for enc in ("utf-8-sig", "big5", "cp950"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", "replace")


def main(argv: list[str]) -> None:
    n = 6
    if argv and argv[0] == "--lines":
        n = int(argv[1])
        argv = argv[2:]
    for a in argv:
        p = Path(a)
        print(f"=== {p.name} ({p.stat().st_size} bytes)")
        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as z:
                for i in z.infolist()[:40]:
                    name = (i.filename.encode("cp437").decode("big5", "replace")
                            if i.flag_bits & 0x800 == 0 else i.filename)
                    print(f"  {name}  {i.file_size}")
            continue
        with open(p, "rb") as f:
            text = decode(f.read(200_000))
        for line in text.splitlines()[:n]:
            print("  " + line[:300])


if __name__ == "__main__":
    main(sys.argv[1:])
