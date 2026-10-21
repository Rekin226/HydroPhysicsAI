"""Print the visible text and the links of a saved HTML page (untrusted data; parse only).

    python -I html_text.py <file.html> [--main] [--links] [--grep REGEX]

``--main`` keeps only the page body between the content anchor (id="c") and </main>.
"""
from __future__ import annotations

import html
import re
import sys
from html.parser import HTMLParser
from pathlib import Path


class _P(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.text: list[str] = []
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._atext: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.skip += 1
        if tag in ("br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4", "table"):
            self.text.append("\n")
        if tag in ("td", "th"):
            self.text.append(" | ")
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._atext = []

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self.skip:
            self.skip -= 1
        if tag == "a" and self._href is not None:
            self.links.append((" ".join("".join(self._atext).split()), self._href))
            self._href = None

    def handle_data(self, data):
        if self.skip:
            return
        self.text.append(data)
        if self._href is not None:
            self._atext.append(data)


def parse(path: str | Path, main_only: bool = False) -> tuple[str, list[tuple[str, str]]]:
    raw = Path(path).read_bytes().decode("utf-8", "replace")
    if main_only:
        i = raw.find('id="c"')
        j = raw.find("</main>", max(i, 0))
        if i >= 0 and j > i:
            raw = raw[i:j]
    p = _P()
    p.feed(raw)
    txt = "".join(p.text)
    txt = re.sub(r"[ \t\r\f\v]+", " ", txt)
    txt = re.sub(r"\n\s*\n+", "\n", txt)
    return txt, [(t, html.unescape(h)) for t, h in p.links]


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    txt, links = parse(argv[0], main_only="--main" in argv)
    pat = None
    if "--grep" in argv:
        pat = re.compile(argv[argv.index("--grep") + 1])
    if "--links" in argv:
        for t, h in links:
            if pat is None or pat.search(t + " " + h):
                print(f"{t} | {h}")
    else:
        for line in txt.splitlines():
            if pat is None or pat.search(line):
                print(line.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
