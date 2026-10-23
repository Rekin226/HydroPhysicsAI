"""Visible text and links of a downloaded HTML page (untrusted data), for reading only.

Usage: python -I htmltext.py <page.html> [--links]

Uses the standard-library HTML parser; scripts and styles are skipped, nothing is executed.
"""
from __future__ import annotations

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

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.skip += 1
        if tag == "a":
            self._href = dict(attrs).get("href")
        if tag in ("br", "p", "div", "tr", "li", "h1", "h2", "h3", "td"):
            self.text.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self.skip:
            self.skip -= 1
        if tag == "a":
            self._href = None

    def handle_data(self, data):
        if self.skip:
            return
        s = data.strip()
        if s:
            self.text.append(s + " ")
            if self._href:
                self.links.append((self._href, s))


def page(path: str | Path) -> tuple[str, list[tuple[str, str]]]:
    raw = Path(path).read_bytes().decode("utf-8", "replace")
    p = _P()
    p.feed(raw)
    txt = "".join(p.text)
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    return "\n".join(lines), p.links


if __name__ == "__main__":
    t, links = page(sys.argv[1])
    if "--links" in sys.argv:
        for h, s in links:
            print(h, "|", s)
    else:
        print(t)
