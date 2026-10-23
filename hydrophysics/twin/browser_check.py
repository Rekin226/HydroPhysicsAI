"""Offline browser acceptance and reproducible screenshots of an existing twin page.

    python -m hydrophysics.twin.browser_check --page results/twin/twin_app.html --out results/browser \
        [--page-no-three results/twin/twin_app_nothree.html]

Requires Playwright and its separately installed Chromium (``TWIN_CHROMIUM`` names the binary when
Playwright's own download is absent). It clicks through the story, every tab, a preset, the year
slider, a pin, both languages, both themes and the 3D drawer, and fails on any console error.
Performance describes the tested client/renderer only; software-rendered headless timings do not
certify a laptop.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .app.prep import unpack
from .release import atomic_json, read_payload, sha256


def _exe() -> dict:
    """TWIN_CHROMIUM names a Chromium binary when Playwright's own download is absent."""
    path = os.environ.get("TWIN_CHROMIUM")
    return {"executable_path": path} if path else {}


_P95 = """async (n) => {
  const times = [], el = document.getElementById('sl0');
  for (let i = 0; i < n; i++) {
    const t = performance.now(); el.value = i % 2 ? 30 : 0;
    el.dispatchEvent(new Event('input', {bubbles: true}));
    await new Promise(requestAnimationFrame); await new Promise(requestAnimationFrame);
    times.push(performance.now() - t);
  }
  el.value = 0; el.dispatchEvent(new Event('input', {bubbles: true})); el.dispatchEvent(new Event('change', {bubbles: true}));
  return {wall: times, flush: window.__flushMs.slice(-n)};
}"""


def _walk(page, errors, width):
    """Drive every feature once. Each assertion appends to ``errors``."""
    def need(cond, msg):
        if not cond:
            errors.append(msg)

    # on a phone the policy bar is a sticky sheet: open it
    if page.locator('#polToggle').is_visible():
        page.locator('#polToggle').click()
    # preset: the headline names what it avoids
    page.locator('#presets button').first.click()
    page.wait_for_function("document.getElementById('headline').textContent.includes('avoids')", timeout=5000)
    # the year slider: one timeline, the headline follows it
    page.evaluate("() => { const y = document.getElementById('year'); y.value = 14; y.dispatchEvent(new Event('input', {bubbles: true})); }")
    page.wait_for_function("document.getElementById('headline').textContent.includes('2026')", timeout=5000)
    page.locator("#yearEnd").click()
    need("2032" in page.locator("#headline").inner_text(), "Year-end button did not move the headline")
    # the three others of the six numbers are one click away
    page.locator("#moreBtn").click()
    need(page.locator("#tilesMore").is_visible(), "More metrics did not open")
    page.locator('#tA .qbtn').click()
    need(page.locator("#pop").is_visible(), "Tile definition popover did not open")
    page.keyboard.press("Escape")
    need(not page.locator("#pop").is_visible(), "Escape did not close the popover")
    # every tab has content
    for tid in ("tTowns", "tHsr", "tSec", "tCmp", "tTrust", "tRank"):
        page.locator("#" + tid).click()
        panel = "p" + tid[1:]
        need(page.locator("#" + panel).is_visible(), f"Tab {tid} did not open")
        need(len(page.locator("#" + panel).inner_text().strip()) > 20, f"Tab {tid} is empty")
    # a pin, then the pin limit toasts
    page.locator("#pinBtn").click()
    need("1/4" in page.locator("#pinCount").inner_text(), "Pin did not register")
    for _ in range(4):
        page.locator("#pinBtn").click()
    need("oldest" in page.locator("#toast").inner_text() or "最舊" in page.locator("#toast").inner_text(),
         "Dropping a pin did not toast")
    # settings popover, keyboard sheet, model card, evidence
    page.locator("#setBtn").click()
    need(page.locator("#setPop").is_visible(), "Settings did not open")
    page.keyboard.press("Escape")
    page.locator("#kbBtn").click()
    need(page.locator("#kbDlg").is_visible(), "Keyboard sheet did not open")
    page.keyboard.press("Escape")
    page.locator("#evBadge").click()
    card = page.locator("#cardBody").inner_text()
    need("Evidence" in card and "Research scenario" in card, "Model card lacks the evidence layer")
    page.keyboard.press("Escape")
    # section drawing can be cancelled with Escape
    page.locator("#tSec").click()
    page.locator("#secDraw").click()
    need(page.locator("#drawHint").is_visible(), "Draw hint missing")
    page.keyboard.press("Escape")
    need(not page.locator("#drawHint").is_visible(), "Escape did not cancel drawing")
    page.locator("#tTowns").click()
    # theme
    light_bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
    page.locator("#themeBtn").click()
    need(page.evaluate("document.documentElement.dataset.theme") == "dark", "Theme did not switch")
    need(page.evaluate("getComputedStyle(document.body).backgroundColor") != light_bg, "Dark theme did not change the page")
    page.screenshot(path=str(Path(errors.out) / f"dark-{width}.png"))
    page.locator("#themeBtn").click()
    # language
    page.locator("#lZh").click()
    need(page.locator("html").get_attribute("lang") == "zh-Hant", "Language switch failed")
    need(any("一" <= ch <= "鿿" for ch in page.locator("#headline").inner_text()), "Headline not translated")
    page.screenshot(path=str(Path(errors.out) / f"zh-{width}.png"))
    keys = page.evaluate("[Object.keys(T.en).filter(k => !(k in T.zh)), Object.keys(T.zh).filter(k => !(k in T.en)), [...MISSING]]")
    need(not keys[0] and not keys[1], f"Dictionary keys differ: {keys[0][:5]} {keys[1][:5]}")
    need(not keys[2], f"Missing dictionary keys used: {keys[2][:5]}")
    page.locator("#lEn").click()
    # story: six steps with the keyboard, then Skip restores the reader's settings
    page.locator("#storyBtn").click()
    need(page.locator("#story").is_visible(), "Story did not open")
    for _ in range(5):
        page.keyboard.press("ArrowRight")
    page.wait_for_timeout(200)
    need("6 / 6" in page.locator("#stStep").inner_text(), "Story did not reach step 6")
    page.locator("#stSkip").click()
    need(not page.locator("#story").is_visible(), "Story did not close")


class _Errors(list):
    out = "."


def check(page_path: Path, out: Path, no_three: Path | None = None) -> dict:
    from playwright.sync_api import sync_playwright

    out.mkdir(parents=True, exist_ok=True)
    report = {"schema": 2, "page": page_path.name, "page_sha256": sha256(page_path),
              "page_bytes": page_path.stat().st_size, "viewports": []}
    payload = read_payload(page_path)
    with sync_playwright() as driver:
        browser = driver.chromium.launch(args=["--no-sandbox", "--use-angle=swiftshader",
                                               "--enable-webgl"], **_exe())
        for width, height in ((1440, 1100), (390, 844)):
            context = browser.new_context(viewport={"width": width, "height": height},
                                          reduced_motion="reduce")
            remote, errors = [], _Errors()
            errors.out = str(out)

            def block(route, remote=remote):
                remote.append(route.request.url)
                route.abort()

            context.route("http://**/*", block)
            context.route("https://**/*", block)
            page = context.new_page()
            page.on("pageerror", lambda error, errors=errors: errors.append("pageerror: " + str(error)))
            page.on("console", lambda msg, errors=errors: errors.append("console: " + msg.text)
                    if msg.type == "error" else None)
            started = time.perf_counter()
            page.goto(page_path.resolve().as_uri(), wait_until="load")
            page.wait_for_selector('body[data-ready="1"]', timeout=30000)
            load_ms = (time.perf_counter() - started) * 1000
            if page.evaluate("typeof THREE") != "undefined":
                errors.append("Three.js was evaluated on load")
            if page.evaluate("document.getElementById('drawer').open"):
                errors.append("The 3D drawer opened by itself")
            page.screenshot(path=str(out / f"overview-{width}.png"))
            # a lever moved on the real slider: measure completion of the repaint, not just JS
            t = page.evaluate(_P95, 12)
            wall, flush = np.array(t["wall"]), np.array(t["flush"])
            if page.evaluate("M.yTested > M.yObs"):
                errors.append("Future calendar years labelled tested")
            page.evaluate("setYear(Y - 1)")
            page.wait_for_timeout(100)
            if "projected, not validated" not in page.locator("#yearLab").inner_text():
                errors.append("Final projection year is not labelled unvalidated")
            challenge = payload["modelcard"].get("new_data_challenge")
            if (challenge and not challenge["beats_best_baseline"]
                    and "not predictions" not in page.locator("#evLine").inner_text()):
                errors.append("Failed new-data challenge is missing from the evidence line")
            expected = next((s for s in payload["solved"] if s["name"] == "cut30"), None)
            if expected:
                page.evaluate("S.f = [0.7, 1, 1, 1, 1, 1]; S.start = 0; update(true)")
                page.wait_for_timeout(150)
                avoid = page.evaluate("KP.avoid")
                index = payload["solved"].index(expected)
                reference = -unpack(payload["arrays"][f"solvedSubs{index}"])[:, -1].mean()
                if not np.isfinite(avoid) or not np.isclose(avoid, reference, atol=0.001):
                    errors.append("Solved policy KPI disagrees with its Python field")
                page.evaluate("S.f = [1, 1, 1, 1, 1, 1]; update(true)")
            # the 3D drawer: built on first open only
            page.locator("#drawer summary").click()
            page.wait_for_function("d3 !== null", timeout=30000)
            renderer = page.evaluate("""() => {
              const gl=document.querySelector('#d3stage canvas').getContext('webgl2') ||
                document.querySelector('#d3stage canvas').getContext('webgl');
              const ext=gl.getExtension('WEBGL_debug_renderer_info');
              return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
            }""")
            page.locator("#d3Explode").check()
            page.locator("#d3Photo").uncheck()
            page.wait_for_timeout(300)
            t3 = page.evaluate(_P95, 8)
            page.screenshot(path=str(out / f"viewer-{width}.png"), full_page=True)
            if page.evaluate("document.documentElement.scrollWidth > innerWidth + 2"):
                errors.append("Horizontal page overflow")
            _walk(page, errors, width)
            report["viewports"].append({
                "width": width, "height": height, "load_ms": round(load_ms, 1), "renderer": renderer,
                "policy_update_p95_ms": round(float(np.percentile(wall, 95)), 1),
                "policy_flush_p95_ms": round(float(np.percentile(flush, 95)), 1),
                "policy_update_3d_open_p95_ms": round(float(np.percentile(t3["wall"], 95)), 1),
                "remote_requests": len(remote), "errors": list(errors)})
            context.close()
        browser.close()
        # A missing WebGL implementation must preserve the quantitative application.
        browser = driver.chromium.launch(args=["--no-sandbox", "--disable-webgl"], **_exe())
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        fallback_errors = []
        page.on("pageerror", lambda error: fallback_errors.append(str(error)))
        page.goto(page_path.resolve().as_uri())
        page.wait_for_selector('body[data-ready="1"]')
        page.locator("#drawer summary").click()
        page.wait_for_function("document.getElementById('d3msg')?.textContent.includes('unavailable')")
        report["no_webgl_fallback"] = page.locator("#tiles").is_visible() and not fallback_errors
        browser.close()
        # A page built with --three none loads, and its drawer says so.
        report["no_three_page"] = None
        if no_three is not None:
            browser = driver.chromium.launch(args=["--no-sandbox"], **_exe())
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            none_errors = []
            page.on("pageerror", lambda error: none_errors.append(str(error)))
            page.on("console", lambda msg: none_errors.append(msg.text) if msg.type == "error" else None)
            page.goto(no_three.resolve().as_uri())
            page.wait_for_selector('body[data-ready="1"]')
            page.locator("#drawer summary").click()
            page.wait_for_function("document.getElementById('d3msg')?.textContent.includes('no 3D')")
            report["no_three_page"] = {"bytes": no_three.stat().st_size, "errors": none_errors}
            browser.close()
    report["passed"] = (report["no_webgl_fallback"]
                        and all(not row["errors"] and row["remote_requests"] == 0 for row in report["viewports"])
                        and (report["no_three_page"] is None or not report["no_three_page"]["errors"]))
    report["performance_note"] = "Headless software renderer; GPU-client 30 FPS target requires separate acceptance."
    atomic_json(out / "report.json", report)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--page", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--page-no-three", type=Path, help="the same page built with --three none")
    args = ap.parse_args(argv)
    report = check(args.page, args.out, args.page_no_three)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
