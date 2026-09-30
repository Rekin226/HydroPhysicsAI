"""Offline browser acceptance and reproducible screenshots of an existing twin page.

    python -m hydrophysics.twin.browser_check --page results/twin/twin_app.html --out results/browser

Requires Playwright and its separately installed Chromium. Performance describes the
tested client/renderer only; software-rendered headless timings do not certify a laptop.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .app.prep import unpack
from .release import atomic_json, read_payload, sha256


def check(page_path: Path, out: Path) -> dict:
    from playwright.sync_api import sync_playwright

    out.mkdir(parents=True, exist_ok=True)
    report = {"schema": 1, "page": page_path.name, "page_sha256": sha256(page_path),
              "viewports": []}
    payload = read_payload(page_path)
    with sync_playwright() as driver:
        browser = driver.chromium.launch(args=["--no-sandbox", "--use-angle=swiftshader",
                                               "--enable-webgl"])
        for width, height in ((1440, 1100), (390, 844)):
            context = browser.new_context(viewport={"width": width, "height": height},
                                          reduced_motion="reduce")
            remote, errors = [], []

            def block(route, remote=remote):
                remote.append(route.request.url)
                route.abort()

            context.route("http://**/*", block)
            context.route("https://**/*", block)
            page = context.new_page()
            page.on("pageerror", lambda error, errors=errors: errors.append(str(error)))
            started = time.perf_counter()
            page.goto(page_path.resolve().as_uri(), wait_until="load")
            page.wait_for_selector('body[data-ready="1"]', timeout=30000)
            load_ms = (time.perf_counter() - started) * 1000
            # Select a solved policy; measure actual repaint completion, not just JS.
            timings = page.evaluate("""async () => {
              const times=[]; for(let i=0;i<8;i++) {
                const t=performance.now(); S.f=[i%2 ? .7 : 1,1,1,1,1,1]; S.start=0;
                update(true); await new Promise(requestAnimationFrame);
                times.push(performance.now()-t);
              } return times;
            }""")
            if page.evaluate("M.yTested > M.yObs"):
                errors.append("Future calendar years labelled tested")
            page.evaluate("setYear(Y - 1)")
            if "projected, not validated" not in page.locator("#yearLab").inner_text():
                errors.append("Final projection year is not labelled unvalidated")
            challenge = payload["modelcard"].get("new_data_challenge")
            if (challenge and not challenge["beats_best_baseline"]
                    and "Accuracy gate failed" not in page.locator("#bannerText").inner_text()):
                errors.append("Failed new-data challenge is missing from the banner")
            expected = next((s for s in payload["solved"] if s["name"] == "cut30"), None)
            if expected:
                avoid = page.evaluate("KP.avoid")
                index = payload["solved"].index(expected)
                reference = -unpack(payload["arrays"][f"solvedSubs{index}"])[:, -1].mean()
                if not np.isfinite(avoid) or not np.isclose(avoid, reference, atol=0.001):
                    errors.append("Solved policy KPI disagrees with its Python field")
            page.locator("#drawer").evaluate("el => el.open = true")
            page.wait_for_function("d3 !== null", timeout=20000)
            renderer = page.evaluate("""() => {
              const gl=document.querySelector('#d3stage canvas').getContext('webgl2') ||
                document.querySelector('#d3stage canvas').getContext('webgl');
              const ext=gl.getExtension('WEBGL_debug_renderer_info');
              return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
            }""")
            page.locator("#d3Explode").check()
            page.locator("#d3Photo").uncheck()
            page.wait_for_timeout(300)
            page.screenshot(path=str(out / f"viewer-{width}.png"), full_page=True)
            if width == 1440:
                page.screenshot(path=str(out / "overview.png"))
            overflow = page.evaluate("document.documentElement.scrollWidth > innerWidth + 2")
            if overflow:
                errors.append("Horizontal page overflow")
            page.locator("#lZh").click()
            if page.locator("html").get_attribute("lang") != "zh-Hant":
                errors.append("Language switch failed")
            report["viewports"].append({"width": width, "height": height,
                                        "load_ms": round(load_ms, 1), "renderer": renderer,
                                        "policy_update_p95_ms": round(float(np.percentile(timings, 95)), 1),
                                        "remote_requests": len(remote), "errors": errors})
            context.close()
        browser.close()
        # A missing WebGL implementation must preserve the quantitative application.
        browser = driver.chromium.launch(args=["--no-sandbox", "--disable-webgl"])
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        fallback_errors = []
        page.on("pageerror", lambda error: fallback_errors.append(str(error)))
        page.goto(page_path.resolve().as_uri())
        page.wait_for_selector('body[data-ready="1"]')
        page.wait_for_function("document.getElementById('d3msg')?.textContent.includes('unavailable')")
        report["no_webgl_fallback"] = page.locator("#tiles").is_visible() and not fallback_errors
        browser.close()
    report["passed"] = report["no_webgl_fallback"] and all(
        not row["errors"] and row["remote_requests"] == 0 for row in report["viewports"])
    report["performance_note"] = "Headless software renderer; GPU-client 30 FPS target requires separate acceptance."
    atomic_json(out / "report.json", report)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--page", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    report = check(args.page, args.out)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
