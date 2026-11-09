"""Public launcher for the twin analyst on a Hugging Face Space (chat + decision page).

Runs the analyst web shell (``hydrophysics_analyst.server``) bound to the promoted page
``results/twin/twin_app.html`` and the claims ledger of the same commit, strictly
(``mode="deploy"``: no ``--unbound``, no ``--allow-pending``).

Open access is a deliberate deployment decision (user, 2026-11-09): the server's own
``make_server`` refuses a non-loopback bind without per-user tokens or a proxy, so this
launcher builds the ``ThreadingHTTPServer`` itself, with no user list (every visitor is the
one anonymous user). Only the public page ships (aggregates; no individual observations).

Cost guard: typed questions call Claude (one structured call each; example questions and the
form never do). At most ``ANALYST_DAILY_LLM_CAP`` uncached calls per UTC day (default 300);
past the cap the backend raises, and the router's existing failure path answers with the
"use an example question or the form" clarification, never a preset. No
``ANTHROPIC_API_KEY`` secret means the same examples-and-form-only mode.
"""

from __future__ import annotations

import datetime as dt
import os
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

from hydrophysics_analyst.ask import make_llm
from hydrophysics_analyst.llm.backend import AnthropicBackend, BackendUnavailable
from hydrophysics_analyst.page import PageBundle
from hydrophysics_analyst.server import App, make_handler

PAGE = Path("results/twin/twin_app.html")
MODEL = os.environ.get("ANALYST_MODEL", "claude-opus-5-5")
CAP = int(os.environ.get("ANALYST_DAILY_LLM_CAP", "300"))
PORT = int(os.environ.get("PORT", "7860"))


class DailyCap:
    """Pass ``complete`` through to the real backend until the day's cap is reached."""

    def __init__(self, inner, cap: int):
        self.inner, self.cap = inner, cap
        self.day, self.n = dt.datetime.now(dt.timezone.utc).date(), 0
        self.lock = threading.Lock()

    def complete(self, *args, **kwargs):
        with self.lock:
            today = dt.datetime.now(dt.timezone.utc).date()
            if today != self.day:
                self.day, self.n = today, 0
            if self.n >= self.cap:
                raise BackendUnavailable(f"daily LLM cap reached ({self.cap})")
            self.n += 1
        return self.inner.complete(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def main() -> None:
    b = PageBundle.load(PAGE, None, mode="deploy")
    use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    llm = None
    if use_llm:
        backend = DailyCap(AnthropicBackend(MODEL, fallbacks=False), CAP)
        llm = make_llm(b, backend=backend)
    app = App(b, use_llm=use_llm, llm=llm, users={}, behind_proxy=False,
              audit_dir=Path("results/analyst/audit"),
              llm_config={"model": MODEL, "fallbacks": False})
    app.embeddable = False  # as make_server sets for any non-loopback bind
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), make_handler(app))
    print(f"twin analyst on :{PORT} (page {b.page_sha256[:12]}, ledger {b.ledger_sha256[:12]}, "
          f"llm {'on, cap ' + str(CAP) + '/day' if use_llm else 'off'})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
