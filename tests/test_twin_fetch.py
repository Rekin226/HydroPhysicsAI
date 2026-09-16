"""The committed data fetcher, exercised against a fake API: credentials come only from
the environment, an expired token is renewed on 401, and resume state records successes
rather than attempts."""

from __future__ import annotations

import io
import json
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from hydrophysics.twin.fetch_amp import (  # noqa: E402
    GW_DATASET,
    AuthError,
    Client,
    Resume,
    fetch_stations,
    fetch_wells,
)


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf)
    return buf.getvalue()


class _Resp:
    def __init__(self, status: int, content: bytes = b"", payload=None):
        self.status_code = status
        self.content = content
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeRequests:
    """Serves a token endpoint, a station list and per-station data; the first data call
    after a token is issued answers 401 once to simulate expiry."""

    RequestException = RuntimeError

    def __init__(self):
        self.tokens_issued = 0
        self.calls = []
        self.expire_next = False

    def post(self, url, data=None, timeout=None):
        assert url.endswith("/token") and data["username"] == "u"
        self.tokens_issued += 1
        self.expire_next = self.tokens_issued == 1
        return _Resp(200, payload={"access_token": f"tok{self.tokens_issued}"})

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append(url)
        assert headers["Authorization"].startswith("Bearer tok")
        if self.expire_next and "/data" in url:
            self.expire_next = False
            return _Resp(401)
        if url.endswith(f"/{GW_DATASET}/station/"):
            df = pd.DataFrame({"sid": ["1", "2", "3"], "GroundwaterZoneIdentifier": [50, 50, 7]})
            return _Resp(200, content=_parquet_bytes(df))
        if "/station/2/data" in url:
            return _Resp(404)
        if "/station/1/data" in url:
            idx = pd.date_range("2012-01-01", periods=3, freq="h")
            df = pd.DataFrame({"datetime": idx, "value": [1.0, 2.0, np.nan]})
            return _Resp(200, content=_parquet_bytes(df))
        return _Resp(500)


@pytest.fixture
def fake(monkeypatch):
    fr = _FakeRequests()
    import hydrophysics.twin.fetch_amp as mod

    monkeypatch.setattr(mod, "requests", fr, raising=False)
    # the module imports requests lazily inside methods; make `import requests` find ours
    import sys

    monkeypatch.setitem(sys.modules, "requests", fr)
    return fr


def test_client_needs_the_environment_and_never_the_repo(monkeypatch):
    for k in ("WISENVR_BASE_URL", "WISENVR_USERNAME", "WISENVR_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(AuthError, match="environment"):
        Client.from_env()
    monkeypatch.setenv("WISENVR_BASE_URL", "https://example.invalid/api/")
    monkeypatch.setenv("WISENVR_USERNAME", "u")
    monkeypatch.setenv("WISENVR_PASSWORD", "p")
    c = Client.from_env()
    assert c.base_url == "https://example.invalid/api"


def test_stations_wells_token_renewal_and_resume(fake, tmp_path):
    c = Client(base_url="https://example.invalid/api", username="u", password="p",
               retries=3)
    stn = fetch_stations(c, str(tmp_path / "fan_stations.parquet"))
    assert list(stn["sid"]) == ["1", "2"]                 # zone 50 only
    assert fake.tokens_issued == 1

    out = tmp_path / "wells"
    n = fetch_wells(c, stn, str(out), log=lambda *_: None)
    assert n == 1
    assert fake.tokens_issued == 2                        # renewed after the 401
    df = pd.read_parquet(out / "1.parquet")
    assert list(df.columns) == ["value"] and len(df) == 3
    assert not (out / "2.parquet").exists()               # 404: skipped, but recorded
    with open(out / ".fetched.json") as fh:
        done = set(json.load(fh))
    assert done == {"1", "2"}

    # a second pass fetches nothing: both are recorded as handled
    calls_before = len(fake.calls)
    assert fetch_wells(c, stn, str(out), log=lambda *_: None) == 0
    assert len(fake.calls) == calls_before


def test_resume_records_successes_only(tmp_path):
    r = Resume(str(tmp_path / "state.json"))
    assert r.done == set()
    r.mark("a")
    assert Resume(str(tmp_path / "state.json")).done == {"a"}
    assert not os.path.exists(str(tmp_path / "state.json.tmp"))
