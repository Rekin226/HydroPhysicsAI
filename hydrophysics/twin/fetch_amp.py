"""Rebuild the twin's data cache (``AMP_V2/data/``) from the WiseEnvr API.

    export WISENVR_BASE_URL=https://<host>/<api-root>     # never committed
    export WISENVR_USERNAME=... WISENVR_PASSWORD=...        # never committed
    python -m hydrophysics.twin.fetch_amp stations --out AMP_V2/data/fan_stations.parquet
    python -m hydrophysics.twin.fetch_amp wells --stations AMP_V2/data/fan_stations.parquet \\
        --out AMP_V2/data/wells
    python -m hydrophysics.twin.fetch_amp pumps --polygon "<fan>.json" \\
        --out AMP_V2/data/tpc_pumps.parquet --kwh-out AMP_V2/data/pump_kwh_all.parquet

Until 2026-09-11 nothing in the repository could rebuild the cache: the vendor's demo
script lives in a gitignored directory and the cache itself came from a backup. This is
the committed, credential-free equivalent. Host, account and token come **only** from the
environment; the endpoint shapes below are the vendor's public contract.

Two lessons from the fetch that built the current cache are encoded here rather than
remembered:

- **The bearer token expires** (about an hour) and the demo never renews it. Every
  request goes through ``Client.get``, which re-authenticates on a 401 and retries once.
- **Resume state records successes, not attempts.** A restart skips only what was
  actually written; anything that failed during an outage is fetched again.

What the API cannot supply: the fan polygon (``Zhuoshui Alluvial Fan.json``), the rain
gauge tables and the leveling panel used by Stage 1. Those stay in ``chou-shui-data/``
from the project's original data delivery and are documented in ``docs/DATA_FORMAT.md``.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
import urllib.parse
from dataclasses import dataclass

import pandas as pd

GW_DATASET = "gw"                       # monitoring-well water levels
PUMP_DATASET = "etc-tpc-etc1mon-obs"    # monthly kWh per registered pump
FAN_ZONE_ID = 50                        # GroundwaterZoneIdentifier of the Choushui fan


class AuthError(RuntimeError):
    pass


@dataclass
class Client:
    """Thin authenticated client. ``base_url`` is the API root (``.../<version>``); the
    token endpoint is ``<base_url>/token`` and datasets hang off ``<base_url>/``."""

    base_url: str
    username: str
    password: str
    timeout: float = 60.0
    retries: int = 5
    _token: str | None = None

    @classmethod
    def from_env(cls) -> Client:
        try:
            return cls(base_url=os.environ["WISENVR_BASE_URL"].rstrip("/"),
                       username=os.environ["WISENVR_USERNAME"],
                       password=os.environ["WISENVR_PASSWORD"])
        except KeyError as e:
            raise AuthError(f"set {e.args[0]} (and WISENVR_BASE_URL, WISENVR_USERNAME, "
                            "WISENVR_PASSWORD) in the environment; nothing is read from "
                            "the repository") from None

    # -- auth ---------------------------------------------------------------------------
    def login(self) -> None:
        import requests

        r = requests.post(f"{self.base_url}/token",
                          data={"username": self.username, "password": self.password},
                          timeout=self.timeout)
        if r.status_code != 200:
            raise AuthError(f"token request failed with HTTP {r.status_code}")
        self._token = r.json().get("access_token")
        if not self._token:
            raise AuthError("token response carried no access_token")

    def _headers(self) -> dict:
        if self._token is None:
            self.login()
        return {"Authorization": f"Bearer {self._token}"}

    def get(self, path: str, params: dict | None = None) -> bytes:
        """GET with token renewal on 401 and exponential back-off on 5xx/network errors."""
        import requests

        url = f"{self.base_url}/{path}"
        delay = 2.0
        for _attempt in range(self.retries):
            try:
                r = requests.get(url, headers=self._headers(), params=params,
                                 timeout=self.timeout)
            except requests.RequestException:
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code == 401:
                self._token = None            # expired: renew and retry
                continue
            if r.status_code >= 500:
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code == 404:
                raise FileNotFoundError(path)
            r.raise_for_status()
            return r.content
        raise RuntimeError(f"gave up on {path} after {self.retries} attempts")

    # -- endpoints ----------------------------------------------------------------------
    def datasets(self) -> list:
        return json.loads(self.get(""))

    def stations(self, dataset: str) -> pd.DataFrame:
        raw = self.get(f"{dataset}/station/", params={"orient": "parquet"})
        return pd.read_parquet(io.BytesIO(raw)).reset_index()

    def station_data(self, dataset: str, station: str, start: str | None = None,
                     end: str | None = None) -> pd.DataFrame:
        params = {"orient": "parquet"}
        if start:
            params["start_datetime"] = start
        if end:
            params["end_datetime"] = end
        raw = self.get(f"{dataset}/station/{urllib.parse.quote(str(station))}/data",
                       params=params)
        return pd.read_parquet(io.BytesIO(raw)).reset_index()


# ---------------------------------------------------------------------------------------
# resume state: successes only
# ---------------------------------------------------------------------------------------
class Resume:
    def __init__(self, path: str):
        self.path = path
        self.done: set[str] = set()
        if os.path.exists(path):
            with open(path) as fh:
                self.done = set(json.load(fh))

    def mark(self, key: str) -> None:
        self.done.add(key)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(sorted(self.done), fh)
        os.replace(tmp, self.path)


# ---------------------------------------------------------------------------------------
# the three fetches
# ---------------------------------------------------------------------------------------
def fetch_stations(client: Client, out: str) -> pd.DataFrame:
    """Every monitoring well on the fan (zone 50) with its metadata -> parquet."""
    df = client.stations(GW_DATASET)
    df = df[df["GroundwaterZoneIdentifier"] == FAN_ZONE_ID].copy()
    df["sid"] = df["sid"].astype(str)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def fetch_wells(client: Client, stations: pd.DataFrame, out_dir: str,
                start: str = "2012-01-01T00:00:00", end: str = "2023-01-01T00:00:00",
                log=print) -> int:
    """Hourly water level per well -> ``<out_dir>/<sid>.parquet`` (index datetime, column
    ``value``), the layout ``heads.build_head_field`` reads."""
    os.makedirs(out_dir, exist_ok=True)
    resume = Resume(os.path.join(out_dir, ".fetched.json"))
    n = 0
    for sid in stations["sid"].astype(str):
        if sid in resume.done:
            continue
        try:
            df = client.station_data(GW_DATASET, sid, start, end)
        except FileNotFoundError:
            resume.mark(sid)
            continue
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
        col = "value" if "value" in df.columns else df.columns[-1]
        df[[col]].rename(columns={col: "value"}).to_parquet(os.path.join(out_dir, f"{sid}.parquet"))
        resume.mark(sid)
        n += 1
        if n % 25 == 0:
            log(f"  {n} wells written")
    return n


def _inside_polygon(df: pd.DataFrame, polygon_path: str) -> pd.Series:
    from .grid import build_grid

    g = build_grid(polygon_path, dx=500.0)
    x = pd.to_numeric(df["TWD97_X"], errors="coerce")
    y = pd.to_numeric(df["TWD97_Y"], errors="coerce")
    return pd.Series([g.cell_of(float(a), float(b)) is not None if pd.notna(a) and pd.notna(b)
                      else False for a, b in zip(x, y, strict=True)], index=df.index)


def fetch_pumps(client: Client, polygon: str, out: str, kwh_out: str,
                start: str = "2012-01-01T00:00:00", end: str = "2023-01-01T00:00:00",
                log=print) -> tuple[int, int]:
    """Pump census inside the fan polygon + monthly kWh per pump -> two parquets."""
    census = client.stations(PUMP_DATASET)
    census["sid"] = census["sid"].astype(str)
    inside = _inside_polygon(census, polygon)
    census = census[inside].copy()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    census.to_parquet(out, index=False)
    log(f"census: {len(census)} pumps inside the fan")

    shard_dir = kwh_out + ".shards"
    os.makedirs(shard_dir, exist_ok=True)
    resume = Resume(os.path.join(shard_dir, ".fetched.json"))
    n = 0
    for sid in census["sid"]:
        if sid in resume.done:
            continue
        try:
            df = client.station_data(PUMP_DATASET, sid, start, end)
        except FileNotFoundError:
            resume.mark(sid)
            continue
        col = "electricity_kwh" if "electricity_kwh" in df.columns else df.columns[-1]
        part = pd.DataFrame({"datetime": pd.to_datetime(df["datetime"]),
                             "electricity_kwh": pd.to_numeric(df[col], errors="coerce"),
                             "pump": sid})
        part.to_parquet(os.path.join(shard_dir, f"{sid}.parquet"), index=False)
        resume.mark(sid)
        n += 1
        if n % 1000 == 0:
            log(f"  {n} pump series written")
    parts = [pd.read_parquet(os.path.join(shard_dir, f)) for f in sorted(os.listdir(shard_dir))
             if f.endswith(".parquet")]
    all_kwh = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["datetime", "electricity_kwh", "pump"])
    all_kwh.to_parquet(kwh_out, index=False)
    return len(census), len(all_kwh)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="rebuild AMP_V2/data from the WiseEnvr API")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stations")
    s.add_argument("--out", default="AMP_V2/data/fan_stations.parquet")
    w = sub.add_parser("wells")
    w.add_argument("--stations", default="AMP_V2/data/fan_stations.parquet")
    w.add_argument("--out", default="AMP_V2/data/wells")
    p = sub.add_parser("pumps")
    p.add_argument("--polygon", required=True)
    p.add_argument("--out", default="AMP_V2/data/tpc_pumps.parquet")
    p.add_argument("--kwh-out", default="AMP_V2/data/pump_kwh_all.parquet")
    args = ap.parse_args(argv)

    client = Client.from_env()
    if args.cmd == "stations":
        df = fetch_stations(client, args.out)
        print(f"wrote {args.out}: {len(df)} fan wells")
    elif args.cmd == "wells":
        stn = pd.read_parquet(args.stations)
        n = fetch_wells(client, stn, args.out)
        print(f"wrote {n} new well files under {args.out}")
    else:
        n_p, n_k = fetch_pumps(client, args.polygon, args.out, args.kwh_out)
        print(f"wrote {args.out} ({n_p} pumps) and {args.kwh_out} ({n_k} rows)")


if __name__ == "__main__":
    main()
