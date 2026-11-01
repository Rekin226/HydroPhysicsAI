"""Table-3 well columns: the printed head fallback of N3 (schema and acceptance rule, fixed now).

Table 3 of the WRA monthly report (深層樁與地下水位變化趨勢比較表) interleaves, next to each
deep pile, the neighbouring observation wells' month and trailing-year head changes in
METRES. twin.report_extract tells the model to ignore those columns; this module is the
companion transcription for them, used only if the primary head driver (the window heads
product granting prospective_scorer, N2's feed) leaves a primary pile with fewer than
MIN_MONTHS months of nest dD.

Acceptance rule (the same two-read discipline as the piles): the page is transcribed twice,
independently (PDF by one model, poppler layout text by another; a third read by the PDF
model on the text settles a two-way disagreement); a value is accepted only if two reads
agree number for number (:func:`accept`). The trailing-year chain Y(m) - Y(m-1) = M(m) -
M(m-12) within CHAIN_TOL_M is a diagnostic, not a veto (as for the piles). Mapping (rule,
fixed now): a printed well name maps to the L1 station whose WellName equals it after
:func:`normalise` and which lies within 3 km of the pile; otherwise it is unmapped and
unused. The model layer is the station's layer from the page model's well list (or the
station metadata's layer_code). The model never computes, fills or infers a number.

Sign convention (declared now, without reading any value): the scorer works with the head
change h(m) - h(m-1) in metres, POSITIVE = the water level ROSE (n3lib.nest_dd_from_changes
turns it into dD = minus that). Which sign the table prints is not assumed: every
transcription copies the table's own sign legend into ``positive_means`` ('rise' or 'fall',
as the 007001 transcription copies its pile legend; 'not_printed' if the page prints none),
both reads must agree on it, and :func:`head_change` converts a printed value to the scorer's
convention. A table whose legend is 'not_printed' or missing is refused by the v3 builder
(build_v3_manifest.well_rows) and never used: no fallback is better than a guessed sign.
"""
from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

CHAIN_TOL_M = 0.021          # four values printed to 0.01 m
MAP_RADIUS_M = 3000.0
SIGN_LEGENDS = ("rise", "fall")          # usable legends; 'not_printed' is refused

SCHEMA_WELLS = {
    "type": "object",
    "additionalProperties": False,
    "required": ["report_roc_year", "report_month", "positive_means", "wells"],
    "properties": {
        "report_roc_year": {"type": "integer", "description": "民國 year in the table title"},
        "report_month": {"type": "integer", "description": "month in the table title"},
        "positive_means": {"type": "string", "enum": ["rise", "fall", "not_printed"],
                           "description": "the table's own legend for the sign of the "
                                          "地下水位 columns: 'rise' if it states that a "
                                          "positive value means the water level rose, "
                                          "'fall' if it states the opposite, 'not_printed' "
                                          "if no such legend is printed"},
        "wells": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["pile_name", "well_name", "month_change_m", "year_change_m"],
                "properties": {
                    "pile_name": {"type": "string",
                                  "description": "樁名 of the row the well is printed next "
                                                 "to, without the depth"},
                    "well_name": {"type": "string", "description": "井名 exactly as printed"},
                    "month_change_m": {"type": ["number", "null"],
                                       "description": "地下水位 月變化, metres, sign as printed; "
                                                      "null if printed as —"},
                    "year_change_m": {"type": ["number", "null"],
                                      "description": "地下水位 年變化, metres, sign as printed"},
                },
            },
        },
    },
}

SYSTEM_WELLS = (
    "You transcribe one printed table from a Taiwanese government report. Copy ONLY the "
    "observation-well (井名) columns of 表 3: for every well printed next to a deep benchmark "
    "row, the pile name of that row, the well name exactly as printed, and the two numbers "
    "under 地下水位 月變化 and 年變化 (metres, sign as printed). Copy the table's own sign "
    "legend for those columns into positive_means ('rise' or 'fall'); if no legend is "
    "printed, return 'not_printed' - never guess it. Ignore the deep-benchmark (深層樁) "
    "numbers. Never compute, round, infer or fill a number: if a value is not printed, "
    "return null. Return only the JSON."
)


def normalise(name: str) -> str:
    """NFKC (full-width brackets and digits to ASCII), no blanks."""
    return "".join(unicodedata.normalize("NFKC", str(name)).split())


def rows_from(table: dict, month: str, source: str) -> pd.DataFrame:
    pm = table.get("positive_means") or "not_printed"
    rows = [{"month": month, "pile": str(w.get("pile_name", "")).strip(),
             "well": normalise(w.get("well_name", "")),
             "month_m": w.get("month_change_m"), "year_m": w.get("year_change_m"),
             "positive_means": pm, "source": source} for w in table.get("wells", [])]
    return pd.DataFrame(rows, columns=["month", "pile", "well", "month_m", "year_m",
                                       "positive_means", "source"])


def head_change(value, positive_means) -> float:
    """A printed well-column value -> h(m) - h(m-1) (m, positive = rose), from the table's
    own legend; NaN for an unusable legend."""
    if value is None or pd.isna(value):
        return float("nan")
    if positive_means == "rise":
        return float(value)
    if positive_means == "fall":
        return -float(value)
    return float("nan")


def _same(x, y) -> bool:
    if pd.isna(x) and pd.isna(y):
        return True
    if pd.isna(x) or pd.isna(y):
        return False
    return abs(float(x) - float(y)) < 1e-9


def accept(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame | None = None) -> pd.DataFrame:
    """Two-read acceptance per (month, pile, well): 'verified' if read a agrees with read b,
    or (when they disagree) with the third read c, on both numbers."""
    key = ["month", "pile", "well"]
    cols = ["month_m", "year_m", "positive_means"]
    m = a.merge(b[key + cols], on=key, how="left", suffixes=("", "_b"))
    m["verified"] = [_same(r.month_m, r.month_m_b) and _same(r.year_m, r.year_m_b)
                     and r.positive_means == r.positive_means_b for r in m.itertuples()]
    if c is not None and len(c):
        m = m.merge(c[key + cols].rename(
            columns={k: f"{k}_c" for k in cols}), on=key, how="left")
        third = [_same(r.month_m, r.month_m_c) and _same(r.year_m, r.year_m_c)
                 and r.positive_means == r.positive_means_c for r in m.itertuples()]
        m["verified"] = m["verified"] | np.asarray(third, bool)
    m["flag"] = np.where(m["verified"], np.where(m["month_m"].isna(), 8, 0), 9)
    return m[key + cols + ["verified", "flag"]]


def chain_check(df: pd.DataFrame, tol: float = CHAIN_TOL_M) -> pd.DataFrame:
    """Diagnostic: Y(m) - Y(m-1) - (M(m) - M(m-12)) per (pile, well)."""
    df = df.copy()
    df["per"] = pd.PeriodIndex(df["month"], freq="M")
    idx = df.set_index(["pile", "well", "per"])
    res = []
    for r in df.itertuples():
        k = (r.pile, r.well)
        try:
            v = [idx.loc[k + (r.per,), "year_m"], idx.loc[k + (r.per - 1,), "year_m"],
                 idx.loc[k + (r.per,), "month_m"], idx.loc[k + (r.per - 12,), "month_m"]]
            res.append(float(v[0]) - float(v[1]) - (float(v[2]) - float(v[3]))
                       if not any(pd.isna(x) for x in v) else np.nan)
        except KeyError:
            res.append(np.nan)
    df["chain_resid"] = res
    df["chain_ok"] = [None if np.isnan(x) else bool(abs(x) <= tol) for x in res]
    return df.drop(columns="per")


def map_wells(printed: pd.Series, stations: pd.DataFrame, pile_xy: tuple[float, float],
              radius_m: float = MAP_RADIUS_M) -> dict[str, str | None]:
    """{printed well name: station sid or None}: exact normalised WellName match among the
    stations within ``radius_m`` of the pile; ambiguous or absent -> None."""
    st = stations.copy()
    st["d"] = np.hypot(st["x_twd97"] - pile_xy[0], st["y_twd97"] - pile_xy[1])
    st = st[st["d"] <= radius_m]
    names = st.assign(n=st["WellName"].map(normalise))
    out = {}
    for w in sorted(set(map(normalise, printed))):
        hit = names[names["n"] == w]
        out[w] = str(hit["sid"].iloc[0]) if len(hit) == 1 else None
    return out


def changes_from_printed(acc: pd.DataFrame, pile: str, sid_of: dict, layer_of: dict
                         ) -> tuple[dict, dict]:
    """Per mapped well, the verified printed monthly head changes converted to h(m) - h(m-1)
    (m, positive = rose; :func:`head_change`) for :func:`n3lib.nest_dd_from_changes`;
    rows without a usable legend are dropped. Returns ({sid: Series by month},
    {sid: layer})."""
    changes, layers = {}, {}
    sub = acc[(acc["pile"] == pile) & acc["verified"] & acc["month_m"].notna()
              & acc["positive_means"].isin(SIGN_LEGENDS)]
    for w, g in sub.groupby("well"):
        sid = sid_of.get(w)
        if sid is None or sid not in layer_of:
            continue
        hc = [head_change(v, pm) for v, pm in zip(g["month_m"], g["positive_means"],
                                                  strict=True)]
        changes[sid] = pd.Series(hc, index=g["month"].to_numpy(), dtype=float).sort_index()
        layers[sid] = int(layer_of[sid])
    return changes, layers
