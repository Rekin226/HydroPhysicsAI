"""WP S3 v2: look-log unit ids and their members (``harness.prereg3`` requires members for
units outside the validation-unit registries). Frozen with the v2 pre-registration: the
acceptance run imports it, so it is a registered input; add new helpers elsewhere.
"""
from __future__ import annotations

SYN = ("synthetic step injections on GNSS height series; no registered well, leveling "
       "benchmark, fold test set or head-data cell")
A2_PAIRS = [("GFES", "光復國小"), ("HLES", "宏崙國小"), ("HNES", "湖南國小"), ("JMES", "拯民國小"),
            ("KTES", "客厝國小"), ("SJES", "溪州國小"), ("SSES", "新生國小"), ("STES", "秀潭國小"),
            ("TKJS", "土庫國中"), ("YCES", "元長國小"), ("GFES", "NTUH_282_29"), ("SJES", "SJES"),
            ("STES", "STES"), ("TKJS", "TKJS_281_29")]
A2_UNITS = [f"s3.pair.{g}-{r}" for g, r in A2_PAIRS]


def a2_members(window: str) -> dict:
    """Members of the A2 pair units: the GNSS station and its ring or pile, in ``window``."""
    return {f"s3.pair.{g}-{r}": {"ids": [f"gnss:{g}", f"ring_or_pile:{r}"], "window": window,
                                 "description": "co-located WRA GNSS station and compaction "
                                                "ring or deep pile (2021 extra compaction)"}
            for g, r in A2_PAIRS}


def seed_members(units: list[str]) -> dict:
    """Members of synthetic injection units: none (simulation draws)."""
    return {u: {"none": SYN, "description": SYN} for u in units}
