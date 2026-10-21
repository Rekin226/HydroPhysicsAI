"""WP X2: public-supply groundwater abstraction from public sources (Changhua, Yunlin).

Scrapes and transcribes public statistics (WRA water-use statistics, Taiwan Water
Corporation statistics, MOEA/WRA subsidence-programme reports) for the groundwater that
public supply abstracts in Changhua and Yunlin, validates each number against the
document's own totals, compares the result with the P6 closure-ramp member and builds
gridded variants. Downloads are untrusted data: only read by ``python -I`` scripts here.
"""
from __future__ import annotations

__version__ = "1.0.0"
