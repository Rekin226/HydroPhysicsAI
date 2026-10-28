"""WP X5: Changhua irrigation water outside the Jiji supply, and its monthly timing (Q3, Q8).

Route 4 of the 2026-10-25 data audit. The annual volume is the MOA irrigation-office total
minus the Jiji weir's Changhua delivery; the monthly timing comes from three held indicators
(LiCSAR coherence, the dry-season river balance, shallow-head change) calibrated on Yunlin,
then Denton-benchmarked to the annual volume. Code only; outputs go to
results/dataprep/x5_changhua_water/ and results/dataprep/reports/.
"""
from __future__ import annotations
