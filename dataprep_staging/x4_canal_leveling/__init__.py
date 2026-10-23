"""WP X4: canal water below the district scale, and public subsidence results after 2021.

Two parts, both from public web pages and documents only:

* canal: irrigation data finer than the irrigation district (work station, canal system,
  township planted area, rotation and fallowing records) and a product that redistributes
  the W1 district canal series spatially;
* leveling: the WRA annual leveling (水準檢測) results printed in public briefings and
  reports, as a report-only post-origin check (U7) plus a pre-origin part for older years.

Code only; downloads go to data_fetch_api/new_measurements/<source_id>/<YYYY-MM-DD>/ and
products to results/dataprep/<product>/v<N>/.
"""
from __future__ import annotations
