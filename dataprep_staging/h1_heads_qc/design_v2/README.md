# H1a round-2 design-time scripts

Exploration scripts of the round-2 design (v2 decision layer). They ran on pre-origin data
only (2012-22) and on the family O injections r = 0..9, which v1 had already scored, so every
result they produced is design-time evidence (logged in the A1 look log; outputs in
`results/dataprep/reports/H1a_evidence/v2/design_time/`). They are kept for provenance; the
product is built by `run_v2.sh`. `x*` scripts ran against the first v2 version (stage-1 pairs
re-checked, design rounds d1a/d1b); `y*` scripts against the later versions (d2, d3).
Run with `python -I` in a CPU slot: `script.py WORKDIR ...` (see each script).

The x* scripts call the first v2 version of `decide_v2.decide_well(res, dec_v1, ...)`, whose
source was overwritten by the later design (its rules are recorded in the `decide_v2.py`
design record and in the round-2 section of the WP report); they no longer run against the
current module. The y* scripts run against the current module.
