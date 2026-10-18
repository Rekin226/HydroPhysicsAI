"""WP S5 repair round 2 (R6): twin-cell values of S4br's repaired pre-origin LiCSBAS run P2.

Run P2 (S4br: multi-prime loop closure with per-pixel nullification, LiCSBAS16 filter, stated
off-fan offset + height reference, one connected network) replaces round 1's run P in every
pre-origin S5 fit. Processing as insar_069A/v3 for its primary frame: s4br_lib.processed("P2")
(vertical-equivalent LOS / U), median of the valid multilooked pixels per active cell (>= 5
pixels). A pixel contributes to an epoch only while its LiCSBAS13 chain from the first epoch is
unbroken (no per-pixel gap before that epoch); so every cell value is tied to the first epoch
(the semantics of insar_069A/v3 flag 0). Epochs whose stable set fails (flag 9) are missing.
No GNSS, leveling, ring or pile value is used.

    tools/cpu_slot.sh python -I dataprep_staging/s5_fusion/s5r2_insarP2.py

Writes results/dataprep/reports/S5_evidence/repair_r2/work/insar_runP2_cells.npz (+ .json).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "s4_insar"))

import numpy as np  # noqa: E402
import s4b_lib as Lb  # noqa: E402
import s4br_config as RCfg  # noqa: E402
import s4br_lib as RL  # noqa: E402
import s5r2_const as K  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402


def main() -> int:
    K.WORK.mkdir(parents=True, exist_ok=True)
    out = K.INSAR_P2_CELLS
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    A = Lb.n_active()
    run, g, v, tab = RL.processed("P2")
    T = v.shape[0]
    seg = np.zeros(v.shape, np.int16)
    if run["gap"] is not None:
        seg[1:] = np.cumsum(run["gap"].astype(np.int16), axis=0)
    v = np.where(seg == 0, v, np.nan).astype(np.float32)
    val, _se, nval = Lb.to_cells(v, run["mask"], g, A)          # (T, A)
    eflag = tab["flag"].to_numpy()[:, None]
    flag = np.where(np.isfinite(val), np.where(eflag == 9, 9, 0), 8).astype(np.uint8)
    dates = np.array([d.strftime("%Y-%m-%d") for d in run["dates"]])
    np.savez_compressed(
        out, value=val.T.astype(np.float32), flag=flag.T,
        segment=np.zeros((A, T), np.int16), network_segment=np.zeros(T, np.int16),
        dates=dates, n_valid_pixels=nval.astype(np.int32),
        note=np.array("run P2 (S4br repaired pre-origin run, 2020-01..2022-12), off-fan offset "
                      "+ height reference, LiCSBAS16-filtered, LOS/U per pixel, median per "
                      "active cell over pixels whose chain from the first epoch is unbroken; "
                      "mm, positive up, relative to the first epoch; one network segment"))
    ins = [RCfg.RUN_DIR / "P2" / "TS_GEOCml2" / RCfg.CUM_FILE,
           RCfg.RUN_DIR / "P2" / "TS_GEOCml2" / "cum.h5",
           RCfg.RUN_DIR / "P2" / "TS_GEOCml2" / "results" / "mask",
           RL.geom_path(run["frame"])]
    code = [HERE / "s5r2_insarP2.py", HERE / "s5r2_const.py",
            HERE.parent / "s4_insar" / "s4b_lib.py", HERE.parent / "s4_insar" / "s4br_lib.py",
            HERE.parent / "s4_insar" / "s4br_config.py",
            HERE.parent / "s4_insar" / "s4b_config.py"]
    prov = {"output": rel_to_repo(out), "sha256": sha256_file(out),
            "inputs": [{"path": rel_to_repo(p), "sha256": sha256_file(p)} for p in ins],
            "code": [{"path": rel_to_repo(p), "sha256": sha256_file(p)} for p in code],
            "network_gaps": run["network_gaps"], "epochs": int(T), "first": str(dates[0]),
            "last": str(dates[-1]), "cells_with_data": int((nval >= 5).sum()),
            "finite_cell_epochs": float(np.isfinite(val).mean()),
            "epochs_flag9": int((tab["flag"] == 9).sum())}
    out.with_suffix(".json").write_text(json.dumps(prov, indent=1))
    print(json.dumps(prov, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
