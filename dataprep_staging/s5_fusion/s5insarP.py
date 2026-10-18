"""WP S5: twin-cell InSAR values of the pre-origin LiCSBAS run P (S4b's gate run).

insar_069A/v2 carries run F (2020-01..2025-12), whose inversion used 2023+ interferograms. U7
forbids 2023+ observations in a hyperparameter or a gate, so the S5 pre-origin runs use run P
(pre-origin pairs only), processed exactly as S4b processed run F for the product: S4b's
stable-area reference (offset + east/north plane + height per epoch), vertical-equivalent
LOS / U, median of the valid multilooked pixels per active cell (>= 5 pixels). Cell segment ids
come from S4b's ``gaps_P.npz``. No GNSS, leveling, ring or pile value is used.

    tools/cpu_slot.sh python -I dataprep_staging/s5_fusion/s5insarP.py

Writes results/dataprep/reports/S5_evidence/work/insar_runP_cells.npz (+ .json provenance).
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
import s4b_evaluate as Ev  # noqa: E402
import s4b_lib as Lb  # noqa: E402
import s5const as C  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402


def main() -> int:
    C.WORK.mkdir(parents=True, exist_ok=True)
    out = C.WORK / "insar_runP_cells.npz"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    A = Lb.n_active()
    run, g, v, tab = Ev.processed("P")
    val, se, nval = Lb.to_cells(v, run["mask"], g, A)          # (T, A)
    gp = np.load(C.INSAR_GAPS_P, allow_pickle=False)
    dates = np.array([d.strftime("%Y%m%d") for d in run["dates"]])
    if not np.array_equal(dates, gp["dates"]):
        raise SystemExit("run-P dates differ from gaps_P.npz")
    eflag = tab["flag"].to_numpy()[:, None]
    flag = np.where(np.isfinite(val), np.where(eflag == 9, 9, 0), 8).astype(np.uint8)
    np.savez_compressed(
        out, value=val.T.astype(np.float32), flag=flag.T, segment=gp["seg_cell"],
        network_segment=gp["seg_net"].astype(np.int16),
        dates=np.array([d.strftime("%Y-%m-%d") for d in run["dates"]]),
        n_valid_pixels=nval.astype(np.int32),
        note=np.array("run P (pre-origin pairs, 2020-01-12..2022-12-27), S4b reference, LOS/U "
                      "per pixel, median per active cell; mm, positive up, relative to the "
                      "first epoch; segment ids from S4b gaps_P.npz"))
    ins = [C.INSAR_CUM_P, C.INSAR_GAPS_P, C.INSAR_RUN_DIR / "geom_069A_06694_131402.npz",
           C.INSAR_RUN_DIR / "P" / "TS_GEOCml2" / "results" / "mask", C.TWIN_GRID]
    code = [HERE / "s5insarP.py", HERE.parent / "s4_insar" / "s4b_lib.py",
            HERE.parent / "s4_insar" / "s4b_evaluate.py", HERE.parent / "s4_insar" /
            "s4b_config.py"]
    prov = {"output": rel_to_repo(out), "sha256": sha256_file(out),
            "inputs": [{"path": rel_to_repo(p), "sha256": sha256_file(p)} for p in ins],
            "code": [{"path": rel_to_repo(p), "sha256": sha256_file(p)} for p in code],
            "epochs": int(len(dates)), "first": str(dates[0]), "last": str(dates[-1]),
            "cells_with_data": int((nval >= 5).sum())}
    (C.WORK / "insar_runP_cells.json").write_text(json.dumps(prov, indent=1))
    print(json.dumps(prov, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
