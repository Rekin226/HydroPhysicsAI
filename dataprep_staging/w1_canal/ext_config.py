"""Fixed settings of the W1 2026 extension (plan W1 step 6), written before any score.

The yearbooks (inflow composite) end in 2025 and the Jiji annual totals too, so 2026
deliveries need a rainfall-runoff substitute for the composite inflow. The design below was
fixed on 2026-10-12 before the CODIS mountain-station rain was fetched or any regression
was scored; the pre-registration (ext_prereg.py) hashes this file.

Station rule (a priori, from the CWA station list alone): CWA stations (manned, automatic
rain or agricultural) inside the drainage areas of the three composite gauges, i.e. the
Choushui mainstem above 寶石橋 H075 (1,542.39 km2), the 陳有蘭溪 above 內茅埔 H049
(367.40 km2) and the 水里溪 at 水里 H064 (80.22 km2) with the Sun Moon Lake (日月潭)
reservoir that drains to it; the record (one ID, or a chain of renumbered IDs at the same
site) must span 2011-2026. Excluded: stations on the Wu River (烏溪) side of 仁愛鄉 (瑞岩,
清流, 阿眉, 翠峰, 翠巒), divide/peak stations (合歡山, 昆陽, 玉山, 新高口, 玉山風口),
stations below the gauges (西巒 on the 陳有蘭溪 below 內茅埔, 集集, 竹山, 鹿谷) and IDs
that closed in 2015 without a successor at the same site (文文社, 望鄉, 東埔, 上安橋, 龍神橋).
Drainage areas are from the 2022 yearbook station table.
"""
from __future__ import annotations

from w1_canal.config import L0_NEW

# ----------------------------------------------------------------------------- L0 fetch
L0_SOURCE_ID = "cwa_codis_rain_upper_choushui"
L0_DIR = L0_NEW / L0_SOURCE_ID
RAW_DIR = L0_DIR / "raw"
FETCH_START = "2011-01-01"        # lags of Jan 2012 need Nov-Dec 2011
FETCH_END = "2026-10-11"          # yesterday (Taipei) on the fetch day

# sub-basin -> (gauge, drainage area km2)
SUBBASINS = {"mainstem": ("H075", 1542.39), "chenyoulan": ("H049", 367.40),
             "shuili": ("H064", 80.22)}

# record id -> (sub-basin, name, chain [(station_id, stn_type, first day, last day|None)])
STATIONS = {
    "lushan": ("mainstem", "廬山", [("C0I010", "auto_C0", "1992-05-12", None)]),
    "wanda": ("mainstem", "萬大", [("C1I020", "auto_C0", "1992-05-12", None)]),
    "wujie": ("mainstem", "武界", [("C1I030", "auto_C0", "1992-05-03", None)]),
    "aowanda": ("mainstem", "奧萬大", [("C1I320", "auto_C0", "2002-05-15", None)]),
    "fengshulin": ("mainstem", "楓樹林", [("C1I330", "auto_C0", "2002-05-01", None)]),
    "danda": ("mainstem", "丹大", [("C1I050", "auto_C0", "1992-05-01", None)]),
    "kanaituowan": ("mainstem", "卡奈托灣", [("C1I140", "auto_C0", "1993-01-21", None)]),
    "qingyun": ("mainstem", "青雲", [("C1I150", "auto_C0", "1992-12-01", None)]),
    "renai": ("mainstem", "仁愛", [("C1H870", "auto_C0", "1991-12-12", "2015-08-10"),
                                   ("C0I390", "auto_C0", "2015-08-10", None)]),
    "heshe": ("chenyoulan", "和社", [("C1I070", "auto_C0", "1992-05-01", None)]),
    "xinxingqiao": ("chenyoulan", "新興橋", [("C1I340", "auto_C0", "2002-03-09", None)]),
    "shenmucun": ("chenyoulan", "神木村", [("C0H9A0", "auto_C0", "1998-12-16", None)]),
    "fengqiu": ("chenyoulan", "豐丘", [("C1I290", "auto_C0", "2002-03-05", None)]),
    "xinyi": ("chenyoulan", "信義", [("C1I080", "auto_C0", "1992-05-01", "2015-09-22"),
                                     ("C0I080", "auto_C0", "2015-09-22", None)]),
    "shuili": ("shuili", "水里", [("C1I270", "auto_C0", "2002-03-01", "2015-09-23"),
                                  ("C0I360", "auto_C0", "2015-09-23", None)]),
    "sunmoonlake": ("shuili", "日月潭", [("467650", "cwb", "1940-01-01", None)]),
}

# ----------------------------------------------------------------------------- rain index
MIN_DAY_FRACTION = 0.8            # a station-month counts with >= 80 % numeric days
#: station normal: mean over the 12 calendar months of the mean valid total of that calendar
#: month in this window (robust to seasonally uneven gaps)
NORMAL_WINDOW = ("2012-01", "2022-12")
MIN_STATIONS_MONTH = 8            # a month of the index needs >= 8 of the 16 records
#: CODIS availability (checked before any score): mid-2014 to 2015 only 2-7 of the 16
#: records are valid (station renumbering), so those months have no index; the regression
#: drops rows without an index and the chain test scores only the evaluation years: fit
#: years whose months from November of the previous year to December all have an index
#: (2012, 2013, 2017-2025 with the fetched data).

# ----------------------------------------------------------------------------- regression
#: target: monthly volume (10^8 m3) of the full variant's composite I (rescaled to the weir
#: inflow); regressors: basin rain index P (mm/month) of months m, m-1, m-2 and calendar-month
#: intercepts. Two specifications; for every prediction the one with the lower
#: leave-one-year-out monthly RMSE on the training years is used (nested selection).
SPECS = ("linear_lag2", "log_lag2")
LAGS = 2
FIT_YEARS = tuple(range(2012, 2026))   # the full variant's years (U4: recorded arm only)

# ----------------------------------------------------------------------------- chain test
#: Leave-one-year-out over FIT_YEARS: the regression (with nested spec selection) and
#: (beta, a_chc, a_yl) are refitted without the held-out year; the shares and winter weight
#: are the full variant's final choice; R (industrial + public) of the held-out year is the
#: previous year's (as for 2026, whose R is unknown); the held-out year's daily inflow is the
#: predicted monthly volume spread evenly over the month's days.
BOOT_B = 10000
BOOT_SEED = 20261013
BOOT_LEVEL = 0.90
N_DRAWS = 200
DRAW_SEED = 2026
EXT_YEAR = 2026
R_EXT_FROM_YEAR = 2025            # R of 2026 = R of 2025 (Jiji table)
