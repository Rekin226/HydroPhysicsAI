"""WP A0 step 2: copy every download the 2026-10-12 scouts made into its L0 directory with a
SOURCE.json (url, fetch time, bytes, sha256, licence).

URLs and fetch times come from the scouts' own commands (planning-session transcripts,
2026-10-11 16:38-19:43 UTC) and from the files' modification times, which the pilot copy
preserved. Failed downloads (empty files, HTML error pages saved as .pdf) and pdftotext
derivatives are not L0 data and are not copied; they stay in the pilot directories and are
listed in the report. The curl cookie jar of one probe (dl_fhy/cj.txt) is not copied.

Usage: python -m common.a0.l0_sources [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.parse as up
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.paths import L0_NEW, PILOTS, rel_to_repo  # noqa: E402
from common.source import build_source, copy_into_l0, write_source_json  # noqa: E402

TZ = dt.timezone(dt.timedelta(hours=8))          # machine clock: CST (UTC+8)
COPIED_BY = ("WP A0 (2026-10-12) from results/dataprep/_pilots/<scout>/; downloaded by the "
             "planning session's scout agents")
OGDL = "Open Government Data License, version 1.0 (data.gov.tw licence code 1)"
OGDL_UNVERIFIED = ("Open Government Data License, version 1.0 (platform terms; not re-checked "
                   "at copy time)")
WRA_WEB = ("WRA website publication; no open-data licence recorded at fetch; cite the "
           "publisher and do not redistribute")
TPC = "https://service.taipower.com.tw/data/opendata/apply/file"
DGT = "https://data.gov.tw/api/v2/rest/dataset"
WRA_RD = "https://web.wra.gov.tw/landsubsidence/water_new/File/ReportData"
LICSAR = "https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products"
LICSAR_PUB = LICSAR + ".public"
F069 = "69/069A_06694_131402"
MOA = "https://data.moa.gov.tw/Service/OpenData"
GH_RAW = "https://raw.githubusercontent.com/comet-licsar/LiCSBAS/main"
HYB = "https://gweb.wra.gov.tw/HydroInfoCore"
CLIENT_DEFAULT = "curl with a browser user agent (Mozilla/5.0)"


def q(s: str) -> str:
    return up.quote(s)


def F(src: str, dest: str | None = None, url: str | None = None, note: str | None = None):
    return {"src": src, "dest": dest or Path(src).name, "url": url, "note": note}


def tree(scout_dir: str, url_of, exclude: tuple[str, ...] = (), prefix: str = "") -> list[dict]:
    """Every file under a pilot download directory, url from ``url_of(relpath)``."""
    base = PILOTS / scout_dir
    out = []
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.name not in exclude and "__pycache__" not in p.parts:
            rel = p.relative_to(base).as_posix()
            u, note = url_of(rel)
            out.append(F(f"{scout_dir}/{rel}", prefix + rel, u, note))
    return out


# ----------------------------------------------------------------------------- URL maps
def _tpc_head(rel):
    return f"{TPC}/{Path(rel).stem}/001.csv", "HTTP Range bytes=0-1500 probe of the dataset file"


def _datagov(rel):
    return f"{DGT}/{Path(rel).stem.split('_')[1]}", "data.gov.tw dataset metadata (API v2)"


def _hydroinfo(rel):
    js = {"core.js": "index-DeiDUKZH.js"}
    m = {
        "hydroinfo_root.html": ("https://gweb.wra.gov.tw/HydroInfo/", "followed redirects"),
        "stdatainfo.html": ("https://gweb.wra.gov.tw/HydroInfo/StDataInfo/StDataInfo?RA", None),
        "wrhygis.html": ("https://gweb.wra.gov.tw/wrhygis/", "followed redirects"),
        "annual_GET.json": (f"{HYB}/api/Hyebook/GetAnnualReports", "GET probe"),
        "annual_POST.json": (f"{HYB}/api/Hyebook/GetAnnualReports", "POST probe, no body"),
        "annual2.json": (f"{HYB}/api/Hyebook/GetAnnualReports",
                         "POST with JSON body {} and Referer; yearbook listing"),
        "ebook2022.htm": (f"{HYB}/AnnualReports/ebook/ebook/hyb2022/default.htm", None),
        "idx.htm": (f"{HYB}/AnnualReports/ebook/ebook/hyb2022/h-0-1-index.htm", None),
        "h-0-1-bar.htm": (f"{HYB}/AnnualReports/ebook/ebook/hyb2022/h-0-1-bar.htm", None),
        "h-0-1-show.htm": (f"{HYB}/AnnualReports/ebook/ebook/hyb2022/h-0-1-show.htm", None),
        "ui_st.js": ("https://gweb.wra.gov.tw/Hydroinfo/Scripts/UI/ViewUI/UI_StDataInfo.js",
                     "web-page script; never executed"),
    }
    if rel in m:
        return m[rel]
    if rel.startswith("idx20"):
        y = rel[3:7]
        return f"{HYB}/AnnualReports/ebook/ebook/hyb{y}/h-0-1-index.htm", None
    if rel.endswith(".js"):
        return f"{HYB}/assets/{js.get(rel, rel)}", "web-page script; never executed"
    return None, "probe file; url not recorded"


def _fhy(rel):
    top = {"sc.html": "/ReservoirPage_2011/StorageCapacity.aspx",
           "highcharts.js": "/fhyv2/script/Highcharts-12.4/highcharts.js",
           "exporting.min.js": "/fhyv2/script/Highcharts-4.0.1/js/modules/exporting.min.js",
           "svg.rain.min.js": "/fhyv2/js/Reservoir/svg.rain.min.js",
           "svg.water.min.js": "/fhyv2/js/Reservoir/svg.water.min.js",
           "snap.svg-min.js": "/fhyv2/js/Reservoir/snap.svg-min.js",
           "chunk-vendors.4edb8c79.js": "/fhyv2/js/chunk-vendors.4edb8c79.js",
           "app.8496b7f3.js": "/fhyv2/js/app.8496b7f3.js"}
    if rel in top:
        return "https://fhy.wra.gov.tw" + top[rel], ("followed redirects" if rel == "sc.html"
                                                     else "web-page script; never executed")
    if rel.startswith("chunks/"):
        return f"https://fhy.wra.gov.tw/fhyv2/js/{rel[7:]}", "web-page script; never executed"
    if rel == "chunklist.txt":
        return None, "list of chunk names parsed by the scout from app.8496b7f3.js (derived)"
    return None, "probe file; url not recorded"


def _jiji(rel):
    if rel == "annual.html":
        return "https://web.wra.gov.tw/jiji/News.aspx?n=8802&sms=13345", None
    if rel.startswith("list"):
        return f"https://web.wra.gov.tw/jiji/News.aspx?n=8802&sms=13345&page={rel[4]}&PageSize=20", None
    return f"https://web.wra.gov.tw/jiji/cp.aspx?n={rel[2:6]}", None


def _moa_irr(rel):
    if rel == "moa_planned_690.csv":
        return f"{MOA}/Tkucod001.aspx?FOTT=CSV&IsTransData=1&UnitId=690", None
    if rel == "moa_actual_691.csv":
        return f"{MOA}/Tkucod001.aspx?FOTT=CSV&IsTransData=1&UnitId=691", None
    u = Path(rel).stem.split("_")[1]
    return f"{MOA}/DataFileService.aspx?UnitId={u}&FOTT=CSV&IsTransData=1", None


def _moa038(rel):
    if rel == "p0.json":
        return f"{MOA}/DataFileService.aspx?UnitId=038&IsTransData=1&$top=5&$skip=0", "probe"
    if rel == "t.json":
        return f"{MOA}/DataFileService.aspx?UnitId=038&IsTransData=1&year=110", (
            "probe; the scout's loop overwrote t.json, this is the last query")
    if rel == "chg110_p0.json":
        return (f"{MOA}/FromM/TownCropData.aspx?$top=5000&$skip=0&year=110&city={q('彰化縣')}"
                "&UnitId=038"), "first page probe"
    _, city, year = Path(rel).stem.split("_")
    return (f"{MOA}/FromM/TownCropData.aspx?$top=5000&$skip=<k*5000>&year={year}&city={q(city)}"
            "&UnitId=038"), "pages concatenated into one JSON list by the scout's fetch_moa.py"


def _moaJ51(rel):
    if rel == "t.json":
        return f"{MOA}/FromM/TownRiceData.aspx?$top=50&$skip=0&year=110&city={q('彰化縣')}", "probe"
    year = Path(rel).stem.split("_")[1]
    return (f"{MOA}/FromM/TownRiceData.aspx?$top=1000&$skip=<k*1000>&year={year}",
            "pages concatenated by the scout's fetch_rice.py and FILTERED client-side to "
            "縣市 in {彰化縣, 雲林縣}: not the raw response bytes")


def _licsar_root(rel):
    n = Path(rel).stem
    if n == "root_listing":
        return f"{LICSAR}/", "directory listing"
    if n.startswith("track_"):
        return f"{LICSAR}/{n[6:]}/", "directory listing"
    if n == "f69a":
        return f"{LICSAR}/69/069A_05428_181818/", "directory listing"
    if n == "f69a_meta":
        return f"{LICSAR}/69/069A_05428_181818/metadata/", "directory listing"
    return None, "listing; url not recorded"


def _licsar_frames(rel):
    fixed = {"ifg_sample.html": f"{LICSAR}/{F069}/interferograms/20221215_20221227/",
             "ifg_sample2": f"{LICSAR}/{F069}/interferograms/20221215_20221227",
             "ep_sample.html": f"{LICSAR}/{F069}/epochs/20221227/",
             "ep2": f"{LICSAR}/{F069}/epochs/20221227",
             "vel.html": f"{LICSAR}/velocities/", "velahb.html": f"{LICSAR}/velocities/ahb/",
             "ahb.html": f"{LICSAR}/AHB_licsbas/", "ahb2.html": f"{LICSAR}/AHB/"}
    if rel in fixed:
        return fixed[rel], "directory listing"
    fr, kind = rel.rsplit("_", 1)
    t = str(int(fr[:3]))
    sub = {"ifg.html": "interferograms/", "ep.html": "epochs/", "root.html": ""}[kind]
    return f"{LICSAR}/{t}/{fr}/{sub}", "directory listing"


def _licsar_poly(rel):
    fr = rel.replace("-poly.txt", "")
    return f"{LICSAR}/{int(fr[:3])}/{fr}/metadata/{fr}-poly.txt", "frame polygon"


def _licsar_subsets(rel):
    box = "lon 120.28-120.50, lat 23.58-23.78"
    if rel == "meta/grid.json":
        return None, f"grid record written by licsar_subset.py for the metadata subsets ({box})"
    if rel == "unw/grid.json":
        return None, "grid record of the first interferogram subset (licsar_subset.py)"
    name = Path(rel).stem
    if rel.startswith("meta/"):
        u = f"{LICSAR_PUB}/{F069}/metadata/{name}.tif"
    else:
        pair = name.split(".")[0]
        u = f"{LICSAR_PUB}/{F069}/interferograms/{pair}/{name}.tif"
    if rel.endswith(".json"):
        return u, ("grid record of the subset (some were copies of unw/grid.json made by the "
                   "scout after a patch; see the scout transcript)")
    return u, (f"HTTP range reads of the tiles covering {box}, decoded to float32 by the "
               "scout's licsar_subset.py: not byte-identical to any server file")


def _licsbas(rel):
    if rel == "sb_inv.py":
        return f"{GH_RAW}/bin/LiCSBAS13_sb_inv.py", "script; never executed"
    return f"{GH_RAW}/{rel}", None


def _cy_url(folder: str, name: str) -> str:
    return f"{WRA_RD}/{folder}/{q(name)}"


# ----------------------------------------------------------------------------- the table
def sources() -> list[dict]:
    tpc = "tpc_aggregate_electricity"
    sw = "surface_water_monthly"
    lu = "landuse_crop_priors"
    sub = "subsidence_gap_sources"
    rv = "review_cf"
    hyb = []
    for line in (PILOTS / sw / "hyb_urls.tsv").read_text(encoding="utf-8").splitlines():
        y, _, u = line.split("\t")
        hyb.append(F(f"{sw}/dl_hyb{y}/hyb{y}_stage_discharge.pdf",
                     f"hyb{y}/hyb{y}_stage_discharge.pdf", u,
                     "Part 2 (river discharge and stage) of the WRA hydrological yearbook"))
    S = []

    def add(source_id, title, publisher, licence, scout, files, subdir="", **kw):
        S.append({"source_id": source_id, "dir": source_id + (f"/{subdir}" if subdir else ""),
                  "title": title, "publisher": publisher, "licence": licence, "scout": scout,
                  "files": files, **kw})

    D = "2026-10-12"
    add("tpc_d007012", "台灣電力公司各縣市住宅、服務業及機關用電統計資料 (county x sector monthly "
        "sales; data.gov.tw 29935)", "台灣電力公司", OGDL, tpc,
        [F(f"{tpc}/dl_d007012/d007012.csv", url=f"{TPC}/d007012/001.csv")], D,
        rolling=True, datagovtw_id=29935)
    add("tpc_d007019", "台灣電力公司各縣市年度住宅、服務業、機關及工業用電比例 (data.gov.tw 38959)",
        "台灣電力公司", OGDL, tpc,
        [F(f"{tpc}/dl_d007019/d007019.csv", url=f"{TPC}/d007019/001.csv")], D,
        rolling=True, datagovtw_id=38959)
    add("tpc_d007021", "TPC d007021: monthly electricity by first-level dissemination area "
        "(一級發布區; title inferred from the CSV header, dataset metadata not fetched)",
        "台灣電力公司", "TPC open data (data.gov.tw); licence not recorded for this dataset",
        tpc, [F(f"{tpc}/dl_d007021/d007021.csv", url=f"{TPC}/d007021/001.csv")], D,
        rolling=True)
    add("tpc_d007023", "台灣電力公司年度電燈營業用戶(依行業別)售電量 (data.gov.tw 122521)",
        "台灣電力公司", OGDL, tpc,
        [F(f"{tpc}/dl_d007023/d007023.csv", url=f"{TPC}/d007023/001.csv")], D,
        rolling=True, datagovtw_id=122521)
    add("tpc_d003004", "台灣電力公司公用售電業之電力營業用戶(依行業別)售電量 (data.gov.tw 31966)",
        "台灣電力公司", OGDL, tpc,
        [F(f"{tpc}/dl_d003004/001.zip", url=f"{TPC}/d003004/001.zip",
           note="zip archive; member names are Big5 encoded")], D,
        rolling=True, datagovtw_id=31966)
    add("tpc_d003005", "台灣電力公司售電量(用途別) (data.gov.tw 35392, resource 002)",
        "台灣電力公司", OGDL, tpc,
        [F(f"{tpc}/dl_d003005/002.csv", url=f"{TPC}/d003005/002.csv")], D,
        rolling=True, datagovtw_id=35392)
    add("tpc_dataset_probes", "First 1,501 bytes of 22 TPC d0070xx open-data files (header "
        "probes used to choose datasets)", "台灣電力公司", OGDL, tpc,
        tree(f"{tpc}/dl_tpc_heads", _tpc_head), D)
    add("datagovtw_metadata", "data.gov.tw dataset metadata records of the datasets the TPC "
        "scout inspected (contain publisher contact fields: never copy them into products)",
        "國家發展委員會 data.gov.tw", OGDL, tpc, tree(f"{tpc}/dl_meta", _datagov), D)
    add("moeaea_agri_electricity", "經濟部能源署 電力消費月資料 (data.gov.tw 161209; set 236 = 2018 "
        "on, set 256 = 2017 and earlier)", "經濟部能源署", OGDL, tpc,
        [F(f"{tpc}/dl_moeaea_236/f.csv", "set_236.csv",
           "https://www.moeaea.gov.tw/ECW/populace/opendata/wHandOpenData_File.ashx?set_id=236"),
         F(f"{tpc}/dl_moeaea_256/f.csv", "set_256.csv",
           "https://www.moeaea.gov.tw/ECW/populace/opendata/wHandOpenData_File.ashx?set_id=256")],
        D, rolling=True, datagovtw_id=161209)
    wra_ds = {"58698": ("55bd8e78-5e8b-47a5-9ea3-d7403d06a131", "agriculture"),
              "58699": ("f7fa35a0-5d47-48c2-a755-111538507110", "livestock"),
              "58700": ("dfc12113-9e62-4185-a3e6-4a83042d0c71", "aquaculture")}
    add("wra_water_use_stats", "水利署 用水統計 (data.gov.tw 58698 農業, 58699 畜牧, 58700 養殖)",
        "經濟部水利署", OGDL, tpc,
        [F(f"{tpc}/dl_wra_{k}/f.csv", f"ds{k}_{v[1]}.csv",
           f"https://opendata.wra.gov.tw/api/v2/{v[0]}?sort=_importdate%20asc&format=CSV")
         for k, v in wra_ds.items()], D, rolling=True)
    add("wra_reports_misc", "WRA and government reports used by the scouts: MOEA 2019 Yunlin-"
        "Changhua subsidence plan annex, 111 drought-well report, Control Yuan 53001, WRA "
        "Changhua-Yunlin subsidence survey results 108-112 (the 110/112 briefings), WRA "
        "subsidence monthly report 2022-01", "經濟部水利署; 監察院 (CY 53001)", WRA_WEB,
        f"{tpc}, {sw}",
        [F(f"{tpc}/dl_wra_plan/plan.pdf", "moea2019_yunchang_plan_annex.pdf",
           _cy_url("007006", "雲彰地區地層下陷具體解決方案暨行動計畫(第二次修正)-附冊.pdf")),
         F(f"{tpc}/dl_drought_wells/r.pdf", "wra111_drought_wells_report.pdf",
           f"{WRA_RD}/007008/b6d4621a-1035-460d-a6b6-7b7abf567623.pdf"),
         F(f"{tpc}/dl_cy53001/d", "cy53001.pdf",
           "https://cybsbox.cy.gov.tw/CYBSBoxSSL/edoc/download/53001",
           "Control Yuan document 53001 (also saved by the billing scout as dl_cy53001/doc.pdf,"
           " identical sha256)"),
         F(f"{sw}/dl_cy_monthly/cy_annual_2019.pdf", "wra_cy_subsidence_results_108.pdf",
           _cy_url("007007", "108年度_108年度彰雲地區地層下陷檢測成果.pdf")),
         F(f"{sw}/dl_cy_monthly/cy_annual_2020.pdf", "wra_cy_subsidence_results_109.pdf",
           _cy_url("007007", "109年度_109年度彰雲地區地層下陷檢測成果.pdf")),
         F(f"{sw}/dl_cy_monthly/cy_annual_2021.pdf", "wra_cy_subsidence_results_110.pdf",
           _cy_url("007007", "110年度_110年度彰雲地區地層下陷檢測成果.pdf"), "the WRA 110 briefing"),
         F(f"{sw}/dl_cy_2022_2023/cy_2022.pdf", "wra_cy_subsidence_results_111.pdf",
           _cy_url("007007", "111年度_111年度彰雲地區地層下陷檢測結果.pdf")),
         F(f"{sw}/dl_cy_2022_2023/cy_2023.pdf", "wra_cy_subsidence_results_112.pdf",
           _cy_url("007007", "112年度_112年度彰雲地區地層下陷檢測結果.pdf"), "the WRA 112 briefing"),
         F(f"{sw}/dl_cy_monthly/cy_monthly_202201.pdf", "wra_subsidence_monthly_2022-01.pdf",
           _cy_url("007003", "111年1月 地層下陷監測月報.pdf"))])
    add("wra_hydro_yearbook", "WRA hydrological yearbooks 2012-2025, Part 2 (river discharge and "
        "stage)", "經濟部水利署", "Yearbook PDFs disallow copying (plan 2.6): raw files and daily "
        "extractions stay gitignored; only scripts and derived monthly aggregates with "
        "attribution could ever be committed", sw, hyb)
    add("wra_hydroinfo_portal", "Probes of the WRA HydroInfo portal (yearbook listing API, "
        "e-book index pages, page scripts)", "經濟部水利署", WRA_WEB, sw,
        tree(f"{sw}/dl_hydroinfo", _hydroinfo))
    add("fhy_reservoir_portal", "Probes of the WRA fhy reservoir page (HTML and page scripts)",
        "經濟部水利署", WRA_WEB, sw, tree(f"{sw}/dl_fhy", _fhy, exclude=("cj.txt",)),
        excluded=["cj.txt (curl cookie jar of an anonymous session; not data)"])
    add("jiji_weir_pages", "集集攔河堰 (WRA Jiji weir office) annual-report listing and content pages",
        "經濟部水利署中區水資源分署", WRA_WEB, sw, tree(f"{sw}/dl_jiji", _jiji))
    add("wra_library_pages", "WRA library catalogue pages (books_<id>)", "經濟部水利署", WRA_WEB, sw,
        tree(f"{sw}/dl_wralib",
             lambda r: (f"https://web.wra.gov.tw/WraLib/wSite/books_{Path(r).stem[1:]}", None)))
    add("moa_irrigation_stats", "MOA open data: irrigation statistics (UnitId 688-693; planned "
        "690 and actual 691 via Tkucod001)", "農業部", OGDL_UNVERIFIED, sw,
        tree(f"{sw}/dl_moa", _moa_irr))
    add("wra_reservoir_daily", "WRA open data: daily reservoir operations (dataset "
        "51023e88-4c76-4dbc-bbb9-470da690d539)", "經濟部水利署", OGDL_UNVERIFIED, sw,
        [F(f"{sw}/dl_wraopen/daily.json", url="https://opendata.wra.gov.tw/api/v2/51023e88-"
           "4c76-4dbc-bbb9-470da690d539?sort=_importdate%20asc&format=JSON")], D, rolling=True)
    add("moa_crop_038", "MOA township crop survey (UnitId 038, TownCropData), Changhua and Yunlin, "
        "ROC years 103, 104, 105, 110, 112", "農業部", OGDL_UNVERIFIED, lu,
        tree(f"{lu}/dl_moa038", _moa038),
        client="curl (no user agent) for the probes; python urllib (the scout's fetch_moa.py) "
               "for crop_*.json")
    add("moa_rice_J51", "MOA township rice survey (TownRiceData), ROC years 103, 104, 105, 110, "
        "112, filtered to Changhua and Yunlin", "農業部", OGDL_UNVERIFIED, lu,
        tree(f"{lu}/dl_moaJ51", _moaJ51),
        client="curl (no user agent) for the probe; python urllib (the scout's fetch_rice.py) "
               "for rice_*.json")
    add("nlsc_landuse_legend", "NLSC 109年版國土利用色碼表 (land-use colour legend image)",
        "內政部國土測繪中心", "NLSC web content; licence not recorded", lu,
        [F(f"{lu}/dl_legend/legend.png", url="https://maps.nlsc.gov.tw/demo/"
           + q("109年版國土利用色碼表.png"))], client="curl (no user agent)")
    add("gadm41_twn", "GADM 4.1 administrative boundaries, Taiwan (GeoPackage)", "GADM",
        "GADM licence: free for academic and other non-commercial use; redistribution or "
        "commercial use not allowed without permission", lu,
        [F(f"{lu}/dl_gadm/gadm41_TWN.gpkg",
           url="https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_TWN.gpkg")],
        client="curl (no user agent)")
    add("taiwan_atlas", "taiwan-atlas npm package, towns-10t TopoJSON (versions 1.0.0 and "
        "2021.9.20) via jsDelivr", "taiwan-atlas (npm), from MOI boundaries",
        "npm package licence not verified at copy time; underlying MOI boundaries are open data",
        lu, [F(f"{lu}/dl_atlas/towns-10t.json",
               url="https://cdn.jsdelivr.net/npm/taiwan-atlas@1.0.0/towns-10t.json"),
             F(f"{lu}/dl_atlas/towns-10t_2021.json",
               url="https://cdn.jsdelivr.net/npm/taiwan-atlas@2021.9.20/towns-10t.json")],
        client="curl (no user agent)")
    add("licsar_069A_06694", "LiCSAR frame 069A_06694_131402: interferogram unwrapped-phase "
        "subsets 2021-10..2023-03 (range reads, decoded), E/N/U metadata subsets, one full "
        "interferogram and coherence GeoTIFF", "COMET LiCSAR (JASMIN)",
        "Copernicus Sentinel-1 derived-work terms; cite LiCSAR (Lazecky et al. 2020, Remote "
        "Sensing 12(15) 2430)", sub,
        tree(f"{sub}/dl_insar", _licsar_subsets, prefix="subsets/")
        + [F(f"{sub}/dl_tifhead/unw_full.tif", "full/20221215_20221227.geo.unw.tif",
             f"{LICSAR_PUB}/{F069}/interferograms/20221215_20221227/20221215_20221227.geo.unw.tif"),
           F(f"{sub}/dl_tifhead/cc_full.tif", "full/20221215_20221227.geo.cc.tif",
             f"{LICSAR_PUB}/{F069}/interferograms/20221215_20221227/20221215_20221227.geo.cc.tif"),
           F(f"{sub}/dl_tifhead/unw_head.bin", "full/20221215_20221227.geo.unw.head131072.bin",
             f"{LICSAR_PUB}/{F069}/interferograms/20221215_20221227/20221215_20221227.geo.unw.tif",
             "HTTP Range bytes=0-131071 (TIFF header probe)")],
        bytes_transferred_note="licsar_subset.py reported 107.6 MB downloaded by range reads "
                               "for the 171 interferogram subsets (dl_insar_unw2.log)",
        url_lists=["_pilots/subsidence_gap_sources/scripts/urls_unw.txt",
                   "_pilots/subsidence_gap_sources/scripts/urls_meta.txt"],
        subset_code="_pilots/subsidence_gap_sources/scripts/licsar_subset.py",
        client="python requests range reads (licsar_subset.py) for subsets/; curl (no user "
               "agent) for full/")
    add("licsar_catalogue", "LiCSAR directory listings (root, tracks, frames, epochs, "
        "velocities) and frame polygons for tracks 3, 32, 69, 76, 105, 134, 156, 171",
        "COMET LiCSAR (JASMIN)", "Copernicus Sentinel-1 derived-work terms; cite LiCSAR", sub,
        tree(f"{sub}/dl_root", _licsar_root, prefix="root/")
        + tree(f"{sub}/dl_frames", _licsar_frames, prefix="frames/")
        + tree(f"{sub}/dl_polys", _licsar_poly, prefix="polys/"), D,
        client="curl (no user agent)")
    add("licsbas_repo_files", "LiCSBAS repository files (requirements, environment, README, "
        "LiCSBAS13_sb_inv.py, LiCSBAS03op_ERA5.py, GitHub bin/ listing)", "COMET (GitHub)",
        "LiCSBAS is GPL-3.0; scripts are untrusted downloads and are never executed", f"{sub}, {rv}",
        tree(f"{sub}/dl_licsbas", _licsbas)
        + [F(f"{rv}/dl_licsbas_era5/era5.py", "LiCSBAS03op_ERA5.py",
             f"{GH_RAW}/bin/LiCSBAS03op_ERA5.py", "script; never executed"),
           F(f"{rv}/dl_licsbas_era5/bin.json", "github_bin_listing.json",
             "https://api.github.com/repos/comet-licsar/LiCSBAS/contents/bin")],
        client="curl (no user agent)")
    add("ngl_gnss", "Nevada Geodetic Laboratory: DataHoldings.txt station table and HUWE IGS20 "
        "tenv3 daily series (review check)", "Nevada Geodetic Laboratory (UNR)",
        "NGL products are openly available; cite Blewitt, Hammond and Kreemer (2018), Eos 99",
        rv, [F(f"{rv}/dl_ngl/DataHoldings.txt",
               url="https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt"),
             F(f"{rv}/dl_ngl_huwe/HUWE.tenv3",
               url="https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/HUWE.tenv3")],
        D, rolling=True, client="curl (no user agent)")
    add("pressure_hourly", "Open-Meteo archive API test response: hourly surface_pressure at "
        "23.7N 120.4E, 2012-01-01..02 (review check)", "Open-Meteo",
        "CC BY 4.0 (Open-Meteo terms; ERA5-based archive)", rv,
        [F(f"{rv}/dl_openmeteo/om.json", "review_test_om_surface_pressure.json",
           "https://archive-api.open-meteo.com/v1/archive?latitude=23.7&longitude=120.4&"
           "start_date=2012-01-01&end_date=2012-01-02&hourly=surface_pressure&"
           "timezone=Asia%2FTaipei")], "review_test_2026-10-12", client="curl (no user agent)")
    add("openmeteo_et0", "Open-Meteo archive API test response: daily et0_fao_evapotranspiration "
        "at 23.7N 120.4E, 2026-09-25..10-05 (review check)", "Open-Meteo",
        "CC BY 4.0 (Open-Meteo terms; ERA5-based archive)", rv,
        [F(f"{rv}/dl_openmeteo/et0.json", "review_test_et0.json",
           "https://archive-api.open-meteo.com/v1/archive?latitude=23.7&longitude=120.4&"
           "start_date=2026-09-25&end_date=2026-10-05&daily=et0_fao_evapotranspiration&"
           "timezone=Asia%2FTaipei")], "review_test_2026-10-12", client="curl (no user agent)")
    return S


#: Pilot files that look like downloads but are not L0 data (not copied; reported).
NOT_COPIED = {
    "tpc_aggregate_electricity/dl_pdf3/file.pdf": "empty file (failed download of a TPC PDF)",
    "tpc_aggregate_electricity/dl_cy53001/d.txt": "pdftotext derivative",
    "tpc_aggregate_electricity/dl_drought_wells/r.txt": "pdftotext derivative",
    "tpc_aggregate_electricity/dl_wra_plan/plan.txt": "pdftotext derivative",
    "billing_and_census_gaps/dl1/f.pdf": "HTML error page saved as .pdf (failed TPC PDF)",
    "billing_and_census_gaps/dl_cy53001/doc.pdf": "duplicate of tpc dl_cy53001/d (same sha256)",
    "billing_and_census_gaps/dl_cy53001/doc.txt": "pdftotext derivative",
    "surface_water_monthly/dl_hyb2022/hyb2022.txt": "pdftotext derivative",
    "surface_water_monthly/dl_fhy/cj.txt": "curl cookie jar (anonymous session), not data",
}
EMPTY_DL_DIRS = ["tpc_aggregate_electricity/dl_chcg", "tpc_aggregate_electricity/dl_pdf1",
                 "tpc_aggregate_electricity/dl_pdf2", "billing_and_census_gaps/dl2",
                 "billing_and_census_gaps/dl3", "landuse_crop_priors/dl_chcg_aqua",
                 "subsidence_gap_sources/dl_meta_probe", "surface_water_monthly/dl_chcg"]


def fetched_utc(p: Path) -> str:
    return dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def run(dry: bool = False) -> dict:
    done, l0_map = [], {}
    for s in sources():
        d = L0_NEW / s["dir"]
        files = []
        times = []
        for f in s["files"]:
            src = PILOTS / f["src"]
            if not src.is_file():
                raise FileNotFoundError(src)
            if not dry:
                copy_into_l0(src, d / f["dest"])
            t = fetched_utc(src)
            times.append(t)
            rec = {"file": f["dest"], "url": f["url"], "fetched_utc": t,
                   "pilot_copy": "results/dataprep/_pilots/" + f["src"]}
            if f["note"]:
                rec["note"] = f["note"]
            files.append(rec)
            l0_map[f["src"]] = rel_to_repo(d / f["dest"])
        first = min(times)
        local = dt.datetime.strptime(first, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=dt.timezone.utc).astimezone(TZ)
        extra = {k: v for k, v in s.items()
                 if k not in ("source_id", "dir", "title", "publisher", "licence", "scout",
                              "files", "client")}
        if dry:
            done.append({"source_id": s["source_id"], "dir": s["dir"], "n": len(files)})
            continue
        rec = build_source(d, source_id=s["source_id"], title=s["title"],
                           publisher=s["publisher"], licence=s["licence"],
                           fetched=local.date().isoformat(), files=files,
                           fetched_by=f"scout(s) {s['scout']} of the 2026-10-12 planning session",
                           copied_by=COPIED_BY,
                           provenance=("url and request details reconstructed from the scout's "
                                       "command in the planning-session transcript; fetched_utc "
                                       "is the file's modification time, preserved through the "
                                       "pilot copy"),
                           client=s.get("client", CLIENT_DEFAULT),
                           untrusted="downloaded data: never execute; read with python -I",
                           **extra)
        p = write_source_json(d, rec)
        done.append({"source_id": s["source_id"], "dir": rel_to_repo(d), "n": len(files),
                     "bytes": sum(f["bytes"] for f in rec["files"]),
                     "source_json": rel_to_repo(p)})
    return {"sources": done, "l0_map": l0_map, "not_copied": NOT_COPIED,
            "empty_download_dirs": EMPTY_DL_DIRS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", help="write the migration summary JSON here")
    a = ap.parse_args()
    res = run(a.dry_run)
    for s in res["sources"]:
        print(f"{s['n']:4d} files {s.get('bytes', 0) / 1e6:9.2f} MB  {s['dir']}")
    print(f"{len(res['sources'])} sources, {sum(s['n'] for s in res['sources'])} files, "
          f"{sum(s.get('bytes', 0) for s in res['sources']) / 1e6:.1f} MB")
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n",
                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
