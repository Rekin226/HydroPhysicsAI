"use strict";
// The analyst shell. Every answer comes from the server (page strings + page numbers); this file
// only lays it out, offers one-click edits of the interpretation, and drives the page iframe
// through its URL hash. "Page agrees" re-reads the page's own headline after the hash is set.
const $ = id => document.getElementById(id);
const UI = {
  en: {title: "Twin analyst", send: "Ask", ph: "Ask about a policy, e.g. retire aquaculture from 2030 — head by 2032?",
       note: "Answers are the page's own numbers and sentences, with claim IDs. Edit the interpretation to correct it.",
       lang: "中文", agree: "page agrees", disagree: "page differs", open: "Open in page", wait: "…",
       err: "Could not get an answer:", slow: "The page is still loading. If it never appears, open it directly:",
       examples: "Try a question", year: "Year", value: "Value",
       metaerr: "Cannot reach the analyst server. Is it still running?"},
  zh: {title: "雙生模型分析助理", send: "提問", ph: "例如：2030年起停止養殖抽水，2032年水位？",
       note: "答案為頁面本身的數值與文字，並附主張編號。可修改解讀以更正。",
       lang: "English", agree: "與頁面一致", disagree: "與頁面不同", open: "在頁面中開啟", wait: "…",
       err: "無法取得答案：", slow: "頁面仍在載入。若一直沒有出現，請直接開啟：",
       examples: "試試這些問題", year: "年份", value: "數值",
       metaerr: "無法連線到分析伺服器，請確認它仍在執行。"}};
let META = null, LANG = (navigator.language || "en").startsWith("zh") ? "zh" : "en", NONCE = 0;
const SESSION = Math.random().toString(36).slice(2);
const t = k => UI[LANG][k];
const S = k => (META.strings[LANG] || META.strings.en)[k] || k;

async function post(url, body) {
  const ctl = new AbortController(), to = setTimeout(() => ctl.abort(), 90000);
  try {
    const r = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body), signal: ctl.signal});
    const v = await r.json();
    if (!r.ok || v.error) throw new Error(v.error || ("HTTP " + r.status));
    return v;
  } finally { clearTimeout(to); }
}
function fail(li, e) {                     // never leave an answer spinning: say what went wrong
  const body = li.querySelector(".body"); body.textContent = "";
  body.appendChild(el("div", "blk refusal", t("err") + " " + (e && e.message ? e.message : e)));
}
function watchPage() {                     // the page is large; say so if it has not finished loading
  const t0 = Date.now(), tick = () => {
    let ok = false;
    try { const d = $("page").contentDocument; ok = !!(d && d.body && d.body.dataset.ready === "1"); } catch (e) {}
    if (ok) { $("note").textContent = t("note"); return; }
    if (Date.now() - t0 > 45000) {
      $("note").textContent = t("slow") + " ";
      const a = el("a", "", META.page); a.href = META.page; a.target = "_blank"; a.rel = "noopener"; $("note").appendChild(a);
      return;
    }
    setTimeout(tick, 1000);
  };
  tick();
}
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }

function setLang(l) {
  LANG = l; document.documentElement.lang = l === "zh" ? "zh-Hant" : "en";
  $("title").textContent = t("title"); $("send").textContent = t("send"); $("q").placeholder = t("ph");
  $("note").textContent = t("note"); $("langBtn").textContent = t("lang");
  if (META) { $("presetsSum").textContent = S("frame_presets"); renderPresets(); }
}
function renderPresets() {
  const ul = $("presetList"); ul.textContent = "";
  for (const p of META.presets) ul.appendChild(el("li", "", S("preset_" + p)));
  renderChips();
}
// ---- example questions: a click answers a preset directly (no LLM) ------------------------
function renderChips() {
  const box = $("chips"); box.textContent = "";
  box.setAttribute("aria-label", t("examples"));
  const groups = new Map();
  for (const ex of META.examples || []) {
    const g = ex.group[LANG] || ex.group.en;
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(ex);
  }
  for (const [g, exs] of groups) {
    const d = el("div", "cgroup"); d.appendChild(el("h2", "", g));
    for (const ex of exs) {
      const btn = el("button", "chipq", ex.label[LANG] || ex.label.en); btn.type = "button";
      btn.onclick = () => askPreset(ex.preset, ex.params, btn.textContent);
      d.appendChild(btn);
    }
    box.appendChild(d);
  }
}
async function askPreset(preset, params, label) {
  const log = $("log"); log.appendChild(el("li", "q", label));
  const li = $("tplAnswer").content.firstElementChild.cloneNode(true); log.appendChild(li);
  li.querySelector(".body").textContent = t("wait");
  try { await show(await post("/api/answer", {preset, params, lang: LANG, session: SESSION}), li); }
  catch (e) { fail(li, e); }
  li.scrollIntoView({block: "end"});
}

// ---- charts: drawn from the answer's own values (each point carries its page text) --------
const SVGNS = "http://www.w3.org/2000/svg";
const COL = ["var(--s1)", "var(--s2)"];
function sv(tag, attrs, text) { const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v); if (text != null) e.textContent = text; return e; }
function niceMax(v) { if (v <= 0) return 0; const p = Math.pow(10, Math.floor(Math.log10(v))), m = v / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p; }
function srTable(ch) {                          // the table view for screen readers
  const tb = el("table", "sr"), hr = el("tr"); hr.appendChild(el("th", "", ch.kind === "line" ? t("year") : ""));
  for (const s of ch.series) hr.appendChild(el("th", "", s.label || t("value")));
  tb.appendChild(hr);
  const xs = ch.kind === "line" ? ch.series[0].points.map(p => p.x) : ch.x;
  xs.forEach((x, i) => { const tr = el("tr"); tr.appendChild(el("th", "", String(x)));
    for (const s of ch.series) { const p = ch.kind === "line" ? s.points.find(q => q.x === x) : s.points[i]; tr.appendChild(el("td", "", p ? p.text : "")); }
    tb.appendChild(tr); });
  return tb;
}
function legend(ch) {
  const lg = el("div", "legend");
  ch.series.forEach((s, i) => { const it = el("span"), sw = el("i"); sw.style.background = COL[i % 2]; it.appendChild(sw); it.appendChild(document.createTextNode(s.label)); lg.appendChild(it); });
  return lg;
}
function tipAt(box, html, x, y) {
  let tip = box.querySelector(".tip"); if (!tip) { tip = el("div", "tip"); box.appendChild(tip); }
  tip.textContent = ""; html.forEach((line, i) => { if (i) tip.appendChild(el("br")); tip.appendChild(document.createTextNode(line)); });
  tip.style.left = Math.max(0, x + 8) + "px"; tip.style.top = Math.max(0, y - 10) + "px"; tip.hidden = false;
}
function lineChart(ch, box) {
  const W = 360, H = 150, m = {l: 34, r: 98, t: 14, b: 20};   // room for "+1.42 from 2030"
  const all = ch.series.flatMap(s => s.points), xs = ch.series[0].points.map(p => p.x);
  if (!all.length) return null;
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const lo = Math.min(0, ...all.map(p => p.y)), hi = niceMax(Math.max(0, ...all.map(p => p.y)) * 1.05) || 1;
  const X = x => m.l + (x - x0) / Math.max(1, x1 - x0) * (W - m.l - m.r);
  const Yp = y => m.t + (hi - y) / Math.max(1e-9, hi - lo) * (H - m.t - m.b);
  const g = sv("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": ch.title});
  for (const v of [lo, hi]) {
    g.appendChild(sv("line", {x1: m.l, x2: W - m.r, y1: Yp(v), y2: Yp(v), stroke: "var(--grid)", "stroke-width": 1}));
    g.appendChild(sv("text", {x: m.l - 4, y: Yp(v) + 3, "text-anchor": "end"}, (v > 0 ? "+" : "") + v.toFixed(v && Math.abs(v) < 1 ? 2 : 1))); }
  if (lo < 0) g.appendChild(sv("line", {x1: m.l, x2: W - m.r, y1: Yp(0), y2: Yp(0), stroke: "var(--grid)", "stroke-width": 1}));
  for (const mk of ch.marker) {                  // the start year(s): a hairline, labelled
    if (mk.x < x0 || mk.x > x1) continue;
    g.appendChild(sv("line", {x1: X(mk.x), x2: X(mk.x), y1: m.t - 4, y2: H - m.b, stroke: "var(--t2)", "stroke-width": 1, "stroke-opacity": .35}));
    g.appendChild(sv("text", {x: X(mk.x) + 3, y: m.t - 5}, mk.label)); }
  for (const xv of [x0, x1]) g.appendChild(sv("text", {x: X(xv), y: H - 6, "text-anchor": xv === x0 ? "start" : "end"}, String(xv)));
  const ends = [];
  ch.series.forEach((s, i) => {
    const d = s.points.map((p, j) => `${j ? "L" : "M"}${X(p.x).toFixed(1)},${Yp(p.y).toFixed(1)}`).join("");
    g.appendChild(sv("path", {d, fill: "none", stroke: COL[i % 2], "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round"}));
    const pe = s.points[s.points.length - 1];
    g.appendChild(sv("circle", {cx: X(pe.x), cy: Yp(pe.y), r: 4, fill: COL[i % 2], stroke: "var(--surface)", "stroke-width": 2}));
    ends.push({y: Yp(pe.y), text: pe.text + (ch.series.length > 1 ? " " + s.label : "")});
  });
  ends.sort((a, b) => a.y - b.y);                 // direct end labels, nudged apart when they collide
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 11) ends[i].y = ends[i - 1].y + 11;
  for (const e of ends) g.appendChild(sv("text", {class: "v", x: W - m.r + 7, y: e.y + 3}, e.text));
  const step = (W - m.l - m.r) / Math.max(1, xs.length - 1);   // hover: one column per year
  xs.forEach((xv, j) => { const r = sv("rect", {x: X(xv) - step / 2, y: m.t, width: step, height: H - m.t - m.b, fill: "transparent"});
    r.addEventListener("mouseenter", ev => { const lines = [String(xv)].concat(ch.series.map(s => (ch.series.length > 1 ? s.label + ": " : "") + (s.points[j] ? s.points[j].text + " " + ch.unit : "")));
      const bb = box.getBoundingClientRect(), rb = r.getBoundingClientRect(); tipAt(box, lines, rb.left - bb.left + rb.width / 2, rb.top - bb.top); });
    r.addEventListener("mouseleave", () => { const tp = box.querySelector(".tip"); if (tp) tp.hidden = true; });
    g.appendChild(r); });
  return g;
}
function barChart(ch, box) {
  const W = 360, n = ch.x.length, k = ch.series.length, bh = k > 1 ? 10 : 14, gap = 2, lab = 13, rowGap = 8;
  const all = ch.series.flatMap(s => s.points.map(p => p.y)); if (!all.length) return null;
  const lo = Math.min(0, ...all), hi = Math.max(0, ...all) || 1, m = {l: 2, r: 70};
  const X = v => m.l + (v - lo) / Math.max(1e-9, hi - lo) * (W - m.l - m.r);
  const rowH = lab + k * bh + (k - 1) * gap + rowGap, H = n * rowH;
  const g = sv("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": ch.title});
  const labelValues = n * k <= 4;                 // selective labels: few bars only; the table has the rest
  ch.x.forEach((cat, ci) => {
    const y0 = ci * rowH;
    g.appendChild(sv("text", {x: m.l, y: y0 + 10}, String(cat)));
    ch.series.forEach((s, si) => {
      const p = s.points[ci]; if (!p) return;
      const y = y0 + lab + si * (bh + gap), xa = X(Math.min(0, p.y)), xb = X(Math.max(0, p.y)), w = Math.max(1, xb - xa), r = Math.min(4, w, bh / 2);
      const pos = p.y >= 0;                        // 4px rounded data end, square at the baseline
      const d = pos ? `M${xa},${y}H${xb - r}Q${xb},${y} ${xb},${y + r}V${y + bh - r}Q${xb},${y + bh} ${xb - r},${y + bh}H${xa}Z`
                    : `M${xb},${y}H${xa + r}Q${xa},${y} ${xa},${y + r}V${y + bh - r}Q${xa},${y + bh} ${xa + r},${y + bh}H${xb}Z`;
      const bar = sv("path", {d, fill: COL[si % 2]});
      bar.addEventListener("mouseenter", () => { const bb = box.getBoundingClientRect(), rb = bar.getBoundingClientRect();
        tipAt(box, [String(cat), (s.label ? s.label + ": " : "") + p.text + " " + ch.unit], rb.right - bb.left, rb.top - bb.top); });
      bar.addEventListener("mouseleave", () => { const tp = box.querySelector(".tip"); if (tp) tp.hidden = true; });
      g.appendChild(bar);
      if (labelValues) g.appendChild(sv("text", {class: "v", x: (pos ? xb : xa) + 5, y: y + bh - 2}, p.text + " " + ch.unit));
    });
  });
  if (lo < 0) g.appendChild(sv("line", {x1: X(0), x2: X(0), y1: 0, y2: H, stroke: "var(--grid)", "stroke-width": 1}));
  return g;
}
function shareBar(ch) {
  const p = ch.series[0].points[0]; if (!p) return null;
  const W = 360, H = 22, w = Math.max(0, Math.min(100, p.y)) / 100 * (W - 60);
  const g = sv("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": ch.title + ": " + p.text});
  g.appendChild(sv("rect", {x: 0, y: 7, width: W - 60, height: 8, rx: 4, fill: "var(--s1track)"}));
  if (w > 0) g.appendChild(sv("rect", {x: 0, y: 7, width: Math.max(8, w), height: 8, rx: 4, fill: "var(--s1)"}));
  g.appendChild(sv("text", {class: "v", x: W - 54, y: 15}, p.text));
  return g;
}
function renderCharts(view, li) {
  const box = li.querySelector(".charts"); box.textContent = "";
  for (const ch of view.charts || []) {
    const c = el("div", "chart " + ch.block); c.appendChild(el("h3", "", ch.title + (ch.unit && ch.kind !== "share" ? " (" + ch.unit + ")" : "")));
    if (ch.series.length > 1) c.appendChild(legend(ch));
    const svg = ch.kind === "line" ? lineChart(ch, c) : ch.kind === "bars" ? barChart(ch, c) : shareBar(ch);
    if (!svg) continue;
    c.appendChild(svg); c.appendChild(srTable(ch)); box.appendChild(c);
  }
}
// clarification candidates become one-click answers (towns, start years)
function renderCands(view, li) {
  const box = li.querySelector(".cands"); box.textContent = "";
  const cl = view.clarify; if (!cl || !(cl.candidates || []).length) return;
  const preset = view.params.for_preset; if (!preset) return;
  for (const cand of cl.candidates.slice(0, 24)) {
    const b = el("button", "chipq", String(cand)); b.type = "button";
    b.onclick = () => { const p = Object.assign({}, view.params); delete p.for_preset;
      if (cl.reason === "ambiguous_town") p.town = cand; else if (cl.reason === "out_of_grid" && META.starts.includes(+cand)) p.start = +cand;
      askPreset(preset, p, String(cand)); };
    box.appendChild(b);
  }
}

// ---- the page iframe ----------------------------------------------------------------------
function syncPage(link) {
  const fr = $("page"), w = fr.contentWindow;
  return new Promise((resolve, reject) => {
    if (!link) return resolve(false);
    const h = link + "&n=" + (++NONCE);           // an unchanged hash fires no hashchange
    const to = setTimeout(() => reject(new Error("page did not update")), 8000);
    w.addEventListener("hashchange", () => w.requestAnimationFrame(() => w.requestAnimationFrame(() => {
      clearTimeout(to); resolve(true); })), {once: true});
    w.location.hash = h;
  });
}
function pageAgrees(view) {
  try {
    const d = $("page").contentDocument, head = view.values.head;
    if (head == null) return null;
    const tw = d.querySelector("#tW .vn"), hl = d.querySelector("#headline");
    return !!tw && tw.textContent === head && !!hl && hl.textContent.includes(head);
  } catch (e) { return null; }
}

// ---- answers ------------------------------------------------------------------------------
function editor(view, li) {
  const box = li.querySelector(".editor"); box.textContent = "";
  const p = Object.assign({}, view.params); delete p.for_preset;
  const preset = view.preset === "clarify" ? (view.params.for_preset || "policy_head") : view.preset;
  const sel = (label, opts, val, on) => { const lab = el("label", "", label + " "), s = el("select");
    for (const [v, txt] of opts) { const o = el("option", "", txt); o.value = v; if (String(v) === String(val)) o.selected = true; s.appendChild(o); }
    s.onchange = () => on(s.value); lab.appendChild(s); box.appendChild(lab); };
  const needs = {policy_head: ["cuts", "start", "year"], start_compare: ["cuts", "year"], lever_rank: ["year", "by"],
    township_internals: ["cuts", "start", "year", "town"], rail_internals: ["cuts", "start", "year"], solved_compare: ["year"]}[preset] || [];
  const cuts = {}; for (const c of (p.cuts || [])) cuts[c.class] = c.pct;
  if (needs.includes("cuts")) for (const c of META.classes)
    sel(LANG === "zh" ? c.zh : c.en, META.pct.map(v => [v, "−" + v + "%"]), cuts[c.key] || 0, v => { cuts[c.key] = +v; });
  if (needs.includes("start")) sel(S("echo_start"), META.starts.map(s => [s, s]), p.start ?? META.starts[0], v => { p.start = +v; });
  if (needs.includes("year")) sel(S("echo_year"), META.years.map(y => [y, y]), p.year ?? META.years[META.years.length - 1], v => { p.year = +v; });
  if (needs.includes("town")) sel(S("echo_town"), META.towns.map(x => [x.i, LANG === "zh" ? x.zh : x.en]), p.town ?? 1, v => { p.town = +v; });
  if (needs.includes("by")) sel(S("echo_metric"), [["head", S("metric_head")], ["gwh", S("metric_gwh")]], p.by || "head", v => { p.by = v; });
  const go = el("button", "", t("send")); go.type = "button";
  go.onclick = async () => {
    p.cuts = Object.entries(cuts).filter(([, v]) => v).map(([k, v]) => ({class: k, pct: v}));
    try { show(await post("/api/answer", {preset, params: p, lang: LANG, session: SESSION}), li); }   // no LLM
    catch (e) { fail(li, e); }
  };
  box.appendChild(go);
}
async function show(view, li) {
  li.querySelector(".echoLbl").textContent = view.echo.length ? S("echo_preset") + ": " : "";
  const chips = li.querySelector(".chips"); chips.textContent = "";
  for (const e of view.echo) chips.appendChild(el("span", "chip", e));
  const ed = li.querySelector(".edit"); ed.textContent = S("echo_edit"); ed.hidden = view.preset === "refuse";
  ed.onclick = () => { const box = li.querySelector(".editor"); box.hidden = !box.hidden; if (!box.hidden) editor(view, li); };
  const bn = li.querySelector(".banners"); bn.textContent = ""; for (const b of view.banners) bn.appendChild(el("div", "", b));
  const body = li.querySelector(".body"); body.textContent = "";
  for (const b of view.blocks) body.appendChild(el("div", "blk " + b.block, b.text));
  renderCharts(view, li); renderCands(view, li);
  const tb = li.querySelector("table.rank"); tb.textContent = ""; tb.hidden = !view.table.length;
  view.table.forEach((row, i) => { const tr = el("tr"); row.forEach(c => tr.appendChild(el(i ? "td" : "th", "", c))); tb.appendChild(tr); });
  const ci = li.querySelector(".cites"); ci.textContent = "";
  for (const c of view.citations) { const s = el("span", "cite " + c.status.split("+")[0], `${c.id} ${c.status}`); s.title = c.topic; ci.appendChild(s); }
  li.querySelector(".aid").textContent = view.audit_id || "";
  const op = li.querySelector(".open"), ag = li.querySelector(".agree");
  op.hidden = !view.deep_link; op.textContent = t("open");
  op.onclick = () => syncPage(view.deep_link).catch(() => {});
  ag.hidden = true;
  if (view.deep_link) {
    try { await syncPage(view.deep_link); const ok = pageAgrees(view);
      if (ok !== null) { ag.hidden = false; ag.className = "agree " + (ok ? "ok" : "bad"); ag.textContent = (ok ? "✓ " : "✗ ") + t(ok ? "agree" : "disagree"); }
    } catch (e) { /* page busy: no badge */ }
  }
}
async function ask(q) {
  const log = $("log"), qi = el("li", "q", q); log.appendChild(qi);
  const li = $("tplAnswer").content.firstElementChild.cloneNode(true); log.appendChild(li);
  li.querySelector(".body").textContent = t("wait");
  let view;
  try { view = await post("/api/ask", {question: q, session: SESSION, lang: null}); }
  catch (e) { fail(li, e); return; }
  if (view.lang && view.lang !== LANG) setLang(view.lang);
  await show(view, li);
  li.scrollIntoView({block: "end"});
}

(async () => {
  try { const r = await fetch("/api/meta"); if (!r.ok) throw new Error("HTTP " + r.status); META = await r.json(); }
  catch (e) { setLang(LANG); $("note").textContent = t("metaerr"); return; }
  $("page").src = META.page;
  setLang(LANG);
  watchPage();
  $("langBtn").onclick = () => setLang(LANG === "zh" ? "en" : "zh");
  $("ask").onsubmit = e => { e.preventDefault(); const q = $("q").value.trim(); if (q) { $("q").value = ""; ask(q); } };
  $("q").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("ask").requestSubmit(); } });
})();
