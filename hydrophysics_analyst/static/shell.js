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
       metaerr: "Cannot reach the analyst server. Is it still running?"},
  zh: {title: "雙生模型分析助理", send: "提問", ph: "例如：2030年起停止養殖抽水，2032年水位？",
       note: "答案為頁面本身的數值與文字，並附主張編號。可修改解讀以更正。",
       lang: "English", agree: "與頁面一致", disagree: "與頁面不同", open: "在頁面中開啟", wait: "…",
       err: "無法取得答案：", slow: "頁面仍在載入。若一直沒有出現，請直接開啟：",
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
