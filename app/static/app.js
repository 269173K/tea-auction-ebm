/* Tea auction outlook: UI + a JavaScript twin of the EBM.
   With the Flask server running, forecasts come from /api/predict (the pickled Python model).
   Opened standalone (no server), the same numbers come from the JS twin below, which reads the
   exported model tables; app/test_parity.js checks the two agree. */
"use strict";

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const ELEV = {
  high: { name: "High grown", where: "Nuwara Eliya, Uva, Dimbula: above 1,200 m", color: "var(--high)" },
  medium: { name: "Medium grown", where: "Kandy, Matale: 600 to 1,200 m", color: "var(--medium)" },
  low: { name: "Low grown", where: "Ratnapura, Galle, Matara: below 600 m", color: "var(--low)" },
};
const ELEV_CODE = { high: 0, medium: 1, low: 2 };
const SHORT = {
  elevation: "Elevation", next_month: "Month to forecast", ret_1m: "Last month's change", ret_12m: "12-month change",
  dev_12m_mean: "Price vs 12-month average", premium_vs_total: "Price vs national average",
  fx_chg_3m: "Rupee, 3-month change", brent_chg_3m: "Brent oil, 3-month change", world_tea_chg_3m: "World tea price, 3-month change",
  ceylon_premium: "Colombo premium over world", rain_anom_2m: "Rainfall vs normal", temp_anom: "Temperature vs normal",
  solar_anom: "Sunshine vs normal", fertilizer_ban: "Fertiliser-ban period",
};
const GROUPS = {
  market: ["Market", "Global prices and the rupee, measured up to the end of the latest month."],
  weather: ["Weather in the growing district", "NASA POWER readings compared with the same calendar month in 2017 to 2022."],
  prices: ["This elevation's recent prices", "Measured in US dollars per kg."],
};

/* ------------------------------------------------------------------ EBM twin */
const Engine = (M) => {
  const order = M.feature_order;
  const feat = Object.fromEntries(M.features.map((f, j) => [f.name, { ...f, j }]));
  const bin = (spec, x) => {
    if (spec.categorical) {              // searchsorted-left on levels, clipped (as numpy)
      const lv = spec.levels; let i = 0; while (i < lv.length && lv[i] < x) i++;
      return Math.min(i, lv.length - 1);
    }
    let i = 0; const e = spec.edges; while (i < e.length && e[i] <= x) i++;   // searchsorted-right
    return i;
  };
  const pairBin = (p, k, x) => bin({ categorical: p.levels[k] !== null, levels: p.levels[k], edges: p.edges[k] }, x);
  const bg = M.shap_background;
  function explain(values) {            // values: {feature: model-unit value}
    let pred = M.intercept, base = M.intercept;
    const phi = Object.fromEntries(order.map((f) => [f, 0]));
    M.features.forEach((f, j) => {
      const s = f.scores[bin(f, values[f.name])];
      pred += s; base += bg.main_means[j]; phi[f.name] += s - bg.main_means[j];
    });
    M.pairs.forEach((p, k) => {
      const [a, b] = p.features, bi = pairBin(p, 0, values[a]), bj = pairBin(p, 1, values[b]);
      const full = p.table[bi][bj], g = bg.pairs[k], Ei = g.Ei_given_j[bj], Ej = g.Ej_given_i[bi];
      pred += full; base += g.E;
      phi[a] += 0.5 * (full - Ei) + 0.5 * (Ej - g.E);
      phi[b] += 0.5 * (full - Ej) + 0.5 * (Ei - g.E);
    });
    return { pred, base, phi };
  }
  function monthEffect(elevCode, m) {   // month main effect + elevation x month pair
    let v = feat.next_month.scores[bin(feat.next_month, m)];
    M.pairs.forEach((p) => {
      if (p.features.includes("next_month") && p.features.includes("elevation")) {
        const vals = { elevation: elevCode, next_month: m };
        v += p.table[pairBin(p, 0, vals[p.features[0]])][pairBin(p, 1, vals[p.features[1]])];
      }
    });
    return v;
  }
  return { explain, monthEffect };
};
if (typeof module !== "undefined") module.exports = { Engine };

/* ------------------------------------------------------------------ helpers */
const $ = (s) => document.querySelector(s);
const svgNS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, text) => {
  const n = document.createElementNS(svgNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  return n;
};
const fmt = (v, d = 0) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const signed = (v, d = 1) => (v > 0 ? "+" : v < 0 ? "−" : "") + fmt(Math.abs(v), d);
const toUI = (f, v) => (f.transform === "pct_log" ? (Math.exp(v) - 1) * 100 : v);
const toModel = (f, u) => (f.transform === "pct_log" ? Math.log(1 + u / 100) : u);
const monthLabel = (ym) => `${MONTHS[Number(ym.slice(5, 7)) - 1]} ${ym.slice(0, 4)}`;

/* ------------------------------------------------------------------ app */
let B, engine, elev = "high", ui = {}, serverOK = true, timer;

async function boot() {
  try {
    const r = await fetch("api/bundle"); if (!r.ok) throw new Error(r.status);
    B = await r.json();
  } catch (e) {
    serverOK = false; B = window.EMBEDDED_BUNDLE;
  }
  if (!B) { $("#fc-title").textContent = "The model bundle could not be loaded. Start the app with python app/app.py."; return; }
  engine = Engine(B.model);
  $("#data-through").textContent = `Auction data up to ${monthLabel(B.data_through)}.`;
  buildElevations(); buildInputs(); buildShapeSelect(); writeHonesty();
  loadLatest(); update();
}

function buildElevations() {
  const row = $("#elev-row");
  for (const [k, e] of Object.entries(ELEV)) {
    const st = B.state[k];
    const b = document.createElement("button");
    b.type = "button"; b.className = "elev"; b.setAttribute("role", "radio"); b.dataset.elev = k;
    b.style.setProperty("--c", e.color);
    b.innerHTML = `<span class="name">${e.name}</span><span class="where">${e.where}</span>
      <span class="now">Rs ${fmt(st.lkr_t)} /kg in ${MONTHS[Number(st.origin.slice(5)) - 1]}</span>`;
    b.addEventListener("click", () => { elev = k; loadLatest(); update(); });
    b.addEventListener("keydown", (ev) => {
      const keys = Object.keys(ELEV), i = keys.indexOf(k);
      if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") {
        const n = keys[(i + (ev.key === "ArrowRight" ? 1 : 2)) % 3];
        elev = n; loadLatest(); update(); row.querySelector(`[data-elev=${n}]`).focus();
      }
    });
    row.appendChild(b);
  }
}

function buildInputs() {
  const ms = $("#month");
  MONTHS.forEach((m, i) => ms.add(new Option(m, i + 1)));
  ms.addEventListener("change", () => { ui.next_month = Number(ms.value); update(); });
  const groups = $("#groups");
  for (const [g, [title, hint]] of Object.entries(GROUPS)) {
    const fs = document.createElement("fieldset"); fs.className = "group";
    fs.innerHTML = `<legend>${title}</legend><p class="hint">${hint}</p>`;
    B.features.filter((f) => f.group === g).forEach((f) => {
      const id = `in-${f.name}`;
      const wrap = document.createElement("div"); wrap.className = "field"; wrap.id = `wrap-${f.name}`;
      wrap.innerHTML = `<span id="lab-${f.name}">${f.label}</span>
        <div class="slider-row">
          <input type="range" id="${id}" min="${f.min}" max="${f.max}" step="${f.step}" aria-labelledby="lab-${f.name}">
          <div class="num"><input type="number" id="${id}-n" step="${f.step}" aria-labelledby="lab-${f.name}"><small>${f.unit}</small></div>
        </div>`;
      fs.appendChild(wrap);
      const r = wrap.querySelector("input[type=range]"), n = wrap.querySelector("input[type=number]");
      const set = (v) => { ui[f.name] = v; r.value = v; n.value = fmt(v, 1).replace(/,/g, ""); markChanged(f); schedule(); };
      r.addEventListener("input", () => set(Number(r.value)));
      n.addEventListener("change", () => { const v = Number(n.value); if (Number.isFinite(v)) set(v); });
    });
    groups.appendChild(fs);
  }
  $("#fx").addEventListener("change", () => { ui.fx = Number($("#fx").value); update(); });
  $("#reset").addEventListener("click", () => { loadLatest(); update(); });
}

function latestUI(f) { return toUI(f, B.state[elev].features[f.name]); }
function markChanged(f) {
  const w = $(`#wrap-${f.name}`);
  w.classList.toggle("changed", Math.abs(ui[f.name] - latestUI(f)) > 1e-6);
}

function loadLatest() {
  const st = B.state[elev];
  document.querySelectorAll(".elev").forEach((b) => b.setAttribute("aria-checked", b.dataset.elev === elev));
  B.features.forEach((f) => {
    const v = Math.round(latestUI(f) * 10) / 10;
    ui[f.name] = v; $(`#in-${f.name}`).value = v; $(`#in-${f.name}-n`).value = v.toFixed(1);
    markChanged(f);
  });
  ui.next_month = Number(st.features.next_month); $("#month").value = ui.next_month;
  ui.fx = Math.round(st.fx_t * 10) / 10; $("#fx").value = ui.fx;
}

function modelValues() {
  const st = B.state[elev], v = { ...st.features };
  B.features.forEach((f) => { v[f.name] = toModel(f, ui[f.name]); });
  v.next_month = ui.next_month; v.elevation = ELEV_CODE[elev];
  return v;
}

function schedule() { clearTimeout(timer); timer = setTimeout(update, 120); }

async function update() {
  const vals = modelValues(), st = B.state[elev];
  let res;
  if (serverOK) {
    try {
      const feats = { ...vals }; delete feats.elevation;
      const r = await fetch("api/predict", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ elevation: elev, features: feats, fx: ui.fx }) });
      if (!r.ok) throw new Error(r.status);
      const j = await r.json();
      res = { pred: Math.log(1 + j.pred_change_pct / 100), base: j.base_pct / 100,
              phi: Object.fromEntries(j.contributions.map((c) => [c.feature, c.pp / 100])), engine: "flask" };
    } catch (e) { serverOK = false; }
  }
  if (!res) res = { ...engine.explain(vals), engine: "browser" };
  renderForecast(res, st); renderWhy(res); renderHistory(res, st); renderShape();
}

function renderForecast(res, st) {
  const pct = (Math.exp(res.pred) - 1) * 100, usdF = st.usd_t * Math.exp(res.pred), lkrF = usdF * ui.fx;
  const isTarget = ui.next_month === Number(st.target.slice(5));
  $("#fc-title").textContent = `${ELEV[elev].name}, ${isTarget ? monthLabel(st.target) : MONTHS[ui.next_month - 1] + " (what-if month)"}`;
  $("#fc-lkr").textContent = fmt(lkrF);
  const ch = $("#fc-change");
  ch.textContent = `${signed(pct)}% in dollars`;
  ch.className = "change " + (pct > 0.05 ? "up" : pct < -0.05 ? "down" : "");
  $("#fc-sub").textContent = `Forecast US$${fmt(usdF, 2)}/kg against US$${fmt(st.usd_t, 2)} in ${monthLabel(st.origin)} (Rs ${fmt(st.lkr_t)}). ` +
    `The rupee price assumes Rs ${fmt(ui.fx, 1)} per dollar.`;
}

function renderWhy(res) {
  const box = $("#why"); box.innerHTML = "";
  const items = Object.entries(res.phi).map(([f, v]) => ({ f, v: v * 100 })).filter((d) => d.f !== "fertilizer_ban")
    .sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
  const top = items.slice(0, 7), rest = items.slice(7).reduce((s, d) => s + d.v, 0);
  const rows = [{ label: "Average month (starting point)", v: res.base * 100, kind: "base" },
    ...top.map((d) => ({ label: SHORT[d.f] || d.f, v: d.v, kind: "term" })),
    { label: `Other ${items.length - 7} conditions`, v: rest, kind: "term" },
    { label: "Forecast change", v: res.pred * 100, kind: "total" }];
  const W = 720, labelW = 230, rowH = 30, H = rows.length * rowH + 30;
  const maxAbs = Math.max(0.6, ...rows.map((r) => Math.abs(r.v))) * 1.15;
  const x0 = labelW + (W - labelW) / 2, sc = (W - labelW - 70) / 2 / maxAbs;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Contribution of each condition to the forecast" });
  svg.appendChild(el("line", { x1: x0, x2: x0, y1: 4, y2: H - 22, stroke: "var(--leaf)", "stroke-width": 1 }));
  rows.forEach((r, i) => {
    const y = 8 + i * rowH, w = Math.abs(r.v) * sc, x = r.v >= 0 ? x0 : x0 - w;
    const fill = r.kind === "base" ? "var(--muted-bar)" : r.kind === "total" ? "var(--leaf)" : r.v >= 0 ? "var(--up)" : "var(--down)";
    if (r.kind === "total") svg.appendChild(el("line", { x1: 0, x2: W, y1: y - 5, y2: y - 5, stroke: "var(--line)" }));
    svg.appendChild(el("text", { x: labelW - 12, y: y + 15, "text-anchor": "end", "font-size": 13, "font-weight": r.kind === "total" ? 600 : 400 }, r.label));
    svg.appendChild(el("rect", { x, y: y + 2, width: Math.max(w, 1.5), height: rowH - 10, rx: 3, fill }));
    const tx = r.v >= 0 ? x0 + w + 6 : x0 - w - 6;
    svg.appendChild(el("text", { x: tx, y: y + 15, "text-anchor": r.v >= 0 ? "start" : "end", "font-size": 12.5, class: "cond", "font-weight": r.kind === "total" ? 700 : 500 },
      `${signed(r.v, 2)} pp`));
  });
  svg.appendChild(el("text", { x: x0 - 8, y: H - 6, "text-anchor": "end", "font-size": 11.5, fill: "var(--down)" }, "pushes price down"));
  svg.appendChild(el("text", { x: x0 + 8, y: H - 6, "font-size": 11.5, fill: "var(--up)" }, "pushes price up"));
  box.appendChild(svg);
}

function renderHistory(res, st) {
  const box = $("#history"); box.innerHTML = "";
  const h = B.history[elev], fc = st.usd_t * Math.exp(res.pred) * ui.fx;
  const pts = h.map((d) => d.lkr), all = [...pts, fc];
  const W = 720, H = 240, L = 56, R = 70, T = 14, Bm = 30;
  const lo = Math.floor(Math.min(...all) / 50) * 50, hi = Math.ceil(Math.max(...all) / 50) * 50;
  const X = (i) => L + (i / h.length) * (W - L - R), Y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - Bm);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${ELEV[elev].name} auction price, last 36 months, with the forecast` });
  const step = (hi - lo) / 4;
  for (let k = 0; k <= 4; k++) {
    const v = lo + k * step;
    svg.appendChild(el("line", { x1: L, x2: W - R + 20, y1: Y(v), y2: Y(v), stroke: "var(--line)", "stroke-width": k === 0 ? 1 : 0.6 }));
    svg.appendChild(el("text", { x: L - 8, y: Y(v) + 4, "text-anchor": "end", "font-size": 11.5, fill: "var(--leaf-soft)" }, fmt(v)));
  }
  svg.appendChild(el("text", { x: L - 8, y: T - 2, "text-anchor": "end", "font-size": 11.5, fill: "var(--leaf-soft)" }, "Rs/kg"));
  h.forEach((d, i) => {
    if (d.month.endsWith("-01")) {
      svg.appendChild(el("line", { x1: X(i), x2: X(i), y1: H - Bm, y2: H - Bm + 5, stroke: "var(--leaf-soft)" }));
      svg.appendChild(el("text", { x: X(i), y: H - 8, "text-anchor": "middle", "font-size": 11.5, fill: "var(--leaf-soft)" }, d.month.slice(0, 4)));
    }
  });
  const path = h.map((d, i) => `${i ? "L" : "M"}${X(i).toFixed(1)} ${Y(d.lkr).toFixed(1)}`).join("");
  svg.appendChild(el("path", { d: path, fill: "none", stroke: ELEV[elev].color, "stroke-width": 2.2, "stroke-linejoin": "round" }));
  const li = h.length - 1;
  svg.appendChild(el("line", { x1: X(li), y1: Y(h[li].lkr), x2: X(h.length), y2: Y(fc), stroke: "var(--leaf)", "stroke-width": 1.6, "stroke-dasharray": "4 4" }));
  svg.appendChild(el("circle", { cx: X(h.length), cy: Y(fc), r: 5, fill: "var(--sheet)", stroke: "var(--leaf)", "stroke-width": 2 }));
  svg.appendChild(el("text", { x: X(h.length) + 9, y: Y(fc) + 4, "font-size": 12.5, class: "cond", "font-weight": 600 }, `${fmt(fc)} forecast`));
  box.appendChild(svg);
}

function buildShapeSelect() {
  const s = $("#shape-feature");
  s.add(new Option("Month to forecast", "next_month"));
  B.features.forEach((f) => s.add(new Option(f.label, f.name)));
  s.value = "next_month";
  s.addEventListener("change", renderShape);
}

function renderShape() {
  const f = $("#shape-feature").value, box = $("#shape"); box.innerHTML = "";
  const W = 720, H = 230, L = 56, R = 16, T = 14, Bm = 34;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Learned effect of ${SHORT[f] || f}` });
  if (f === "next_month") {
    const code = ELEV_CODE[elev], vals = MONTHS.map((_, i) => engine.monthEffect(code, i + 1) * 100);
    const m = Math.max(...vals.map(Math.abs)) * 1.15, Y = (v) => T + (1 - (v + m) / (2 * m)) * (H - T - Bm);
    const bw = (W - L - R) / 12;
    svg.appendChild(el("line", { x1: L, x2: W - R, y1: Y(0), y2: Y(0), stroke: "var(--leaf)" }));
    vals.forEach((v, i) => {
      const x = L + i * bw + bw * 0.18, sel = i + 1 === ui.next_month;
      svg.appendChild(el("rect", { x, y: Math.min(Y(v), Y(0)), width: bw * 0.64, height: Math.max(Math.abs(Y(v) - Y(0)), 1), rx: 2,
        fill: v >= 0 ? "var(--up)" : "var(--down)", opacity: sel ? 1 : 0.45 }));
      svg.appendChild(el("text", { x: L + i * bw + bw / 2, y: H - 14, "text-anchor": "middle", "font-size": 12, "font-weight": sel ? 600 : 400 }, MONTHS[i].slice(0, 3)));
      if (sel || Math.abs(v) === Math.max(...vals.map(Math.abs)))
        svg.appendChild(el("text", { x: L + i * bw + bw / 2, y: v >= 0 ? Y(v) - 6 : Y(v) + 14, "text-anchor": "middle", "font-size": 11.5, class: "cond" }, `${signed(v, 2)}`));
    });
    svg.appendChild(el("text", { x: L - 8, y: T + 4, "text-anchor": "end", "font-size": 11.5, fill: "var(--leaf-soft)" }, "pp"));
    $("#shape-note").textContent = `Effect of the calendar month on the forecast for ${ELEV[elev].name.toLowerCase()}, in percentage points. June is the seasonal low point; the selected month is solid.`;
  } else {
    const s = B.shapes[f], meta = B.features.find((x) => x.name === f), cur = ui[f];
    const xs = s.x, lo = Math.min(meta.min, ...xs), hi = Math.max(meta.max, ...xs);
    const yAll = s.score_pp.flatMap((v, i) => [v - s.std_pp[i], v + s.std_pp[i]]);
    const ylo = Math.min(...yAll, 0) * 1.15, yhi = Math.max(...yAll, 0) * 1.15;
    const X = (v) => L + (v - lo) / (hi - lo) * (W - L - R), Y = (v) => T + (1 - (v - ylo) / (yhi - ylo)) * (H - T - Bm);
    // steps: each bin spans midway to its neighbours' representative points
    const bounds = xs.map((x, i) => [i === 0 ? lo : (xs[i - 1] + x) / 2, i === xs.length - 1 ? hi : (x + xs[i + 1]) / 2]);
    let band = "", line = "";
    bounds.forEach(([a, b], i) => {
      const v = s.score_pp[i], sd = s.std_pp[i];
      band += `<rect x="${X(a)}" y="${Y(v + sd)}" width="${Math.max(X(b) - X(a), 0)}" height="${Math.max(Y(v - sd) - Y(v + sd), 0)}" fill="var(--leaf)" opacity="0.1"/>`;
      line += `${i ? "L" : "M"}${X(a).toFixed(1)} ${Y(v).toFixed(1)}L${X(b).toFixed(1)} ${Y(v).toFixed(1)}`;
    });
    svg.innerHTML = band;
    svg.appendChild(el("line", { x1: L, x2: W - R, y1: Y(0), y2: Y(0), stroke: "var(--leaf)", "stroke-width": 0.8 }));
    svg.appendChild(el("path", { d: line, fill: "none", stroke: "var(--leaf)", "stroke-width": 2.2 }));
    const ci = bounds.findIndex(([a, b]) => cur >= a && cur <= b), cy = ci >= 0 ? s.score_pp[ci] : s.score_pp.at(cur < lo ? 0 : -1);
    svg.appendChild(el("line", { x1: X(Math.min(Math.max(cur, lo), hi)), x2: X(Math.min(Math.max(cur, lo), hi)), y1: T, y2: H - Bm, stroke: "var(--focus)", "stroke-dasharray": "3 3" }));
    svg.appendChild(el("circle", { cx: X(Math.min(Math.max(cur, lo), hi)), cy: Y(cy), r: 5, fill: "var(--focus)" }));
    const ticks = 5;
    for (let k = 0; k <= ticks; k++) {
      const v = lo + (hi - lo) * k / ticks;
      svg.appendChild(el("text", { x: X(v), y: H - 12, "text-anchor": "middle", "font-size": 11.5, fill: "var(--leaf-soft)" }, fmt(v, 1)));
    }
    svg.appendChild(el("text", { x: W - R, y: H - 0, "text-anchor": "end", "font-size": 11.5, fill: "var(--leaf-soft)" }, meta.unit));
    [ylo, 0, yhi].forEach((v) => svg.appendChild(el("text", { x: L - 8, y: Y(v) + 4, "text-anchor": "end", "font-size": 11.5, fill: "var(--leaf-soft)" }, signed(v, 2))));
    $("#shape-note").textContent = `${meta.label}: its effect on the forecast in percentage points. The shaded band is the spread across 16 bootstrap bags; the blue marker is the current input.`;
  }
  box.appendChild(svg);
}

function writeHonesty() {
  const m = B.metrics;
  $("#honesty").innerHTML = `<strong>How far to trust it.</strong> On the held-out 2025 to 2026 months the forecast missed the real monthly change by
    ${fmt(m.test_rmse_ebm, 1)}% on average (RMSE), against ${fmt(m.test_rmse_naive, 1)}% for simply assuming no change, and called the direction right
    ${fmt(m.test_dir_ebm, 0)}% of the time. Across July 2023 to September 2026 it was the most accurate model tested
    (${fmt(m.rolling_rmse_ebm, 2)}% vs ${fmt(m.rolling_rmse_naive, 2)}%). Monthly auction prices are close to a random walk: treat this as a structured
    second opinion on direction and drivers, not a price guarantee. Sources: Forbes &amp; Walker / Sri Lanka Tea Board, FRED, x-rates.com, NASA POWER.
    <span id="engine-note"></span>`;
}

if (typeof window !== "undefined") window.addEventListener("DOMContentLoaded", boot);
