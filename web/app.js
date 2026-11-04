import { createBackend } from "./backend.js";
import { BoxChart, LineChart, formatNumber } from "./charts.js";

const COLORS = ["#2f6fdf", "#e8710a", "#1e9e5a", "#c43b8a", "#7a52c7", "#a77b0e", "#0e9aa7", "#d93a3a"];
const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children) if (c) node.append(c);
  return node;
};

const state = {
  backend: null,
  scenarios: [],
  scenario: null,
  system: {},
  filters: [],
  nextFilterId: 1,
  session: null,
  running: false,
};

// ------------------------------------------------------------------ start-up
async function main() {
  setupTabs();
  setupRunControls();
  state.backend = createBackend();
  $("#backend-info").textContent =
    state.backend.kind === "pyodide" ? "Python runs in your browser (Pyodide)" : "Python runs on the local dev server";
  try {
    await state.backend.init(setStatus);
    state.scenarios = await state.backend.call("scenarios");
  } catch (err) {
    setStatus("Could not start Python");
    showError(`Could not load the Python runtime: ${err.message ?? err}`);
    return;
  }
  const select = $("#scenario");
  for (const sc of state.scenarios) select.append(el("option", { value: sc.key, text: sc.title }));
  select.addEventListener("change", () => chooseScenario(select.value));
  await chooseScenario(state.scenarios[0].key);
  $("#start").disabled = false;
  setStatus("Ready");
}

function setStatus(text) {
  $("#status").textContent = text;
}

function showError(text) {
  $("#error").textContent = text || "";
}

function setupTabs() {
  for (const button of document.querySelectorAll(".tabs button")) {
    button.addEventListener("click", () => showTab(button.dataset.tab));
  }
}

function showTab(name) {
  for (const button of document.querySelectorAll(".tabs button")) {
    button.setAttribute("aria-selected", String(button.dataset.tab === name));
  }
  for (const panel of document.querySelectorAll(".tab-panel")) panel.hidden = panel.id !== `tab-${name}`;
}

// ------------------------------------------------------------------ setup tab
async function chooseScenario(key) {
  state.scenario = state.scenarios.find((s) => s.key === key);
  $("#scenario").value = key;
  $("#scenario-description").textContent = state.scenario.description;
  state.system = defaultsOf(state.scenario.system_params);
  renderSystem();
  const typeSelect = $("#filter-type");
  typeSelect.replaceChildren(...state.scenario.filter_types.map((ft) => el("option", { value: ft.key, text: ft.label })));
  state.filters = [];
  const first = state.scenario.filter_types[0].key;
  await addFilter(first, `${shortName(first)} (matched)`);
}

function defaultsOf(params) {
  return Object.fromEntries(params.map((p) => [p.key, p.default]));
}

function renderSystem() {
  const container = $("#system-params");
  container.replaceChildren(
    paramControls(state.scenario.system_params, state.system, defaultsOf(state.scenario.system_params), (k, v) => {
      state.system[k] = v;
    }),
  );
}

$("#reset-system").addEventListener("click", () => {
  state.system = defaultsOf(state.scenario.system_params);
  renderSystem();
});

$("#add-filter").addEventListener("click", () => addFilter($("#filter-type").value));

async function matchedDefaults(type) {
  return state.backend.call("filter_defaults", state.scenario.key, type, JSON.stringify(state.system));
}

async function addFilter(type, name) {
  const id = state.nextFilterId++;
  const used = new Set(state.filters.map((f) => f.color));
  const color = COLORS.find((c) => !used.has(c)) ?? COLORS[id % COLORS.length];
  const params = await matchedDefaults(type);
  state.filters.push({ id, type, name: name ?? uniqueName(`${shortName(type)} ${state.filters.filter((f) => f.type === type).length + 1}`), color, params, matched: { ...params } });
  renderFilters();
}

function shortName(type) {
  return type.toUpperCase();  // kf -> KF, ekf -> EKF, ukf -> UKF, pf -> PF
}

function uniqueName(base) {
  const names = new Set(state.filters.map((f) => f.name));
  if (!names.has(base)) return base;
  for (let k = 2; ; k++) if (!names.has(`${base} (${k})`)) return `${base} (${k})`;
}

function renderFilters() {
  const container = $("#filters");
  container.replaceChildren(...state.filters.map(filterCard));
  if (!state.filters.length) container.append(el("p", { class: "hint", text: "Add at least one filter." }));
}

function filterCard(f) {
  const ft = state.scenario.filter_types.find((t) => t.key === f.type);
  const name = el("input", {
    type: "text",
    value: f.name,
    "aria-label": "Filter name",
    oninput: (e) => (f.name = e.target.value),
  });
  const card = el(
    "div",
    { class: "filter-card", style: `--color: ${f.color}` },
    el(
      "header",
      {},
      name,
      el(
        "span",
        { class: "actions" },
        el("button", {
          class: "link", type: "button", title: "Set the filter's constants to the true system's values",
          text: "Match truth",
          onclick: async () => {
            f.params = await matchedDefaults(f.type);
            f.matched = { ...f.params };
            renderFilters();
          },
        }),
        el("button", {
          class: "link", type: "button", text: "Duplicate",
          onclick: () => {
            const used = new Set(state.filters.map((g) => g.color));
            const id = state.nextFilterId++;
            state.filters.splice(state.filters.indexOf(f) + 1, 0, {
              ...f, id, name: uniqueName(`${f.name} copy`), params: { ...f.params },
              color: COLORS.find((c) => !used.has(c)) ?? COLORS[id % COLORS.length],
            });
            renderFilters();
          },
        }),
        el("button", {
          class: "link", type: "button", text: "Remove", "aria-label": `Remove ${f.name}`,
          onclick: () => {
            state.filters = state.filters.filter((g) => g !== f);
            renderFilters();
          },
        }),
      ),
    ),
    el("p", { class: "type", text: `${ft.label} — ${ft.description}` }),
    paramControls(ft.params, f.params, f.matched, (k, v) => (f.params[k] = v)),
  );
  return card;
}

// One control per parameter, grouped; "reference" marks values that differ from it.
function paramControls(params, values, reference, onChange) {
  const root = el("div");
  const groups = new Map();
  for (const p of params) {
    const g = p.group || "";
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(p);
  }
  for (const [group, items] of groups) {
    const box = el("div", { class: "group" }, group ? el("h3", { text: group }) : null);
    for (const p of items) box.append(paramControl(p, values, reference, onChange));
    root.append(box);
  }
  return root;
}

function paramControl(p, values, reference, onChange) {
  const id = `p-${Math.random().toString(36).slice(2, 9)}`;
  const row = el("div", { class: `param ${p.kind === "bool" ? "bool" : ""}`, title: p.help || "" });
  const mark = () => row.classList.toggle("changed", reference && reference[p.key] !== values[p.key]);
  if (p.kind === "bool") {
    const box = el("input", {
      type: "checkbox", id, checked: !!values[p.key],
      onchange: (e) => { values[p.key] = e.target.checked; onChange(p.key, values[p.key]); mark(); },
    });
    row.append(box, el("label", { for: id, text: p.label }));
    mark();
    return row;
  }
  const number = el("input", { type: "number", id, step: "any", min: p.min, max: p.max, value: values[p.key] });
  const slider = el("input", { type: "range", "aria-label": p.label });
  const log = p.log && p.min > 0;
  if (log) {
    Object.assign(slider, { min: Math.log10(p.min), max: Math.log10(p.max), step: (Math.log10(p.max) - Math.log10(p.min)) / 400 });
  } else {
    Object.assign(slider, { min: p.min, max: p.max, step: p.step ?? (p.max - p.min) / 400 });
  }
  const toSlider = (v) => (log ? Math.log10(v) : v);
  slider.value = toSlider(values[p.key]);
  const set = (v, from) => {
    if (!isFinite(v)) return;
    v = Math.min(p.max, Math.max(p.min, v));
    if (p.kind === "int") v = Math.round(v);
    values[p.key] = v;
    if (from !== "number") number.value = log ? Number(v.toPrecision(3)) : Number(v.toFixed(6));
    if (from !== "slider") slider.value = toSlider(v);
    onChange(p.key, v);
    mark();
  };
  slider.addEventListener("input", () => {
    const raw = Number(slider.value);
    set(log ? Math.pow(10, raw) : raw, "slider");
  });
  number.addEventListener("change", () => set(Number(number.value), "number"));
  row.append(el("label", { for: id, text: p.label }), number, slider);
  mark();
  return row;
}

// ------------------------------------------------------------------ run controls
function setupRunControls() {
  const update = () => {
    const mc = document.querySelector('input[name="mode"]:checked').value === "montecarlo";
    for (const n of document.querySelectorAll(".mc-only")) n.hidden = !mc;
    for (const n of document.querySelectorAll(".single-only")) n.hidden = mc;
    $("#seed-label").textContent = mc ? "First seed (run k uses seed + k)" : "Seed";
  };
  for (const r of document.querySelectorAll('input[name="mode"]')) r.addEventListener("change", update);
  update();
  $("#start").addEventListener("click", start);
  $("#stop").addEventListener("click", () => stop("Stopped"));
  $("#stop-top").addEventListener("click", () => stop("Stopped"));
}

function readConfig() {
  const mode = document.querySelector('input[name="mode"]:checked').value;
  const names = state.filters.map((f) => f.name.trim());
  if (!state.filters.length) throw new Error("Add at least one filter.");
  if (names.some((n) => !n)) throw new Error("Every filter needs a name.");
  if (new Set(names).size !== names.length) throw new Error("Filter names must be unique.");
  return {
    scenario: state.scenario.key,
    system: state.system,
    filters: state.filters.map((f) => ({ name: f.name.trim(), type: f.type, params: f.params })),
    mode,
    seed: Math.max(0, Math.round(Number($("#seed").value) || 0)),
    runs: Math.min(10000, Math.max(1, Math.round(Number($("#runs").value) || 1))),
  };
}

async function start() {
  showError("");
  let config;
  try {
    config = readConfig();
  } catch (err) {
    showError(err.message);
    return;
  }
  await stop();
  try {
    state.session = await state.backend.call("create_session", JSON.stringify(config));
  } catch (err) {
    showError(`Could not start: ${err.message ?? err}`);
    return;
  }
  state.running = true;
  state.colors = Object.fromEntries(state.filters.map((f) => [f.name.trim(), f.color]));
  $("#start").disabled = true;
  $("#stop").disabled = false;
  $("#stop-top").hidden = false;
  $("#sim-empty").hidden = true;
  $("#metrics-empty").hidden = true;
  $("#metrics-content").hidden = false;
  showTab("simulation");
  try {
    if (config.mode === "single") await runSingle(config);
    else await runMonteCarlo(config);
  } catch (err) {
    showError(`Simulation error: ${err.message ?? err}`);
    await stop("Error");
  }
}

async function stop(message) {
  const sid = state.session;
  state.running = false;
  state.session = null;
  if (sid !== null && sid !== undefined) {
    try { await state.backend.call("close_session", sid); } catch { /* already gone */ }
  }
  $("#start").disabled = false;
  $("#stop").disabled = true;
  $("#stop-top").hidden = true;
  if (message) setStatus(message);
}

const nextFrame = () => new Promise((resolve) => requestAnimationFrame(() => resolve()));

function setProgress(fraction, text) {
  $("#progress-bar").style.width = `${Math.round(100 * Math.min(1, Math.max(0, fraction)))}%`;
  $("#progress-text").textContent = text;
}

// ------------------------------------------------------------------ single run
async function runSingle(config) {
  const sid = state.session;
  $("#sim-single").hidden = false;
  $("#sim-mc").hidden = true;
  const data = { t: [], truth: [], meas: {}, est: {}, current: null, metrics: null };
  let charts = null;
  const speed = Number($("#speed").value);
  let wall0 = null, info = null, lastMetrics = 0, lastDraw = 0;

  while (state.running && state.session === sid) {
    const now = performance.now();
    const args = { budget: 0.025, metrics: now - lastMetrics > 400 };
    if (speed > 0) {
      if (!info) args.until_t = -1e300; // first call: only the first step, to learn the time span
      else {
        if (wall0 === null) wall0 = now;
        args.until_t = info.t_start + ((now - wall0) / 1000) * speed;
      }
    }
    const chunk = await state.backend.call("advance", sid, JSON.stringify(args));
    if (state.session !== sid) return;
    if (chunk.info) {
      info = chunk.info;
      charts = buildSingleCharts(info);
      for (const name of info.filters) data.est[name] = { mean: [], var: [] };
    }
    append(data, chunk);
    if (args.metrics || chunk.metrics) lastMetrics = now;
    if (chunk.metrics) data.metrics = chunk.metrics;
    if (now - lastDraw > 50 || chunk.done) {
      drawSingle(charts, info, data);
      lastDraw = now;
    }
    setProgress(chunk.progress, `t = ${formatNumber(chunk.t_now)} s of ${formatNumber(info.t_end)} s`);
    setStatus(chunk.done ? "Finished" : `Running… ${Math.round(100 * chunk.progress)}%`);
    if (chunk.done) break;
    await nextFrame();
  }
  if (state.session === sid) await stop("Finished");
}

function append(data, chunk) {
  for (let i = 0; i < chunk.t.length; i++) {
    data.t.push(chunk.t[i]);
    data.truth.push(chunk.truth[i]);
  }
  for (const m of chunk.measurements) {
    (data.meas[m.sensor] ??= { t: [], z: [] });
    data.meas[m.sensor].t.push(m.t);
    data.meas[m.sensor].z.push(m.z[0]);
  }
  for (const [name, rows] of Object.entries(chunk.estimates)) {
    data.est[name].mean.push(...rows.mean);
    data.est[name].var.push(...rows.var);
  }
  if (chunk.current) data.current = chunk.current;
}

function chartCard(container, opts, tall = false) {
  const canvas = el("canvas");
  container.append(el("div", { class: `chart-card ${tall ? "tall" : ""}` }, canvas));
  return new LineChart(canvas, opts);
}

function buildSingleCharts(info) {
  const sim = $("#sim-single");
  const metricsCharts = $("#metrics-charts");
  for (const c of state.charts ?? []) c.destroy();
  sim.replaceChildren();
  metricsCharts.replaceChildren();
  const range = [info.t_start, info.t_end];
  const states = info.state_names.map((name) =>
    chartCard(sim, { title: `${name}: truth, measurements and estimates (±3σ)`, ylabel: name, xlabel: "t [s]", xRange: range }, true));
  const variances = info.state_names.map((name) =>
    chartCard(metricsCharts, { title: `Variance of ${name} (filter uncertainty)`, ylabel: "variance", xlabel: "t [s]", logY: true, xRange: range }));
  state.charts = [...states, ...variances];
  return { states, variances };
}

function drawSingle(charts, info, data) {
  if (!charts) return;
  const text = getComputedStyle(document.documentElement).getPropertyValue("--text").trim() || "#222";
  info.state_names.forEach((_, i) => {
    const series = [];
    for (const name of info.filters) {
      const e = data.est[name];
      const mean = e.mean.map((r) => r[i]);
      const sd = e.var.map((r) => Math.sqrt(Math.max(0, r[i])));
      series.push({ type: "band", x: data.t, lo: mean.map((m, k) => m - 3 * sd[k]), hi: mean.map((m, k) => m + 3 * sd[k]), color: state.colors[name], alpha: 0.15 });
    }
    series.push({ type: "line", x: data.t, y: data.truth.map((r) => (r ? r[i] : null)), color: text, width: 1.6, label: "truth" });
    for (const [sensor, idx] of Object.entries(info.sensor_state)) {
      if (idx === i && data.meas[sensor]) {
        series.push({ type: "points", x: data.meas[sensor].t, y: data.meas[sensor].z, color: "#8a8f99", size: 2.2, label: `${sensor} measurements` });
      }
    }
    for (const name of info.filters) {
      series.push({ type: "line", x: data.t, y: data.est[name].mean.map((r) => r[i]), color: state.colors[name], width: 1.5, label: name });
    }
    charts.states[i].setSeries(series);
    charts.variances[i].setSeries(info.filters.map((name) => ({
      type: "line", x: data.t, y: data.est[name].var.map((r) => r[i]), color: state.colors[name], width: 1.5, label: name,
    })));
  });
  // Metrics tab
  $("#metrics-title").textContent = "Metrics (this run, so far)";
  $("#metrics-hint").textContent =
    "Consistency verdicts compare the run's mean NEES / NIS with a 95% chi-square interval (corrected for time correlation).";
  if (data.metrics) renderTable($("#metrics-table"), data.metrics.header, data.metrics.rows);
  $("#meanvar-title").textContent = "Current estimate: mean and variance";
  $("#meanvar-hint").textContent = "The filters' latest estimate of each state (mean) and their uncertainty about it (variance), next to the truth.";
  if (data.current) {
    const last = data.truth[data.truth.length - 1];
    const header = ["Filter", ...info.state_names.flatMap((n) => [`${n}: mean`, `${n}: variance`, `${n}: error`])];
    const rows = info.filters.map((name) => [
      name,
      ...info.state_names.flatMap((_, i) => {
        const m = data.current[name].mean[i];
        return [formatNumber(m), formatNumber(data.current[name].var[i]), last ? formatNumber(m - last[i]) : "—"];
      }),
    ]);
    if (last) rows.unshift(["truth", ...info.state_names.flatMap((_, i) => [formatNumber(last[i]), "—", "—"])]);
    renderTable($("#meanvar-table"), header, rows);
  }
}

// ------------------------------------------------------------------ Monte Carlo
async function runMonteCarlo(config) {
  const sid = state.session;
  $("#sim-single").hidden = true;
  $("#sim-mc").hidden = false;
  for (const c of state.charts ?? []) c.destroy();
  const nees = new LineChart($("#chart-nees"), { title: "NEES averaged over the runs (95% bounds)", ylabel: "NEES", xlabel: "t [s]", logY: true });
  const nis = new LineChart($("#chart-nis"), { title: "NIS averaged over the runs (95% bounds)", ylabel: "NIS", xlabel: "t [s]", logY: true });
  const metricsCharts = $("#metrics-charts");
  metricsCharts.replaceChildren();
  let boxes = null;
  state.charts = [nees, nis];

  while (state.running && state.session === sid) {
    const chunk = await state.backend.call("advance", sid, JSON.stringify({ budget: 0.08 }));
    if (state.session !== sid) return;
    if (chunk.info && !boxes) {
      const names = [...chunk.info.state_names, "total"];
      boxes = names.map((n) => {
        const canvas = el("canvas");
        metricsCharts.append(el("div", { class: "chart-card" }, canvas));
        return new BoxChart(canvas, { title: `RMSE ${n} — one value per run`, ylabel: "RMSE" });
      });
      state.charts.push(...boxes);
      state.mcInfo = chunk.info;
    }
    if (chunk.runs_done > 0) drawMonteCarlo(chunk, { nees, nis, boxes });
    setProgress(chunk.runs_done / chunk.runs_total, `${chunk.runs_done} / ${chunk.runs_total} runs`);
    setStatus(chunk.done ? "Finished" : `Monte Carlo… ${chunk.runs_done}/${chunk.runs_total}`);
    if (chunk.done) break;
    await nextFrame();
  }
  if (state.session === sid) await stop("Finished");
}

function consistencySeries(block) {
  const series = [];
  const names = Object.keys(block);
  if (!names.length) return series;
  const first = block[names[0]];
  const sameBounds = names.every((n) => block[n].upper.length === first.upper.length && block[n].upper.every((u, k) => u === first.upper[k]));
  if (sameBounds) series.push({ type: "band", x: first.t, lo: first.lower, hi: first.upper, color: "#8a8f99", alpha: 0.22, label: "95% bounds" });
  for (const n of names) {
    const b = block[n];
    if (!sameBounds) {
      series.push({ type: "line", x: b.t, y: b.lower, color: state.colors[n], dash: [3, 3], width: 1 });
      series.push({ type: "line", x: b.t, y: b.upper, color: state.colors[n], dash: [3, 3], width: 1 });
    }
    series.push({ type: b.t.length > 300 ? "line" : "points", x: b.t, y: b.mean, color: state.colors[n], width: 1.4, size: 2.4, label: n });
  }
  return series;
}

function drawMonteCarlo(chunk, charts) {
  const info = state.mcInfo;
  charts.nees.setSeries(consistencySeries(chunk.nees));
  charts.nis.setSeries(consistencySeries(chunk.nis));
  $("#metrics-title").textContent = `Metrics over ${chunk.runs_done} runs`;
  $("#metrics-hint").textContent =
    "RMSE: mean ± standard deviation over runs. ANEES / ANIS: average over runs and time, its 95% interval and verdict, " +
    "and the share of instants whose run-averaged value lies inside its bounds (ideal ≈ 95%).";
  renderTable($("#metrics-table"), chunk.metrics.header, chunk.metrics.rows);
  $("#meanvar-title").textContent = "RMSE across runs: mean and variance";
  $("#meanvar-hint").textContent = "For each filter, the mean and the variance of the per-run RMSE. A low variance means the filter performs consistently from run to run.";
  const names = Object.keys(chunk.rmse_stats);
  const labels = [...info.state_names, "total"];
  const header = ["Filter", ...labels.flatMap((n) => [`RMSE ${n}: mean`, `RMSE ${n}: variance`])];
  const rows = names.map((n) => [n, ...labels.flatMap((_, i) => [formatNumber(chunk.rmse_stats[n].mean[i]), formatNumber(chunk.rmse_stats[n].var[i])])]);
  renderTable($("#meanvar-table"), header, rows);
  if (charts.boxes) {
    labels.forEach((_, i) => {
      charts.boxes[i].setGroups(names.map((n) => ({
        label: n,
        color: state.colors[n],
        values: i < info.state_names.length ? chunk.rmse[n].map((r) => r[i]) : chunk.rmse_total[n],
      })));
    });
  }
}

// ------------------------------------------------------------------ tables
function renderTable(table, header, rows) {
  const verdictCols = header.map((h, i) => (/^(A?NEES|A?NIS)/.test(h) ? i : -1)).filter((i) => i >= 0);
  const thead = el("thead", {}, el("tr", {}, ...header.map((h) => el("th", { text: h }))));
  const tbody = el("tbody");
  for (const row of rows) {
    const tr = el("tr");
    row.forEach((cell, i) => {
      const td = el("td");
      if (i === 0 && state.colors?.[cell]) td.append(el("span", { class: "swatch", style: `background:${state.colors[cell]}` }));
      td.append(document.createTextNode(cell));
      if (verdictCols.includes(i)) {
        if (/optimistic|pessimistic/.test(cell)) td.className = "bad";
        else if (/consistent/.test(cell)) td.className = "good";
      }
      tr.append(td);
    });
    tbody.append(tr);
  }
  table.replaceChildren(thead, tbody);
}

main();
