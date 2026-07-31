/* Cost Lab frontend. Vanilla JS, hand-drawn SVG — no chart library, so the
   page works on a box with no CDN reachable and the interval ribbon (the one
   mark this console is built around) can be drawn exactly as intended.

   The whole analysis is derived client-side from the raw prediction series
   the backend ships, so a checkpoint grid can be re-sliced without a round
   trip. */

const S = {
  repo: null,
  problems: [],
  experiments: [],
  selected: null,      // experiment name
  detail: null,        // full payload for `selected`
  metric: "duration",
  picked: new Set(),
  timer: null,
};

const GRID = Array.from({ length: 20 }, (_, i) => (i + 1) * 5);       // 5%…100%
const CHECKPOINTS = [5, 10, 15, 25, 50, 75, 100];
const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

const METRIC = {
  duration: { pred: "predicted_duration_s", lo: "ci_low_s", hi: "ci_high_s",
              actual: "actual_duration", unit: "s", label: "duration" },
  tokens:   { pred: "predicted_tokens", lo: "token_ci_low", hi: "token_ci_high",
              actual: "actual_tokens", unit: "tok", label: "tokens" },
};

/* ─────────────────────────────────────────────────────────── utils ── */

const fmt = (v, unit) => {
  if (v == null) return "—";
  if (unit === "s") return v >= 3600 ? `${(v / 3600).toFixed(1)}h`
    : v >= 60 ? `${(v / 60).toFixed(1)}m` : `${Math.round(v)}s`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  return String(Math.round(v));
};
const pct = (v) => (v == null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`);
const median = (xs) => {
  if (!xs.length) return null;
  const a = [...xs].sort((x, y) => x - y), m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
};
const quantile = (xs, q) => {
  if (!xs.length) return null;
  const a = [...xs].sort((x, y) => x - y), i = (a.length - 1) * q;
  const lo = Math.floor(i), hi = Math.ceil(i);
  return a[lo] + (a[hi] - a[lo]) * (i - lo);
};
const ago = (ts) => {
  const d = Date.now() / 1000 - ts;
  if (d < 90) return "just now";
  if (d < 5400) return `${Math.round(d / 60)} min ago`;
  if (d < 172800) return `${Math.round(d / 3600)} h ago`;
  return `${Math.round(d / 86400)} d ago`;
};

/** Signed % error of `metric` at run progress `p` (percent of the run's own
 *  prediction points), so two conditions that emitted different numbers of
 *  points are still compared at the same stage of the run. */
function errAt(run, p, metric) {
  const m = METRIC[metric], n = run.series.length, actual = run[m.actual];
  if (!n || !actual) return null;
  const i = Math.min(n - 1, Math.max(0, Math.round((p / 100) * n) - 1));
  return ((run.series[i][m.pred] - actual) / actual) * 100;
}

const usable = (run) => run.series.length > 0 && run[METRIC[S.metric].actual];

function byTask(runs) {
  const map = new Map();
  for (const r of runs) {
    if (!map.has(r.key)) map.set(r.key, {});
    map.get(r.key)[r.condition] = r;
  }
  return map;
}

/* ────────────────────────────────────────────────── svg primitives ── */

const NS = "http://www.w3.org/2000/svg";
const svgEl = (tag, attrs = {}) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
};

function chartFrame(w, h, pad) {
  const svg = svgEl("svg", { viewBox: `0 0 ${w} ${h}`, class: "chart",
                             preserveAspectRatio: "xMidYMid meet" });
  const x = (v, d0, d1) => pad.l + ((v - d0) / (d1 - d0 || 1)) * (w - pad.l - pad.r);
  const y = (v, d0, d1) => h - pad.b - ((v - d0) / (d1 - d0 || 1)) * (h - pad.t - pad.b);
  return { svg, x, y, w, h, pad };
}

function yAxis(f, d0, d1, ticks, label) {
  for (const t of ticks) {
    const yy = f.y(t, d0, d1);
    f.svg.appendChild(svgEl("line", { class: "grid", x1: f.pad.l, x2: f.w - f.pad.r, y1: yy, y2: yy }));
    const tx = svgEl("text", { class: "tick", x: f.pad.l - 8, y: yy + 3.5, "text-anchor": "end" });
    tx.textContent = t;
    f.svg.appendChild(tx);
  }
  if (label) {
    // anchored at the plot's left edge, not outside it — an end-anchored label
    // longer than the gutter gets clipped by the viewBox
    const t = svgEl("text", { class: "axlabel", x: f.pad.l, y: f.pad.t - 10, "text-anchor": "start" });
    t.textContent = label;
    f.svg.appendChild(t);
  }
}

function xAxis(f, d0, d1, ticks, suffix = "") {
  const yy = f.h - f.pad.b;
  f.svg.appendChild(svgEl("line", { class: "axis", x1: f.pad.l, x2: f.w - f.pad.r, y1: yy, y2: yy }));
  for (const t of ticks) {
    const xx = f.x(t, d0, d1);
    const tx = svgEl("text", { class: "tick", x: xx, y: yy + 15, "text-anchor": "middle" });
    tx.textContent = t + suffix;
    f.svg.appendChild(tx);
  }
}

const linePath = (pts) => pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("");
const areaPath = (top, bot) =>
  linePath(top) + "L" + [...bot].reverse().map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("L") + "Z";

/* tooltip */
const tip = $("#tooltip");
function bindTip(node, html) {
  node.addEventListener("mouseenter", (e) => {
    tip.innerHTML = html;
    tip.hidden = false;
    moveTip(e);
  });
  node.addEventListener("mousemove", moveTip);
  node.addEventListener("mouseleave", () => { tip.hidden = true; });
}
function moveTip(e) {
  const pad = 14;
  let x = e.clientX + pad, y = e.clientY + pad;
  const r = tip.getBoundingClientRect();
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = e.clientY - r.height - pad;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}

function toast(msg, bad = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.toggle("is-bad", bad);
  t.hidden = false;
  clearTimeout(t._t);
  t._t = setTimeout(() => { t.hidden = true; }, 5200);
}

/* ───────────────────────────────────────────────────────── api ── */

/* Paths are written with a leading slash for readability but requested
   relative to the page, so the same build works served at the root and
   behind nginx's `location /costlab/` (which strips the prefix). Relies on
   the page URL ending in a slash — nginx redirects /costlab to /costlab/. */
const api = async (path, opts) => {
  const r = await fetch(path.replace(/^\//, ""), opts);
  if (!r.ok) throw new Error((await r.text()).slice(0, 300));
  return r.json();
};
const post = (path, body) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" },
              body: body ? JSON.stringify(body) : undefined });

/* ────────────────────────────────────────────────────── topbar ── */

function renderRepo() {
  const r = S.repo, box = $("#repo");
  box.textContent = "";
  if (!r) return;
  const branch = el("span", "chip");
  branch.append(el("span", null, r.branch));
  box.append(branch);

  const head = el("span", "chip");
  head.append(el("span", null, r.head), el("span", "chip__subject", r.subject));
  box.append(head);

  if (r.behind > 0) {
    box.append(el("span", "chip chip--behind", `${r.behind} behind origin`));
  }
  if (r.dirty) box.append(el("span", "chip chip--warn", "uncommitted changes"));

  const btn = el("button", "ghost", r.behind > 0 ? `Pull ${r.behind}` : "Sync code");
  btn.onclick = async () => {
    btn.disabled = true;
    btn.textContent = "Pulling…";
    try {
      const res = await post("/api/repo/pull");
      S.repo = res.repo;
      renderRepo();
      toast(res.ok ? `Now on ${res.repo.head} — ${res.repo.subject}`
                   : `Pull failed: ${res.output.split("\n").slice(-2).join(" ")}`, !res.ok);
      loadProblems();
    } catch (e) { toast(String(e.message), true); btn.disabled = false; btn.textContent = "Sync code"; }
  };
  box.append(btn);
}

/* ────────────────────────────────────────────────────── launch ── */

function renderTasks() {
  const groups = new Map();
  for (const p of S.problems) {
    if (!groups.has(p.family)) groups.set(p.family, []);
    groups.get(p.family).push(p);
  }
  // alphaevolve first: it's the benchmark family this console was built for
  const order = [...groups.keys()].sort((a, b) =>
    a === "alphaevolve" ? -1 : b === "alphaevolve" ? 1 : a.localeCompare(b));

  const bar = $("#familybar");
  bar.textContent = "";
  for (const fam of order) {
    const b = el("button", null, `${fam} · ${groups.get(fam).length}`);
    b.onclick = () => {
      const names = groups.get(fam).map((p) => p.name);
      const allOn = names.every((n) => S.picked.has(n));
      names.forEach((n) => (allOn ? S.picked.delete(n) : S.picked.add(n)));
      renderTasks();
    };
    bar.append(b);
  }
  const clear = el("button", null, "clear");
  clear.onclick = () => { S.picked.clear(); renderTasks(); };
  bar.append(clear);

  const list = $("#tasklist");
  list.textContent = "";
  for (const fam of order) {
    const g = el("div", "taskgroup");
    g.append(el("div", "taskgroup__label", fam));
    for (const p of groups.get(fam)) {
      const row = el("label", "task");
      const cb = el("input");
      cb.type = "checkbox";
      cb.checked = S.picked.has(p.name);
      cb.onchange = () => {
        cb.checked ? S.picked.add(p.name) : S.picked.delete(p.name);
        renderLedger();
      };
      row.append(cb, el("span", "task__name", p.label),
                 el("span", "task__seeds", `${p.seeds} seed${p.seeds === 1 ? "" : "s"}`));
      g.append(row);
    }
    list.append(g);
  }
  renderLedger();
}

function renderLedger() {
  const n = S.picked.size;
  const wave = +$("#wave-size").value;
  const waves = Math.ceil(n / wave) || 0;
  const led = $("#ledger");
  led.innerHTML = n === 0
    ? "Pick at least one task. Each one runs twice — agentless, then with the agent."
    : `<b>${n}</b> task${n === 1 ? "" : "s"} → <b>${n * 2}</b> runs<br>` +
      `<b>${waves}</b> wave${waves === 1 ? "" : "s"} of up to <b>${wave * 2}</b> concurrent runs<br>` +
      `<b>${+$("#max-mutants").value}</b> attempts each`;
  $("#start").disabled = n === 0;
}

async function launch() {
  const btn = $("#start");
  btn.disabled = true;
  btn.textContent = "Starting…";
  try {
    const res = await post("/api/experiments", {
      tasks: [...S.picked],
      max_mutants: +$("#max-mutants").value,
      wave_size: +$("#wave-size").value,
      llm: $("#llm").value,
      label: $("#label").value,
    });
    $("#launch-note").textContent = res.cmd;
    toast(`Launched ${res.name}`);
    await loadExperiments();
    select(res.name);
    goto("live");
  } catch (e) {
    toast(`Could not start: ${e.message}`, true);
  }
  btn.textContent = "Start experiment";
  renderLedger();
}

/* ────────────────────────────────────────────────────── rail ── */

function renderExperiments() {
  const list = $("#explist");
  list.textContent = "";
  if (!S.experiments.length) {
    list.append(el("li", "explist__empty", "No experiments yet. Launch one."));
    return;
  }
  for (const e of S.experiments) {
    const li = el("li");
    const b = el("button", "exp");
    b.setAttribute("aria-current", String(e.name === S.selected));
    b.append(el("div", "exp__name", e.name));
    const meta = el("div", "exp__meta");
    if (e.running > 0 || e.launching) meta.append(el("span", "pulse"));
    meta.append(el("span", null,
      `${e.done}/${e.n_runs} done · ${e.tasks.length} task${e.tasks.length === 1 ? "" : "s"} · ${ago(e.created)}`));
    b.append(meta);
    b.onclick = () => { select(e.name); goto(e.running > 0 ? "live" : "analyze"); };
    li.append(b);
    list.append(li);
  }
}

/* ────────────────────────────────────────────────────── live ── */

/** The signature mark, in miniature: the prediction interval as a ribbon,
 *  with the run's own elapsed time drawn through it. When the ribbon stops
 *  straddling the elapsed line, the forecast has gone wrong. */
function ribbon(run) {
  const m = METRIC[S.metric], s = run.series;
  const f = chartFrame(340, 74, { l: 4, r: 4, t: 6, b: 6 });
  if (s.length < 2) return f.svg;
  const lo = s.map((p) => p[m.lo]), hi = s.map((p) => p[m.hi]), pr = s.map((p) => p[m.pred]);
  const actual = run[m.actual];
  let d0 = Math.min(...lo, ...pr), d1 = Math.max(...hi, ...pr);
  if (actual) { d0 = Math.min(d0, actual); d1 = Math.max(d1, actual); }
  const X = (i) => f.x(i, 0, s.length - 1), Y = (v) => f.y(v, d0, d1);

  f.svg.appendChild(svgEl("path", {
    class: "band", fill: run.condition === "withagent" ? "var(--agent)" : "var(--auto)",
    d: areaPath(s.map((p, i) => [X(i), Y(p[m.hi])]), s.map((p, i) => [X(i), Y(p[m.lo])])),
  }));
  f.svg.appendChild(svgEl("path", {
    class: "line", stroke: run.condition === "withagent" ? "var(--agent)" : "var(--auto)",
    d: linePath(s.map((p, i) => [X(i), Y(p[m.pred])])),
  }));
  if (actual) {
    f.svg.appendChild(svgEl("line", {
      x1: f.pad.l, x2: f.w - f.pad.r, y1: Y(actual), y2: Y(actual),
      stroke: "var(--ink)", "stroke-width": 1.5, "stroke-dasharray": "3 3", opacity: 0.75,
    }));
  }
  // notches: where the agent woke up
  s.forEach((p, i) => {
    if (!p.agent_due) return;
    f.svg.appendChild(svgEl("line", {
      x1: X(i), x2: X(i), y1: 4, y2: f.h - 4,
      stroke: "var(--wake)", "stroke-width": 1, opacity: 0.55,
    }));
  });
  return f.svg;
}

function renderLive() {
  const d = S.detail;
  $("#live-eyebrow").textContent = d ? d.name : "No experiment selected";
  $("#live-title").textContent = d ? "Runs in flight" : "Live";
  const box = $("#runs");
  box.textContent = "";
  if (!d) { box.append(el("p", "empty", "Pick an experiment on the left, or launch a new one.")); return; }

  for (const run of d.runs) {
    const m = METRIC[S.metric];
    const card = el("div", "runcard");
    const top = el("div", "runcard__top");
    top.append(el("div", "runcard__task", run.task),
               el("span", `runcard__cond cond--${run.condition}`,
                  run.condition === "withagent" ? "with agent" : "agentless"));
    card.append(top);

    const last = run.series[run.series.length - 1];
    const stats = el("div", "runcard__stats");
    const stat = (k, v, sub) => {
      const s = el("div", "stat");
      s.append(el("div", "stat__k", k));
      const val = el("div", "stat__v", v);
      if (sub) { const sm = el("small", null, ` ${sub}`); val.append(sm); }
      s.append(val);
      return s;
    };
    stats.append(stat("attempts", String(run.attempts ?? 0)));
    stats.append(stat("forecast", last ? fmt(last[m.pred], m.unit) : "—"));
    stats.append(stat("actual", run[m.actual] ? fmt(run[m.actual], m.unit) : "—"));
    if (run[m.actual] && last) {
      const e = ((last[m.pred] - run[m.actual]) / run[m.actual]) * 100;
      const s = stat("error", pct(e));
      s.querySelector(".stat__v").classList.add(Math.abs(e) <= 10 ? "win" : "loss");
      stats.append(s);
    }
    if (run.agent_events.length) {
      stats.append(stat("wakeups", String(run.agent_events.length),
                        `${run.agent_events.filter((e) => e.moved.length).length} acted`));
    }
    card.append(stats);
    card.append(ribbon(run));

    const st = el("div", "status");
    st.append(el("span", `dot dot--${run.status}`), el("b", null, run.status));
    if (last?.concurrency) st.append(el("span", null, `· concurrency ${last.concurrency.toFixed(1)}`));
    st.append(el("span", null, `· db ${run.db}`));
    const logBtn = el("button", "ghost", "log");
    logBtn.style.cssText = "margin-left:auto;padding:2px 8px;font-size:11px";
    logBtn.onclick = () => showLog(run.stem);
    st.append(logBtn);
    card.append(st);
    box.append(card);
  }
}

async function showLog(stem) {
  try {
    const r = await api(`/api/experiments/${S.selected}/log/${stem}?tail=200`);
    const w = window.open("", "_blank");
    w.document.write(
      `<title>${stem}</title><body style="background:#0B0E13;color:#A8B3C2;font:12px ui-monospace,monospace;padding:16px;white-space:pre-wrap">`
      + r.lines.join("\n").replace(/[<&]/g, (c) => (c === "<" ? "&lt;" : "&amp;")));
  } catch (e) { toast(e.message, true); }
}

/* ──────────────────────────────────────────────────── analyze ── */

function renderAnalyze() {
  const box = $("#analysis");
  box.textContent = "";
  const d = S.detail;
  $("#an-eyebrow").textContent = d ? d.name : "No experiment selected";
  $("#an-title").textContent = "Automatic estimator vs estimator + agent";
  if (!d) { box.append(el("p", "empty", "Pick an experiment on the left.")); return; }

  const done = d.runs.filter(usable);
  if (!done.length) {
    box.append(el("p", "empty",
      "No finished run has both a prediction series and a measured total yet. Come back when a wave lands."));
    return;
  }
  const tasks = byTask(done);
  const paired = [...tasks.entries()].filter(([, c]) => c.noagent && c.withagent);

  box.append(headline(paired, tasks));
  box.append(checkpointTable(paired));
  box.append(agentBlock(d.runs));
  box.append(smallMultiples(paired));
  box.append(finalTable(tasks));
}

function condSeries(paired, cond) {
  // median and IQR of |error| across tasks, at every 5% checkpoint
  return GRID.map((p) => {
    const vals = paired.map(([, c]) => errAt(c[cond], p, S.metric))
                       .filter((v) => v != null).map(Math.abs);
    return { p, med: median(vals), q1: quantile(vals, 0.25), q3: quantile(vals, 0.75), n: vals.length };
  });
}

function headline(paired, tasks) {
  const block = el("div", "block");
  const head = el("div", "block__head");
  head.append(el("h3", null, `Forecast error as the run proceeds`));
  head.append(el("p", null,
    paired.length
      ? `Median absolute ${METRIC[S.metric].label} error across ${paired.length} paired task${paired.length === 1 ? "" : "s"}, sampled at every 5% of run progress. The shaded arm is the interquartile range — how much the tasks disagree.`
      : "No task has both conditions finished yet, so there is nothing to pair. Showing what is available below."));
  block.append(head);

  if (!paired.length) {
    block.append(el("p", "empty", `${tasks.size} task(s) present, none with both conditions complete.`));
    return block;
  }

  const A = condSeries(paired, "noagent"), B = condSeries(paired, "withagent");
  const w = 980, h = 380, f = chartFrame(w, h, { l: 54, r: 158, t: 30, b: 40 });
  const all = [...A, ...B].flatMap((d) => [d.q3, d.med]).filter((v) => v != null);
  const top = Math.max(20, Math.ceil(Math.max(...all) / 10) * 10);
  const ticks = Array.from({ length: 6 }, (_, i) => Math.round((top / 5) * i));
  yAxis(f, 0, top, ticks, `|${METRIC[S.metric].label} error| %`);
  xAxis(f, 5, 100, [5, 10, 15, 25, 50, 75, 100], "%");

  const X = (p) => f.x(p, 5, 100), Y = (v) => f.y(Math.min(v, top), 0, top);

  const placed = [];  // y positions already used by a direct label
  for (const [data, color, name, dash] of [
    [A, "var(--auto)", "estimator alone", "5 4"],
    [B, "var(--agent)", "estimator + agent", null],
  ]) {
    const pts = data.filter((d) => d.med != null);
    if (!pts.length) continue;
    f.svg.appendChild(svgEl("path", {
      class: "band", fill: color,
      d: areaPath(pts.map((d) => [X(d.p), Y(d.q3)]), pts.map((d) => [X(d.p), Y(d.q1)])),
    }));
    const line = svgEl("path", { class: "line", stroke: color, d: linePath(pts.map((d) => [X(d.p), Y(d.med)])) });
    if (dash) line.setAttribute("stroke-dasharray", dash);
    f.svg.appendChild(line);

    pts.forEach((d) => {
      const c = svgEl("circle", { cx: X(d.p), cy: Y(d.med), r: 4.5, fill: color,
                                  stroke: "var(--surface)", "stroke-width": 2 });
      bindTip(c, `<b>${d.p}% of the run</b><br>${name}<br>median |error| <em>${d.med.toFixed(1)}%</em><br>IQR ${d.q1.toFixed(1)}–${d.q3.toFixed(1)}% · n=${d.n}`);
      f.svg.appendChild(c);
    });

    // direct label at the curve's end, nudged clear of one already placed
    const lastPt = pts[pts.length - 1];
    let ly = Y(lastPt.med) + 4;
    while (placed.some((p) => Math.abs(p - ly) < 15)) ly += 15;
    placed.push(ly);
    const lbl = svgEl("text", { class: "dlabel", fill: color, x: X(lastPt.p) + 10, y: ly });
    lbl.textContent = name;
    f.svg.appendChild(lbl);
  }

  // the 10% band every forecast is trying to get inside
  const y10 = Y(10);
  if (y10 > f.pad.t) {
    f.svg.appendChild(svgEl("line", { x1: f.pad.l, x2: w - f.pad.r, y1: y10, y2: y10,
                                       stroke: "var(--ink-3)", "stroke-width": 1, "stroke-dasharray": "2 4" }));
    const t = svgEl("text", { class: "tick", x: w - f.pad.r - 2, y: y10 - 6, "text-anchor": "end" });
    t.textContent = "10% — usable forecast";
    f.svg.appendChild(t);
  }

  const wrap = el("div", "chartwrap");
  wrap.append(f.svg);
  block.append(wrap);
  return block;
}

function checkpointTable(paired) {
  const block = el("div", "block");
  const head = el("div", "block__head");
  head.append(el("h3", null, "Checkpoint ledger"));
  head.append(el("p", null,
    "Median absolute error at the checkpoints a budget decision actually gets made at. Delta is agent minus estimator: negative means the agent helped."));
  block.append(head);

  const A = condSeries(paired, "noagent"), B = condSeries(paired, "withagent");
  const at = (arr, p) => arr.find((d) => d.p === p);

  const t = el("table", "grid-table");
  const thead = el("thead"), hr = el("tr");
  hr.append(el("th", null, "Condition"));
  CHECKPOINTS.forEach((p) => hr.append(el("th", null, `${p}%`)));
  thead.append(hr);
  t.append(thead);

  const tb = el("tbody");
  const row = (label, arr) => {
    const tr = el("tr");
    tr.append(el("td", null, label));
    CHECKPOINTS.forEach((p) => {
      const d = at(arr, p);
      tr.append(el("td", null, d?.med == null ? "—" : `${d.med.toFixed(1)}`));
    });
    tb.append(tr);
  };
  row("estimator alone", A);
  row("estimator + agent", B);

  const tr = el("tr");
  tr.append(el("td", null, "delta"));
  CHECKPOINTS.forEach((p) => {
    const a = at(A, p)?.med, b = at(B, p)?.med;
    const td = el("td", null, a == null || b == null ? "—" : pct(b - a));
    if (a != null && b != null) td.classList.add(b < a ? "win" : "loss");
    tr.append(td);
  });
  tb.append(tr);
  t.append(tb);
  block.append(t);
  return block;
}

function agentBlock(runs) {
  const block = el("div", "block");
  const head = el("div", "block__head");
  head.append(el("h3", null, "What the agent actually did"));
  head.append(el("p", null,
    "Every wakeup, on the run's own attempt axis. A filled notch moved at least one lever; a hollow one means the agent looked and declined. Hover for the trigger and its reasoning."));
  block.append(head);

  const withAgent = runs.filter((r) => r.condition === "withagent");
  if (!withAgent.some((r) => r.agent_events.length)) {
    block.append(el("p", "empty", "No wakeups recorded yet."));
    return block;
  }

  const legend = el("div", "legend");
  legend.innerHTML =
    `<span><i class="swatch" style="background:var(--wake)"></i>moved a lever</span>` +
    `<span><i class="swatch" style="background:var(--ink-3)"></i>woke, changed nothing</span>` +
    `<span><i class="swatch" style="background:var(--agent)"></i>flagged an outlier bucket</span>`;
  block.append(legend);

  const maxAtt = Math.max(...withAgent.map((r) => r.attempts || 1), 1);
  for (const run of withAgent) {
    const lane = el("div", "lane");
    lane.append(el("div", "lane__t", run.task));

    const f = chartFrame(760, 34, { l: 2, r: 2, t: 2, b: 2 });
    f.svg.appendChild(svgEl("line", { x1: 2, x2: 758, y1: 17, y2: 17, stroke: "var(--rule)", "stroke-width": 6 }));
    for (const ev of run.agent_events) {
      const x = f.x(ev.attempt, 0, maxAtt);
      const moved = ev.moved.length > 0;
      const mark = svgEl("line", {
        x1: x, x2: x, y1: 5, y2: 29,
        stroke: ev.outliers > 0 ? "var(--agent)" : moved ? "var(--wake)" : "var(--ink-3)",
        "stroke-width": moved || ev.outliers ? 3 : 1.5,
        opacity: moved || ev.outliers ? 1 : 0.6,
      });
      const levers = ev.moved.length
        ? ev.moved.map((k) => `${k} → ${ev.levers[k]}`).join("<br>")
        : "<em>no lever moved</em>";
      bindTip(mark, `<b>attempt ${ev.attempt}</b>${ev.ts ? ` <em>${ev.ts}</em>` : ""}<br>` +
                    `${levers}${ev.outliers ? `<br>outliers flagged: ${ev.outliers}` : ""}<br>` +
                    `<em>trigger</em> ${ev.trigger || "—"}` +
                    (ev.reason ? `<br><em>reason</em> ${ev.reason.slice(0, 260)}` : ""));
      f.svg.appendChild(mark);
    }
    lane.append(f.svg);
    const acted = run.agent_events.filter((e) => e.moved.length || e.outliers).length;
    lane.append(el("div", "lane__n", `${acted}/${run.agent_events.length} acted`));
    block.append(lane);
  }
  return block;
}

function smallMultiples(paired) {
  const block = el("div", "block");
  const head = el("div", "block__head");
  head.append(el("h3", null, "Task by task"));
  head.append(el("p", null, "The same error curve per task, so a single win or blow-up cannot hide inside the median."));
  block.append(head);

  const legend = el("div", "legend");
  legend.innerHTML =
    `<span style="color:var(--auto)"><i class="swatch swatch--dash"></i>estimator alone</span>` +
    `<span><i class="swatch" style="background:var(--agent)"></i>estimator + agent</span>`;
  block.append(legend);

  const grid = el("div", "smalls");
  for (const [key, c] of paired) {
    const cell = el("div", "small");
    cell.append(el("div", "small__t", key));
    const f = chartFrame(300, 128, { l: 30, r: 8, t: 10, b: 20 });
    const curves = ["noagent", "withagent"].map((cond) =>
      GRID.map((p) => ({ p, v: Math.abs(errAt(c[cond], p, S.metric) ?? 0) })));
    const top = Math.max(20, Math.ceil(Math.max(...curves.flat().map((d) => d.v)) / 10) * 10);
    yAxis(f, 0, top, [0, Math.round(top / 2), top], null);
    xAxis(f, 5, 100, [5, 50, 100], "");
    curves.forEach((pts, i) => {
      const path = svgEl("path", {
        class: "line", "stroke-width": 1.8,
        stroke: i ? "var(--agent)" : "var(--auto)",
        d: linePath(pts.map((d) => [f.x(d.p, 5, 100), f.y(Math.min(d.v, top), 0, top)])),
      });
      if (!i) path.setAttribute("stroke-dasharray", "4 3");
      f.svg.appendChild(path);
    });
    cell.append(f.svg);
    const fin = ["noagent", "withagent"].map((cond) => errAt(c[cond], 100, S.metric));
    const d = el("div", "small__d");
    d.innerHTML = `final <span style="color:var(--auto)">${pct(fin[0])}</span> → <span style="color:var(--agent)">${pct(fin[1])}</span>`;
    cell.append(d);
    grid.append(cell);
  }
  block.append(grid);
  return block;
}

function finalTable(tasks) {
  const block = el("div", "block");
  const head = el("div", "block__head");
  head.append(el("h3", null, "Final forecast vs measured"));
  head.append(el("p", null, "The last prediction each run published, against what the run actually cost."));
  block.append(head);

  const m = METRIC[S.metric];
  const t = el("table", "grid-table");
  t.innerHTML = `<thead><tr><th>Task</th><th>Condition</th><th>Forecast</th><th>Measured</th><th>Error</th><th>Wakeups</th><th>Points</th></tr></thead>`;
  const tb = el("tbody");
  for (const [key, c] of tasks) {
    let first = true;
    for (const cond of ["noagent", "withagent"]) {
      const run = c[cond];
      if (!run) continue;
      const last = run.series[run.series.length - 1];
      const actual = run[m.actual];
      const err = last && actual ? ((last[m.pred] - actual) / actual) * 100 : null;
      const tr = el("tr");
      tr.append(el("td", null, first ? key : ""));
      tr.append(el("td", null, cond === "withagent" ? "with agent" : "estimator"));
      tr.append(el("td", null, last ? fmt(last[m.pred], m.unit) : "—"));
      tr.append(el("td", null, fmt(actual, m.unit)));
      const e = el("td", null, pct(err));
      if (err != null) e.classList.add(Math.abs(err) <= 10 ? "win" : "loss");
      tr.append(e);
      tr.append(el("td", null, run.agent_events.length ? String(run.agent_events.length) : "—"));
      tr.append(el("td", null, String(run.series.length)));
      tb.append(tr);
      first = false;
    }
  }
  t.append(tb);
  block.append(t);
  return block;
}

/* ───────────────────────────────────────────────── navigation ── */

function goto(view) {
  document.documentElement.dataset.view = view;
  document.querySelectorAll(".views button").forEach((b) =>
    b.setAttribute("aria-selected", String(b.dataset.goto === view)));
  if (view === "analyze") renderAnalyze();
  if (view === "live") renderLive();
}

async function select(name) {
  S.selected = name;
  renderExperiments();
  await loadDetail();
}

async function loadDetail() {
  if (!S.selected) return;
  try {
    S.detail = await api(`/api/experiments/${S.selected}`);
    if (document.documentElement.dataset.view === "live") renderLive();
    if (document.documentElement.dataset.view === "analyze") renderAnalyze();
  } catch (e) { toast(e.message, true); }
}

async function loadExperiments() {
  S.experiments = await api("/api/experiments");
  renderExperiments();
}

async function loadProblems() {
  S.problems = await api("/api/problems");
  renderTasks();
}

function tick() {
  clearInterval(S.timer);
  S.timer = setInterval(async () => {
    if (!$("#autorefresh").checked) return;
    await loadExperiments();
    if (S.selected) await loadDetail();
  }, 6000);
}

/* ─────────────────────────────────────────────────────── boot ── */

document.querySelectorAll("[data-goto]").forEach((b) => {
  b.onclick = (e) => { e.preventDefault(); goto(b.dataset.goto); };
});

$("#max-mutants").oninput = (e) => { $("#max-mutants-out").textContent = e.target.value; renderLedger(); };
$("#wave-size").oninput = (e) => { $("#wave-size-out").textContent = e.target.value; renderLedger(); };
$("#start").onclick = launch;
$("#refresh-experiments").onclick = loadExperiments;

$("#metric-switch").onclick = (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  S.metric = b.dataset.metric;
  document.querySelectorAll("#metric-switch button").forEach((x) => x.classList.toggle("is-on", x === b));
  renderAnalyze();
  renderLive();
};

$("#stop-exp").onclick = async () => {
  if (!S.selected) return toast("No experiment selected", true);
  await post(`/api/experiments/${S.selected}/stop`);
  toast(`Stop signal sent to ${S.selected}`);
  loadDetail();
};

$("#build-report").onclick = async () => {
  if (!S.selected) return toast("No experiment selected", true);
  const b = $("#build-report");
  b.disabled = true; b.textContent = "Building…";
  try {
    const r = await post(`/api/experiments/${S.selected}/report`);
    toast(r.ok ? `Report written: ${r.files.join(", ")}` : `Report failed — ${r.output.slice(-200)}`, !r.ok);
  } catch (e) { toast(e.message, true); }
  b.disabled = false; b.textContent = "Build PNG report";
};

(async function boot() {
  try {
    S.repo = await api("/api/repo");
    renderRepo();
    await loadProblems();
    await loadExperiments();
    if (S.experiments.length) await select(S.experiments[0].name);
    tick();
  } catch (e) { toast(`Backend unreachable: ${e.message}`, true); }
})();
