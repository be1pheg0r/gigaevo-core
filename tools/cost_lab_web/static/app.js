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
  taskFilter: "__all__",   // "__all__" or a task key
  picked: new Set(),
  timer: null,
  launchTimer: null,
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
              actual: "actual_duration", unit: "s", label: "времени", noun: "Время" },
  tokens:   { pred: "predicted_tokens", lo: "token_ci_low", hi: "token_ci_high",
              actual: "actual_tokens", unit: "tok", label: "токенам", noun: "Токены" },
};

const COND_RU = { noagent: "автомат", withagent: "автомат + агент" };
const LEVER_RU = {
  cold_start_factor: "cold_start", golden_ratio: "golden_ratio",
  growth_rate_mult: "growth_rate", concurrency_mult: "concurrency",
  cold_start: "cold_start", growth_rate: "growth_rate", concurrency: "concurrency",
};
const TOOL_RU = {
  get_trigger: "get_trigger — что разбудило",
  get_progress: "get_progress — сколько пройдено",
  get_backpressure: "get_backpressure — очередь и слоты",
  get_model_params: "get_model_params — текущие коэффициенты",
  get_last_adjustment_outcome: "get_last_adjustment_outcome — чем кончилась прошлая правка",
  get_recent_calls: "get_recent_calls — последние вызовы LLM",
};
const ACTION_RU = {
  adjust_model: "adjust_model",
  flag_as_outlier: "flag_as_outlier",
  skip_next_calibration: "skip_next_calibration",
};

/* ─────────────────────────────────────────────────────────── utils ── */

const fmt = (v, unit) => {
  if (v == null) return "—";
  if (unit === "s") return v >= 3600 ? `${(v / 3600).toFixed(1)}ч`
    : v >= 60 ? `${(v / 60).toFixed(1)}м` : `${Math.round(v)}с`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(2)}М`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}тыс`;
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
  if (d < 90) return "только что";
  if (d < 5400) return `${Math.round(d / 60)} мин назад`;
  if (d < 172800) return `${Math.round(d / 3600)} ч назад`;
  return `${Math.round(d / 86400)} дн назад`;
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
    box.append(el("span", "chip chip--behind", `отстаёт на ${r.behind}`));
  }
  if (r.dirty) box.append(el("span", "chip chip--warn", "есть незакоммиченное"));

  const btn = el("button", "ghost", r.behind > 0 ? `Подтянуть ${r.behind}` : "Обновить код");
  btn.onclick = async () => {
    btn.disabled = true;
    btn.textContent = "Тяну…";
    try {
      const res = await post("/api/repo/pull");
      S.repo = res.repo;
      renderRepo();
      toast(res.ok ? `Теперь на ${res.repo.head} — ${res.repo.subject}`
                   : `Не удалось подтянуть: ${res.output.split("\n").slice(-2).join(" ")}`, !res.ok);
      loadProblems();
    } catch (e) { toast(String(e.message), true); btn.disabled = false; btn.textContent = "Обновить код"; }
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
  const clear = el("button", null, "снять всё");
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
      row.append(cb, el("span", "task__name", p.label));
      if (p.missing_deps && p.missing_deps.length) {
        // The trap prompts/sudoku fell into: the run starts, then every mutant
        // dies in the validator because this box has no such module.
        row.classList.add("task--blocked");
        const w = el("span", "task__warn", `нет ${p.missing_deps.slice(0, 2).join(", ")}`);
        w.title = `Задача импортирует модули, которых нет в окружении: ${p.missing_deps.join(", ")}. Прогон стартует, но мутанты будут падать на валидации.`;
        row.append(w);
      } else {
        row.append(el("span", "task__seeds", `сидов: ${p.seeds}`));
      }
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
  const blocked = [...S.picked].filter((t) => (S.problems.find((x) => x.name === t)?.missing_deps || []).length);
  led.innerHTML = n === 0
    ? "Выберите хотя бы одну задачу. Каждая пойдёт дважды — без агента и с агентом."
    : `задач: <b>${n}</b> → прогонов: <b>${n * 2}</b><br>` +
      `волн: <b>${waves}</b>, одновременно до <b>${wave * 2}</b> прогонов<br>` +
      `попыток в каждом: <b>${+$("#max-mutants").value}</b>` +
      (blocked.length
        ? `<br><span style="color:var(--crit)">задачам без зависимостей: ${blocked.length} — прогон стартует, но мутанты будут падать</span>`
        : "");
  $("#start").disabled = n === 0;
}

async function launch() {
  const btn = $("#start");
  btn.disabled = true;
  btn.textContent = "Запускаю…";
  try {
    const res = await post("/api/experiments", {
      tasks: [...S.picked],
      max_mutants: +$("#max-mutants").value,
      wave_size: +$("#wave-size").value,
      llm: $("#llm").value,
      label: $("#label").value,
    });
    $("#launch-note").textContent = res.cmd;
    toast(`Запущено: ${res.name}`);
    // Switch to the live view FIRST, so the driver panel is on screen while
    // the launcher is still picking redis dbs — the wait is the part that
    // used to look like nothing happening.
    S.selected = res.name;
    goto("live");
    await select(res.name);
    await loadExperiments();
    watchLaunch(res.name);
  } catch (e) {
    toast(`Не удалось запустить: ${e.message}`, true);
  }
  btn.textContent = "Запустить эксперимент";
  renderLedger();
}

/** Poll hard for the first two minutes of a launch, then hand back to the
 *  normal 6s tick. Also reports a driver that died, once. */
function watchLaunch(name) {
  clearInterval(S.launchTimer);
  const until = Date.now() + 120000;
  let warned = false;
  S.launchTimer = setInterval(async () => {
    if (S.selected !== name || Date.now() > until) return clearInterval(S.launchTimer);
    await loadDetail();
    const drv = S.detail?.driver;
    if (drv?.failed && !warned) {
      warned = true;
      toast(`Запуск упал (код ${drv.returncode}) — смотрите вывод драйвера`, true);
      clearInterval(S.launchTimer);
    }
    if (S.detail?.runs?.length) clearInterval(S.launchTimer);
  }, 2000);
}

/* ────────────────────────────────────────────────────── rail ── */

function renderExperiments() {
  const list = $("#explist");
  list.textContent = "";
  if (!S.experiments.length) {
    list.append(el("li", "explist__empty", "Экспериментов пока нет."));
    return;
  }
  for (const e of S.experiments) {
    const li = el("li");
    const b = el("button", "exp");
    b.setAttribute("aria-current", String(e.name === S.selected));
    b.append(el("div", "exp__name", e.name));
    const meta = el("div", "exp__meta");
    if (e.running > 0 || e.launching) meta.append(el("span", "pulse"));
    if (e.failed) meta.append(el("span", "chip chip--warn", "запуск упал"));
    meta.append(el("span", null,
      e.n_runs === 0 && e.launching ? `запускается · ${ago(e.created)}`
        : `готово ${e.done}/${e.n_runs} · задач ${e.tasks.length} · ${ago(e.created)}`));
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

/** What the launcher itself is doing.
 *
 *  The driver spends its first minute scanning 128 redis dbs and staggering
 *  launches, so between the click and the first run's log there used to be
 *  nothing on screen at all — which read as "the button did nothing". This
 *  panel shows the driver's own output until the runs take over, and stays
 *  if it died so the reason is visible instead of silence. */
function driverPanel(drv, nRuns) {
  if (!drv || !drv.known) return null;
  if (!drv.alive && !drv.failed && nRuns > 0) return null;   // launcher done, runs speak for themselves

  const p = el("div", `driver${drv.failed ? " driver--bad" : ""}`);
  const head = el("div", "driver__head");
  if (drv.alive) head.append(el("span", "pulse"));
  head.append(el("b", null,
    drv.failed ? `Запуск упал (код ${drv.returncode})`
      : drv.stopped ? "Остановлено вручную"
      : drv.alive ? (nRuns ? "Запускаю остальные прогоны…" : "Запускаю: ищу свободные базы redis…")
      : "Запускающий процесс завершился"));
  if (drv.alive && !nRuns) {
    head.append(el("span", "driver__hint",
      "первый прогон появится, как только драйвер выберет базу и стартует run.py"));
  }
  p.append(head);

  const pre = el("pre", "driver__log");
  pre.textContent = (drv.log_lines || []).slice(-14).join("\n") || "(драйвер пока ничего не написал)";
  p.append(pre);
  return p;
}

function renderLive() {
  const d = S.detail;
  $("#live-eyebrow").textContent = d ? d.name : "Эксперимент не выбран";
  $("#live-title").textContent = d ? "Прогоны в работе" : "Наблюдение";
  const box = $("#runs");
  box.textContent = "";
  if (!d) { box.append(el("p", "empty", "Выберите эксперимент слева или запустите новый.")); return; }

  const panel = driverPanel(d.driver, d.runs.length);
  if (panel) box.append(panel);
  if (!d.runs.length && !panel) {
    box.append(el("p", "empty", "У этого эксперимента ещё нет ни одного прогона."));
  }

  for (const run of d.runs) {
    const m = METRIC[S.metric];
    const card = el("div", "runcard");
    const top = el("div", "runcard__top");
    top.append(el("div", "runcard__task", run.task),
               el("span", `runcard__cond cond--${run.condition}`, COND_RU[run.condition]));
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
    stats.append(stat("попыток", String(run.attempts ?? 0)));
    stats.append(stat("прогноз", last ? fmt(last[m.pred], m.unit) : "—"));
    stats.append(stat("факт", run[m.actual] ? fmt(run[m.actual], m.unit) : "—"));
    if (run[m.actual] && last) {
      const e = ((last[m.pred] - run[m.actual]) / run[m.actual]) * 100;
      const s = stat("ошибка", pct(e));
      s.querySelector(".stat__v").classList.add(Math.abs(e) <= 10 ? "win" : "loss");
      stats.append(s);
    }
    if (run.agent_events.length) {
      stats.append(stat("пробуждений", String(run.agent_events.length),
                        `${run.agent_events.filter((e) => e.moved.length).length} с действием`));
    }
    card.append(stats);
    card.append(ribbon(run));

    const st = el("div", "status");
    st.append(el("span", `dot dot--${run.status}`), el("b", null, run.status));
    if (last?.concurrency) st.append(el("span", null, `· конкурентность ${last.concurrency.toFixed(1)}`));
    st.append(el("span", null, `· db ${run.db}`));
    const logBtn = el("button", "ghost", "лог");
    logBtn.style.cssText = "margin-left:auto;padding:2px 8px;font-size:11px";
    logBtn.onclick = () => openLog(run.stem, 0);
    st.append(logBtn);
    card.append(st);
    box.append(card);
  }
}

/* ──────────────────────────────────────────── просмотр лога ── */

const LOG = { stem: null, focus: 0 };

/** Open the log pane. `focus` is a 1-indexed line to centre on — an agent
 *  trace carries the line it was written at, so a decision links straight
 *  to its own evidence in the raw log. */
async function openLog(stem, focus = 0, query = "") {
  LOG.stem = stem;
  LOG.focus = focus;
  const pane = $("#logpane");
  pane.hidden = false;
  $("#log-eyebrow").textContent = stem;
  $("#log-title").textContent = focus ? `Строка ${focus}` : "Хвост лога";
  const body = $("#log-body");
  body.textContent = "";
  body.append(el("p", "empty", "Читаю…"));

  const qs = query
    ? `q=${encodeURIComponent(query)}&tail=400`
    : focus ? `around=${focus}&ctx=70` : "tail=400";
  let r;
  try {
    r = await api(`/api/experiments/${S.selected}/log/${stem}?${qs}`);
  } catch (e) { toast(e.message, true); return; }

  body.textContent = "";
  if (query) {
    $("#log-title").textContent = `Совпадений: ${r.matches}`;
    if (!r.numbered.length) { body.append(el("p", "empty", "Ничего не найдено.")); return; }
  }
  let focusNode = null;
  for (const { n, text } of r.numbered) {
    const row = el("div", "logline");
    if (/CostMonitorAgentTrace|CostMonitorHook\] adjustments/.test(text)) row.classList.add("logline--agent");
    else if (/CostMonitorHookJSON/.test(text)) row.classList.add("logline--pred");
    else if (/ERROR|Traceback|EXCEPTION/.test(text)) row.classList.add("logline--err");
    if (n === focus) { row.classList.add("is-focus"); focusNode = row; }
    row.append(el("span", "logline__n", String(n)), el("span", "logline__t", text));
    body.append(row);
  }
  if (focusNode) focusNode.scrollIntoView({ block: "center" });
}

/* ───────────────────────────── разбор решений агента ── */

function decisionsBlock(runs) {
  const block = el("div", "block");
  const head = el("div", "block__head");
  head.append(el("h3", null, "Почему агент так решил"));
  head.append(el("p", null,
    "Каждое пробуждение разворачивается в то, на чём оно строилось: что его разбудило, "
    + "какие инструменты агент читал и что они вернули, какие действия он в итоге вызвал. "
    + "Ссылка открывает ту самую строку в сыром логе."));
  block.append(head);

  const withAgent = runs.filter((r) => r.condition === "withagent" && r.agent_events.length);
  if (!withAgent.length) {
    block.append(el("p", "empty", "Решений пока нет."));
    return block;
  }

  const pick = el("select");
  pick.className = "logpane__pick";
  for (const r of withAgent) {
    const o = el("option", null, `${r.task} — пробуждений ${r.agent_events.length}`);
    o.value = r.stem;
    pick.append(o);
  }
  const list = el("div");
  const draw = () => {
    const run = withAgent.find((r) => r.stem === pick.value) || withAgent[0];
    list.textContent = "";
    run.agent_events.forEach((ev, i) => list.append(decisionCard(run, ev, i)));
  };
  pick.onchange = draw;
  block.append(pick, list);
  draw();
  return block;
}

function decisionCard(run, ev, i) {
  const acted = ev.moved.length || ev.outliers || ev.skip_calibration;
  const card = el("div", `decision${acted ? "" : " decision--idle"}`);

  const head = el("div", "decision__head");
  head.append(el("div", "decision__at", `попытка ${ev.attempt}`));
  const trig = el("div", "decision__trigger");
  trig.innerHTML = ev.trigger ? `<b>разбудило:</b> ${escapeHtml(ev.trigger)}` : "<b>разбудило:</b> —";
  head.append(trig);

  const acts = el("div", "decision__acts");
  const shown = ev.actions && ev.actions.length
    ? ev.actions
    : (acted ? ["adjust_model"] : []);
  if (!shown.length) acts.append(el("span", "act act--none", "ничего не вызвал"));
  for (const a of shown) acts.append(el("span", "act act--fire", ACTION_RU[a] || a));
  head.append(acts);
  head.onclick = () => card.classList.toggle("is-open");
  card.append(head);

  const body = el("div", "decision__body");

  const levers = el("div", "levers");
  for (const line of actionLines(ev)) {
    const lv = el("div", "lever");
    lv.innerHTML = line;
    levers.append(lv);
  }
  for (const [k, v] of Object.entries(ev.levers || {})) {
    if (v != null) continue;
    const lv = el("div", "lever lever--off", `${LEVER_RU[k] || k} — не тронут`);
    levers.append(lv);
  }
  body.append(levers);

  if (ev.reason) {
    const why = el("div", "decision__why");
    why.innerHTML = `<em>обоснование агента</em>${escapeHtml(ev.reason)}`;
    body.append(why);
  }

  const evidence = ev.evidence || {};
  const keys = Object.keys(evidence);
  if (keys.length) {
    const box = el("div", "evidence");
    box.append(el("div", "evidence__t", "что агент прочитал перед решением"));
    for (const k of keys) {
      const d = el("details", "tool");
      const sum = el("summary", null, TOOL_RU[k] || k);
      const pre = el("pre", null, evidence[k]);
      d.append(sum, pre);
      box.append(d);
    }
    body.append(box);
  } else {
    body.append(el("p", "decision__why",
      "Улики не записаны: прогон сделан до того, как в лог начал писаться [CostMonitorAgentTrace]."));
  }

  if (ev.line) {
    const open = el("button", "ghost", "показать в логе");
    open.onclick = (e) => { e.stopPropagation(); openLog(run.stem, ev.line); };
    body.append(open);
  }

  card.append(body);
  if (i === 0) card.classList.add("is-open");
  return card;
}

const escapeHtml = (s) =>
  String(s).replace(/[<>&]/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" }[c]));

/* ──────────────────────────────────────────────────── analyze ── */

function renderAnalyze() {
  const box = $("#analysis");
  box.textContent = "";
  const d = S.detail;
  $("#an-eyebrow").textContent = d ? d.name : "Эксперимент не выбран";
  $("#an-title").textContent = "Автомат против автомата с агентом";
  if (!d) { box.append(el("p", "empty", "Выберите эксперимент слева.")); return; }

  const done = d.runs.filter(usable);
  if (!done.length) {
    box.append(el("p", "empty",
      "Пока нет ни одного прогона, где есть и ряд прогнозов, и измеренный итог. Загляните, когда добежит волна."));
    return;
  }
  const tasks = byTask(done);
  const paired = [...tasks.entries()].filter(([, c]) => c.noagent && c.withagent);

  box.append(headline(paired, tasks));
  box.append(checkpointTable(paired));
  box.append(harnessBlock(d.runs));
  box.append(observerBlock(d.runs));
  box.append(stabilityBlock(done));
  box.append(decisionsBlock(d.runs));
  box.append(smallMultiples(paired));
  box.append(finalTable(tasks));
}

/** Median and IQR across tasks at every 5% checkpoint.
 *
 *  `signed` keeps the sign, so the chart can show WHICH WAY the forecast is
 *  wrong — a model that overshoots by 30% and one that undershoots by 30%
 *  are the same point once you take the modulus, and they are not the same
 *  failure. The checkpoint table stays on the modulus: there "lower is
 *  better" has to hold for the delta row to mean anything. */
function condSeries(paired, cond, signed = false) {
  return GRID.map((p) => {
    const vals = paired.map(([, c]) => errAt(c[cond], p, S.metric))
                       .filter((v) => v != null).map((v) => (signed ? v : Math.abs(v)));
    return { p, med: median(vals), q1: quantile(vals, 0.25), q3: quantile(vals, 0.75), n: vals.length };
  });
}

/** Ticks covering [d0,d1] that always include 0, so a signed axis reads
 *  against a baseline rather than against whatever the data happened to hit. */
function signedTicks(d0, d1) {
  const span = d1 - d0;
  const raw = span / 6;
  const mag = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  const step = [1, 2, 2.5, 5, 10].map((s) => s * mag).find((s) => s >= raw) || mag * 10;
  const out = [];
  for (let t = Math.ceil(d0 / step) * step; t <= d1 + 1e-9; t += step) out.push(Math.round(t * 10) / 10);
  if (!out.includes(0) && d0 <= 0 && d1 >= 0) out.push(0);
  return out.sort((a, b) => a - b);
}

/** The chart shows either the median across every paired task, or one task on
 *  its own. A single task has no spread to draw, so the IQR band collapses to
 *  its own curve — which is exactly what you want when checking whether the
 *  median is hiding something. */
function taskFilterBar(paired, onChange) {
  const bar = el("div", "taskfilter");
  bar.append(el("span", "taskfilter__label", "Задача"));
  const sel = el("select");
  const all = el("option", null, `все задачи — медиана по ${paired.length}`);
  all.value = "__all__";
  sel.append(all);
  for (const [key] of paired) {
    const o = el("option", null, key);
    o.value = key;
    sel.append(o);
  }
  if (!paired.some(([k]) => k === S.taskFilter)) S.taskFilter = "__all__";
  sel.value = S.taskFilter;
  sel.onchange = () => { S.taskFilter = sel.value; onChange(); };
  bar.append(sel);
  return bar;
}

function headline(paired, tasks) {
  const block = el("div", "block");
  const head = el("div", "block__head");
  head.append(el("h3", null, "Ошибка прогноза по ходу прогона"));
  head.append(el("p", null,
    paired.length
      ? (S.taskFilter === "__all__"
          ? `Медиана ошибки по ${METRIC[S.metric].label} со знаком на ${paired.length} парных задачах, замер на каждых 5% прогресса. Выше нуля — прогноз завышен, ниже — занижен; модуль этого не показывает. Заливка — межквартильный размах.`
          : `Задача ${S.taskFilter}, ошибка по ${METRIC[S.metric].label} со знаком на каждых 5% прогресса. Флажки — вмешательства агента.`)
      : "Ни у одной задачи не готовы оба условия, пары строить не из чего."));
  block.append(head);

  if (!paired.length) {
    block.append(el("p", "empty", `Задач: ${tasks.size}, но ни одна не завершена в обоих условиях.`));
    return block;
  }

  const scoped = S.taskFilter === "__all__" ? paired : paired.filter(([k]) => k === S.taskFilter);
  block.append(taskFilterBar(paired, renderAnalyze));
  const A = condSeries(scoped, "noagent", true), B = condSeries(scoped, "withagent", true);
  const w = 980, h = 400, f = chartFrame(w, h, { l: 62, r: 158, t: 30, b: 40 });

  // Signed domain, always straddling zero: above the line the forecast is too
  // expensive, below it too cheap. Clamped at ±400% so one cold-start spike
  // (confidence_width(n=1) is 100% by design) cannot flatten the whole chart.
  const CLAMP = 400;
  const all = [...A, ...B].flatMap((d) => [d.q1, d.med, d.q3])
                          .filter((v) => v != null).map((v) => Math.max(-CLAMP, Math.min(CLAMP, v)));
  let d0 = Math.min(0, ...all), d1 = Math.max(0, ...all);
  const padv = Math.max(5, (d1 - d0) * 0.08);
  d0 -= padv; d1 += padv;
  const ticks = signedTicks(d0, d1);
  yAxis(f, d0, d1, ticks, `ошибка по ${METRIC[S.metric].label} со знаком, %`);
  xAxis(f, 5, 100, [5, 10, 15, 25, 50, 75, 100], "%");

  const X = (p) => f.x(p, 5, 100);
  const Y = (v) => f.y(Math.max(d0, Math.min(d1, v)), d0, d1);

  // the ±10% corridor, then the zero line on top of it
  if (d0 < 10 && d1 > -10) {
    f.svg.appendChild(svgEl("rect", {
      x: f.pad.l, width: w - f.pad.l - f.pad.r,
      y: Y(Math.min(10, d1)), height: Math.abs(Y(Math.max(-10, d0)) - Y(Math.min(10, d1))),
      fill: "var(--good)", opacity: 0.09,
    }));
  }
  f.svg.appendChild(svgEl("line", {
    x1: f.pad.l, x2: w - f.pad.r, y1: Y(0), y2: Y(0),
    stroke: "var(--ink)", "stroke-width": 1.5, opacity: 0.5,
  }));
  // no "0" caption here: the y axis already ticks it, and the caption sat in
  // the gutter where the curves put their own direct labels
  for (const [v, txt] of [[d1, "выше нуля: переоценка"], [d0, "ниже нуля: недооценка"]]) {
    const t = svgEl("text", { class: "axhint", x: f.pad.l + 6,
                               y: v > 0 ? f.pad.t + 12 : h - f.pad.b - 6 });
    t.textContent = txt;
    f.svg.appendChild(t);
  }

  const placed = [];  // y positions already used by a direct label
  for (const [data, color, name, dash] of [
    [A, "var(--auto)", "автомат", "5 4"],
    [B, "var(--agent)", "автомат + агент", null],
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
      bindTip(c, `<b>${d.p}% прогона</b><br>${name}<br>медиана ошибки <em>${pct(d.med)}</em> ` +
                 `(${d.med > 0 ? "переоценка" : "недооценка"})<br>IQR ${pct(d.q1)}…${pct(d.q3)} · задач ${d.n}`);
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

  // Where the agent actually intervened, pinned to the curve it is meant to
  // be bending. Placed at the 5% checkpoint nearest the wakeup, so a marker
  // always sits on a drawn point rather than floating between them.
  const atB = (p) => B.find((d) => d.p === p);
  for (const [key, c] of scoped) {
    const run = c.withagent;
    const total = run.attempts || 1;
    for (const ev of run.agent_events) {
      if (!(ev.moved.length || ev.outliers || ev.skip_calibration)) continue;
      const prog = Math.min(100, Math.max(5, (ev.attempt / total) * 100));
      const snap = GRID.reduce((a, b) => (Math.abs(b - prog) < Math.abs(a - prog) ? b : a), 5);
      const d = atB(snap);
      if (!d || d.med == null) continue;
      const x = X(snap), y = Y(d.med);
      const mark = svgEl("path", {
        d: `M${x},${y - 9} l5.5,9.5 l-11,0 Z`,        // a small flag on the curve
        fill: "var(--wake)", stroke: "var(--surface)", "stroke-width": 1.5,
      });
      f.svg.appendChild(mark);
      // the flag is 11px wide — give the pointer something it can actually hit
      const hit = svgEl("circle", { cx: x, cy: y - 4, r: 12, fill: "transparent",
                                     style: "cursor:help" });
      bindTip(hit, agentTipHtml(key, run, ev, snap));
      f.svg.appendChild(hit);
    }
  }

  const legend = el("div", "legend");
  legend.innerHTML =
    `<span style="color:var(--auto)"><i class="swatch swatch--dash"></i>автомат</span>` +
    `<span><i class="swatch" style="background:var(--agent)"></i>автомат + агент</span>` +
    `<span><i class="swatch" style="background:var(--wake)"></i>вмешательство агента</span>` +
    `<span><i class="swatch" style="background:var(--good);opacity:.45"></i>коридор ±10%</span>`;
  block.append(legend);

  const wrap = el("div", "chartwrap");
  wrap.append(f.svg);
  block.append(wrap);
  return block;
}

/** One wakeup, spelled out: what tripped it, why, what it did, what changed
 *  after. Used by the chart markers and, in longer form, by the decision cards. */
function actionLines(ev) {
  const out = [];
  for (const k of ev.moved) {
    const name = LEVER_RU[k] || k;
    const was = (ev.levers_before || {})[k];
    out.push(was != null && was > 0
      ? `${name}: ${(+was).toFixed(2)} → <b>${ev.levers[k]}</b>`
      : `${name} → <b>${ev.levers[k]}</b>`);
  }
  if (ev.outliers) {
    const idx = (ev.outlier_indices || []).join(", ");
    out.push(`флаг выброса на вызов${ev.outliers > 1 ? "ах" : ""} <b>${idx || ev.outliers}</b>`);
  }
  if (ev.skip_calibration) out.push("пропустить следующую калибровку");
  return out;
}

function agentTipHtml(taskKey, run, ev, snap) {
  const before = Math.abs(errAt(run, snap, S.metric) ?? 0);
  const afterP = Math.min(100, snap + 20);
  const after = Math.abs(errAt(run, afterP, S.metric) ?? 0);
  const acts = actionLines(ev);
  return `<b>${escapeHtml(taskKey)} · попытка ${ev.attempt}</b><br>` +
    `<em>триггер</em> ${escapeHtml(ev.trigger || "—")}<br>` +
    (ev.reason ? `<em>причина</em> ${escapeHtml(ev.reason.slice(0, 220))}<br>` : "") +
    `<em>действие</em> ${acts.join("<br>") || "—"}<br>` +
    `<em>ошибка</em> ${before.toFixed(1)}% на ${snap}% → ${after.toFixed(1)}% на ${afterP}%`;
}

function checkpointTable(paired) {
  const block = el("div", "block");
  const head = el("div", "block__head");
  head.append(el("h3", null, "Чекпоинты"));
  head.append(el("p", null,
    "Медиана модуля ошибки в тех точках, где реально принимают решение по бюджету. Дельта считается как (автомат + агент) − автомат: отрицательная означает, что агент снизил ошибку."));
  block.append(head);

  const A = condSeries(paired, "noagent"), B = condSeries(paired, "withagent");
  const at = (arr, p) => arr.find((d) => d.p === p);

  const t = el("table", "grid-table");
  const thead = el("thead"), hr = el("tr");
  hr.append(el("th", null, "Условие"));
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
  row("автомат", A);
  row("автомат + агент", B);

  const tr = el("tr");
  tr.append(el("td", null, "дельта = (автомат + агент) − автомат"));
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

/* ──────────────────────── эффективность харнесса агента ── */

/** Observability tools whose output the agent is TOLD it has, but which
 *  `CostMonitorAgent.build_prompt` does not actually inline into the prompt.
 *  The trace logs all six (`_log_agent_trace` calls them itself), so the
 *  console would otherwise present evidence the model never saw. */
const TOOLS_IN_PROMPT = new Set(["get_recent_calls", "get_backpressure", "get_model_params", "get_program_diff"]);

/** Signed error at the checkpoint nearest this wakeup, and `lookahead`
 *  further on — the crude before/after any single intervention gets. */
function wakeupImpact(run, ev, lookahead = 20) {
  const total = run.attempts || 1;
  const prog = Math.min(100, Math.max(5, (ev.attempt / total) * 100));
  const snap = GRID.reduce((a, b) => (Math.abs(b - prog) < Math.abs(a - prog) ? b : a), 5);
  const before = errAt(run, snap, S.metric);
  const after = errAt(run, Math.min(100, snap + lookahead), S.metric);
  if (before == null || after == null) return null;
  return { snap, before, after, delta: Math.abs(after) - Math.abs(before) };
}

/** Every wakeup in the experiment, flattened, with its impact attached. */
function harnessRows(runs) {
  const out = [];
  for (const run of runs) {
    if (run.condition !== "withagent") continue;
    for (const ev of run.agent_events) {
      out.push({ run, ev: { ...ev, run_stem: run.stem }, impact: wakeupImpact(run, ev) });
    }
  }
  return out;
}

function harnessBlock(runs) {
  const block = el("div", "block");
  const head = el("div", "block__head");
  head.append(el("h3", null, "Эффективность харнесса агента"));
  head.append(el("p", null,
    "Чем агент пользовался и что это дало. Дельта — изменение модуля ошибки от чекпоинта, "
    + "ближайшего к пробуждению, до чекпоинта на 20% дальше. Отрицательная дельта значит, что "
    + "после вызова ошибка стала меньше. Это корреляция на одном прогоне, а не доказательство "
    + "причины — рядом с каждой цифрой стоит число наблюдений."));
  block.append(head);

  const rows = harnessRows(runs);
  if (!rows.length) {
    block.append(el("p", "empty", "Пробуждений пока не записано."));
    return block;
  }

  block.append(harnessSummary(rows));
  block.append(toolTable(rows));
  block.append(impactStrip(rows));
  return block;
}

function harnessSummary(rows) {
  const acted = rows.filter(({ ev }) => ev.moved.length || ev.outliers || ev.skip_calibration);
  const withImpact = rows.filter((r) => r.impact);
  const helped = withImpact.filter((r) => r.impact.delta < 0);
  // Two wakeups naming the same surprise are that surprise handled twice —
  // the hook can dispatch the agent concurrently from the attempt path and
  // the accept path, and both copies then act on overlapping evidence.
  // Keyed on the event, not the whole string: two dispatches of one breach
  // quote estimates seconds apart ("2402s" vs "2407s") and would otherwise
  // look like different triggers.
  const seen = new Map();
  for (const { ev } of rows) {
    const k = `${ev.run_stem}|${(ev.trigger || "").replace(/:.*$/, "").trim()}`;
    seen.set(k, (seen.get(k) || 0) + 1);
  }
  const dup = [...seen.values()].filter((n) => n > 1).reduce((a, n) => a + n - 1, 0);

  const wrap = el("div", "scorecard");
  const card = (k, v, sub, cls) => {
    const c = el("div", "score");
    c.append(el("div", "score__k", k));
    const val = el("div", `score__v${cls ? " " + cls : ""}`, v);
    c.append(val);
    if (sub) c.append(el("div", "score__s", sub));
    return c;
  };
  wrap.append(card("пробуждений", String(rows.length),
                   `на ${new Set(rows.map((r) => r.run.stem)).size} прогонах`));
  wrap.append(card("с действием", `${acted.length}`,
                   `${Math.round((acted.length / rows.length) * 100)}% — остальные посмотрели и отказались`));
  const medD = median(withImpact.map((r) => r.impact.delta));
  wrap.append(card("медиана дельты", medD == null ? "—" : pct(medD),
                   `по ${withImpact.length} пробуждениям с измеримым эффектом`,
                   medD == null ? "" : medD < 0 ? "win" : "loss"));
  wrap.append(card("помогли", withImpact.length ? `${helped.length}/${withImpact.length}` : "—",
                   "ошибка снизилась после вызова"));
  wrap.append(card("повторных триггеров", String(dup),
                   dup ? "одно и то же событие обработано дважды" : "дубликатов нет",
                   dup ? "loss" : "win"));
  return wrap;
}

/** Per-tool scorecard: how often it fired and what happened to the error.
 *  Observability tools are counted from the recorded evidence, action tools
 *  from what the agent actually invoked. */
function toolTable(rows) {
  const stat = new Map();
  const bump = (name, kind, r) => {
    if (!stat.has(name)) stat.set(name, { name, kind, n: 0, deltas: [] });
    const s = stat.get(name);
    s.n += 1;
    if (r.impact) s.deltas.push(r.impact.delta);
  };
  for (const r of rows) {
    for (const k of Object.keys(r.ev.evidence || {})) bump(k, "obs", r);
    const acts = r.ev.actions && r.ev.actions.length
      ? r.ev.actions
      : [...(r.ev.moved.length ? ["adjust_model"] : []),
         ...(r.ev.outliers ? ["flag_as_outlier"] : []),
         ...(r.ev.skip_calibration ? ["skip_next_calibration"] : [])];
    for (const a of acts) bump(a, "act", r);
    for (const k of r.ev.moved) bump(`рычаг ${LEVER_RU[k] || k}`, "lever", r);
  }

  const list = [...stat.values()].sort((a, b) =>
    a.kind === b.kind ? b.n - a.n : ["act", "lever", "obs"].indexOf(a.kind) - ["act", "lever", "obs"].indexOf(b.kind));
  const maxN = Math.max(...list.map((s) => s.n), 1);

  const t = el("table", "grid-table tooltable");
  t.innerHTML = "<thead><tr><th>Инструмент</th><th>Тип</th><th>Вызовов</th>"
    + "<th>Медиана дельты</th><th>Помогло</th><th>Доехало до модели</th></tr></thead>";
  const tb = el("tbody");
  const KIND = { obs: "наблюдение", act: "действие", lever: "рычаг" };
  for (const s of list) {
    const tr = el("tr");
    const nameCell = el("td");
    nameCell.append(el("span", "tool__n", TOOL_RU[s.name]?.split(" — ")[0] || ACTION_RU[s.name] || s.name));
    const hint = TOOL_RU[s.name]?.split(" — ")[1];
    if (hint) nameCell.append(el("span", "tool__h", hint));
    tr.append(nameCell);
    tr.append(el("td", null, KIND[s.kind]));

    const bar = el("td", "cell--bar");
    const fill = el("span", "bar__f");
    fill.style.width = `${(s.n / maxN) * 100}%`;
    bar.append(fill, el("span", "bar__n", String(s.n)));
    tr.append(bar);

    // Attribution only for things the agent CHOSE. Every wakeup reads every
    // observability tool, so their per-tool delta is just the overall delta
    // repeated once per row — a number that looks like evidence and isn't.
    if (s.kind === "obs") {
      const na = el("td", "cell--na", "не приписывается");
      na.title = "Инструмент наблюдения читается на каждом пробуждении, "
        + "поэтому его «эффект» — это просто общая дельта, повторённая в каждой строке.";
      na.colSpan = 2;
      tr.append(na);
    } else {
      const med = median(s.deltas);
      const dc = el("td", null, med == null ? "—" : pct(med));
      if (med != null) dc.classList.add(med < 0 ? "win" : "loss");
      tr.append(dc);
      const good = s.deltas.filter((d) => d < 0).length;
      tr.append(el("td", null, s.deltas.length ? `${good}/${s.deltas.length}` : "—"));
    }

    const reach = el("td");
    if (s.kind !== "obs") reach.textContent = "—";
    else if (TOOLS_IN_PROMPT.has(s.name)) reach.append(el("span", "flag flag--ok", "да"));
    else {
      const bad = el("span", "flag flag--bad", "нет");
      bad.title = "Инструмент пишется в трейс, но build_prompt не вкладывает его вывод "
        + "в промпт — агент этих данных не видел, хотя системный промпт на них ссылается.";
      reach.append(bad);
    }
    tr.append(reach);
    tb.append(tr);
  }
  t.append(tb);

  const wrap = el("div");
  wrap.append(t);
  const miss = list.filter((s) => s.kind === "obs" && !TOOLS_IN_PROMPT.has(s.name));
  if (miss.length) {
    wrap.append(el("p", "note",
      `Внимание: ${miss.length} инструмент(ов) наблюдения записаны в трейс, но не попадают в промпт `
      + `(${miss.map((s) => s.name).join(", ")}). Агент принимал решение без них.`));
  }
  return wrap;
}

/** Every wakeup as one signed bar: error before vs after. The one view that
 *  shows a lever that fired and made things worse. */
function impactStrip(rows) {
  const pts = rows.filter((r) => r.impact);
  if (!pts.length) return el("p", "empty", "Ни у одного пробуждения нет точек до и после.");

  const w = 980, h = 220, f = chartFrame(w, h, { l: 62, r: 20, t: 26, b: 46 });
  const vals = pts.map((r) => r.impact.delta);
  let d0 = Math.min(0, ...vals), d1 = Math.max(0, ...vals);
  const padv = Math.max(3, (d1 - d0) * 0.1);
  d0 -= padv; d1 += padv;
  yAxis(f, d0, d1, signedTicks(d0, d1), "изменение модуля ошибки после вызова, п.п.");
  const Y = (v) => f.y(Math.max(d0, Math.min(d1, v)), d0, d1);
  f.svg.appendChild(svgEl("line", { x1: f.pad.l, x2: w - f.pad.r, y1: Y(0), y2: Y(0),
                                     stroke: "var(--ink)", "stroke-width": 1.5, opacity: 0.5 }));

  const bw = Math.max(3, Math.min(26, (w - f.pad.l - f.pad.r) / (pts.length * 1.4)));
  pts.forEach((r, i) => {
    const x = f.x(i + 0.5, 0, pts.length);
    const d = r.impact.delta;
    const acted = r.ev.moved.length || r.ev.outliers || r.ev.skip_calibration;
    const rect = svgEl("rect", {
      x: x - bw / 2, width: bw,
      y: Math.min(Y(0), Y(d)), height: Math.max(1.5, Math.abs(Y(d) - Y(0))),
      fill: d < 0 ? "var(--good)" : "var(--crit)",
      opacity: acted ? 0.9 : 0.35,
    });
    bindTip(rect, `<b>${escapeHtml(r.run.task)} · попытка ${r.ev.attempt}</b><br>` +
      `<em>триггер</em> ${escapeHtml((r.ev.trigger || "—").slice(0, 140))}<br>` +
      `<em>действие</em> ${actionLines(r.ev).join("<br>") || "ничего не вызвал"}<br>` +
      `<em>ошибка</em> ${pct(r.impact.before)} на ${r.impact.snap}% → ${pct(r.impact.after)} далее<br>` +
      `<em>дельта модуля</em> ${pct(r.impact.delta)}`);
    f.svg.appendChild(rect);
  });

  const legend = el("div", "legend");
  legend.innerHTML =
    `<span><i class="swatch" style="background:var(--good)"></i>ошибка снизилась</span>` +
    `<span><i class="swatch" style="background:var(--crit)"></i>ошибка выросла</span>` +
    `<span><i class="swatch" style="background:var(--ink-3);opacity:.35"></i>полупрозрачные — пробуждение без действия</span>`;

  const wrap = el("div");
  wrap.append(legend);
  const cw = el("div", "chartwrap");
  cw.append(f.svg);
  wrap.append(cw);
  return wrap;
}

function smallMultiples(paired) {
  const block = el("div", "block");
  const head = el("div", "block__head");
  head.append(el("h3", null, "По задачам"));
  head.append(el("p", null, "Та же кривая ошибки со знаком на каждой задаче отдельно — чтобы одиночная удача или провал не спрятались внутри медианы. Тонкая линия — ноль."));
  block.append(head);

  const legend = el("div", "legend");
  legend.innerHTML =
    `<span style="color:var(--auto)"><i class="swatch swatch--dash"></i>автомат</span>` +
    `<span><i class="swatch" style="background:var(--agent)"></i>автомат + агент</span>`;
  block.append(legend);

  const grid = el("div", "smalls");
  for (const [key, c] of paired) {
    const cell = el("div", "small");
    cell.append(el("div", "small__t", key));
    const f = chartFrame(300, 132, { l: 38, r: 8, t: 10, b: 20 });
    // signed, like the headline chart: the direction of the miss is the point
    const curves = ["noagent", "withagent"].map((cond) =>
      GRID.map((p) => ({ p, v: errAt(c[cond], p, S.metric) })).filter((d) => d.v != null));
    const vals = curves.flat().map((d) => Math.max(-400, Math.min(400, d.v)));
    let lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
    const pd = Math.max(4, (hi - lo) * 0.1);
    lo -= pd; hi += pd;
    yAxis(f, lo, hi, signedTicks(lo, hi).filter((_, i, a) => a.length <= 4 || i % 2 === 0), null);
    xAxis(f, 5, 100, [5, 50, 100], "");
    const zy = f.y(0, lo, hi);
    f.svg.appendChild(svgEl("line", { x1: f.pad.l, x2: 300 - f.pad.r, y1: zy, y2: zy,
                                       stroke: "var(--ink)", "stroke-width": 1, opacity: 0.45 }));
    curves.forEach((pts, i) => {
      if (!pts.length) return;
      const path = svgEl("path", {
        class: "line", "stroke-width": 1.8,
        stroke: i ? "var(--agent)" : "var(--auto)",
        d: linePath(pts.map((d) => [f.x(d.p, 5, 100), f.y(Math.max(lo, Math.min(hi, d.v)), lo, hi)])),
      });
      if (!i) path.setAttribute("stroke-dasharray", "4 3");
      f.svg.appendChild(path);
    });
    cell.append(f.svg);
    const fin = ["noagent", "withagent"].map((cond) => errAt(c[cond], 100, S.metric));
    const d = el("div", "small__d");
    d.innerHTML = `итог <span style="color:var(--auto)">${pct(fin[0])}</span> → <span style="color:var(--agent)">${pct(fin[1])}</span>`;
    cell.append(d);
    grid.append(cell);
  }
  block.append(grid);
  return block;
}

function finalTable(tasks) {
  const block = el("div", "block");
  const head = el("div", "block__head");
  head.append(el("h3", null, "Итог: прогноз против факта"));
  head.append(el("p", null,
    "Медиана всех прогнозов, которые прогон успел опубликовать, против того, во что он реально обошёлся — "
    + "и по времени, и по токенам сразу: это два разных провала, время анкорится на измеренном wall time, "
    + "токены нет. Медиана, а не последнее значение: последняя точка — это оценка, у которой почти не "
    + "осталось неизвестного хвоста, она польстила бы модели."));
  block.append(head);

  // Both metrics side by side: the run is only "predicted well" if time AND
  // tokens land, and they fail in different ways — duration is anchored on
  // measured wall time, tokens are not.
  const t = el("table", "grid-table finaltable");
  t.innerHTML =
    "<thead>"
    + "<tr><th rowspan='2'>Задача</th><th rowspan='2'>Условие</th>"
    + "<th colspan='3' class='grp'>Время</th><th colspan='3' class='grp'>Токены</th>"
    + "<th rowspan='2'>Пробуждений</th></tr>"
    + "<tr><th>прогноз</th><th>факт</th><th>ошибка</th>"
    + "<th>прогноз</th><th>факт</th><th>ошибка</th></tr>"
    + "</thead>";
  const tb = el("tbody");

  const cells = (run, metric, tr) => {
    const m = METRIC[metric];
    const med = median(run.series.map((x) => x[m.pred]).filter((v) => v != null));
    const actual = run[m.actual];
    const err = med != null && actual ? ((med - actual) / actual) * 100 : null;
    tr.append(el("td", "num", med != null ? fmt(med, m.unit) : "—"));
    tr.append(el("td", "num", actual ? fmt(actual, m.unit) : "—"));
    const e = el("td", "num", pct(err));
    if (err != null) e.classList.add(Math.abs(err) <= 10 ? "win" : "loss");
    tr.append(e);
    return err;
  };

  const errs = { duration: { noagent: [], withagent: [] }, tokens: { noagent: [], withagent: [] } };
  for (const [key, c] of tasks) {
    let first = true;
    for (const cond of ["noagent", "withagent"]) {
      const run = c[cond];
      if (!run) continue;
      const tr = el("tr");
      if (first) tr.classList.add("row--group");
      tr.append(el("td", null, first ? key : ""));
      tr.append(el("td", null, COND_RU[cond]));
      for (const metric of ["duration", "tokens"]) {
        const e = cells(run, metric, tr);
        if (e != null) errs[metric][cond].push(Math.abs(e));
      }
      tr.append(el("td", "num", run.agent_events.length ? String(run.agent_events.length) : "—"));
      tb.append(tr);
      first = false;
    }
  }

  // medians across tasks, so the table answers "and overall?" without a
  // second pass by eye
  for (const cond of ["noagent", "withagent"]) {
    const tr = el("tr", "row--total");
    tr.append(el("td", null, "медиана |ошибки|"));
    tr.append(el("td", null, COND_RU[cond]));
    for (const metric of ["duration", "tokens"]) {
      tr.append(el("td", "num", ""), el("td", "num", ""));
      const med = median(errs[metric][cond]);
      const c = el("td", "num", med == null ? "—" : `${med.toFixed(1)}%`);
      if (med != null) c.classList.add(med <= 10 ? "win" : "loss");
      tr.append(c);
    }
    tr.append(el("td", "num", ""));
    tb.append(tr);
  }
  t.append(tb);
  block.append(t);
  return block;
}

/* ─────────────────────────── накладной расход наблюдателя ── */

const OBSERVER_STAGE = "CostMonitorAgent";

/** What the monitor costs the very run it is measuring.
 *
 *  CostMonitorAgent's own inference is emitted as an LLM_CALL like any other
 *  stage, so it is inside the telemetry the cost model is fitted on. When one
 *  of those calls is slow it shows up as the run's biggest latency outlier —
 *  and the agent has been observed flagging it as an anomaly. */
function observerBlock(runs) {
  const rows = runs.filter((r) => r.stages && r.stages[OBSERVER_STAGE]);
  const block = el("div", "block");
  const head = el("div", "block__head");
  head.append(el("h3", null, "Наблюдатель внутри наблюдаемого"));
  head.append(el("p", null,
    "Вызовы самого CostMonitorAgent идут по той же шине событий, что и мутации, "
    + "и попадают в телеметрию, по которой строится прогноз. Здесь видно, сколько "
    + "измеренной латентности и токенов прогона принадлежит наблюдателю, и насколько "
    + "его самый долгий вызов выделяется на фоне остальных."));
  block.append(head);
  if (!rows.length) {
    block.append(el("p", "empty", "В логах нет вызовов CostMonitorAgent — либо это прогоны без агента, либо старый формат."));
    return block;
  }

  const t = el("table", "grid-table");
  t.innerHTML = "<thead><tr><th>Прогон</th><th>Вызовов агента</th><th>Доля латентности</th>"
    + "<th>Доля токенов</th><th>Самый долгий вызов</th><th>Медиана по прогону</th></tr></thead>";
  const tb = el("tbody");
  for (const run of rows) {
    const st = run.stages;
    const totLat = Object.values(st).reduce((a, s) => a + s.latency_ms, 0) || 1;
    const totTok = Object.values(st).reduce((a, s) => a + s.tokens, 0) || 1;
    const o = st[OBSERVER_STAGE];
    const share = (o.latency_ms / totLat) * 100;
    // median per-call latency across every other stage, as the yardstick the
    // observer's worst call is measured against
    const others = Object.entries(st).filter(([k]) => k !== OBSERVER_STAGE)
                         .map(([, s]) => s.latency_ms / Math.max(s.calls, 1));
    const base = median(others) || 0;

    const tr = el("tr");
    tr.append(el("td", null, `${run.task} · ${COND_RU[run.condition]}`));
    tr.append(el("td", "num", String(o.calls)));
    const sc = el("td", "num", `${share.toFixed(1)}%`);
    if (share > 10) sc.classList.add("loss");
    else if (share <= 3) sc.classList.add("win");
    tr.append(sc);
    tr.append(el("td", "num", `${((o.tokens / totTok) * 100).toFixed(1)}%`));
    const mx = el("td", "num", fmt(o.max_latency_ms / 1000, "s"));
    if (base && o.max_latency_ms / base > 5) {
      mx.classList.add("loss");
      mx.title = `в ${(o.max_latency_ms / base).toFixed(1)}× дольше медианного вызова прогона — `
        + "такой вызов агент видит как выброс в собственной телеметрии";
    }
    tr.append(mx);
    tr.append(el("td", "num", base ? fmt(base / 1000, "s") : "—"));
    tb.append(tr);
  }
  t.append(tb);
  block.append(t);
  return block;
}

/* ──────────────────────────────── устойчивость оценщика ── */

/** Two calibration numbers the raw series already contains:
 *
 *  breaches — how often a new estimate lands outside the interval the PREVIOUS
 *  estimate published. That is exactly the agent's wakeup condition, so it is
 *  also the wakeup rate the harness is going to see.
 *
 *  coverage — how often the published interval actually contained the final
 *  measured value. An interval that is never breached but never covers the
 *  truth is just a wide interval. */
function stabilityRow(run, metric) {
  const m = METRIC[metric], s = run.series, actual = run[m.actual];
  let breaches = 0, covered = 0, n = 0;
  let prev = null;
  for (const p of s) {
    const lo = p[m.lo], hi = p[m.hi], v = p[m.pred];
    if (prev && prev.hi > prev.lo && prev.lo > 0 && !(v >= prev.lo && v <= prev.hi)) breaches += 1;
    if (actual && hi > lo) { n += 1; if (actual >= lo && actual <= hi) covered += 1; }
    prev = { lo, hi };
  }
  const last = s[s.length - 1] || {};
  return {
    run, points: s.length,
    breach: s.length > 1 ? (breaches / (s.length - 1)) * 100 : null,
    coverage: n ? (covered / n) * 100 : null,
    // >1 means the run had to widen its own interval to stop being surprised,
    // <1 that it was too cautious. Absent on runs logged before calibration.
    aciScale: last.aci_scale ?? null,
  };
}

function stabilityBlock(runs) {
  const block = el("div", "block");
  const head = el("div", "block__head");
  head.append(el("h3", null, "Устойчивость и калибровка оценщика"));
  head.append(el("p", null,
    "«Пробитий» — доля точек, где новая оценка вышла за интервал, который объявила предыдущая. "
    + "Это же условие будит агента, то есть это и есть частота его пробуждений. «Покрытие» — "
    + "доля точек, чей интервал реально накрыл итоговый факт: интервал, который никогда не "
    + "пробивается, но и не накрывает правду, просто слишком широк."));
  block.append(head);

  const rows = runs.filter((r) => r.series.length > 1).map((r) => stabilityRow(r, S.metric));
  if (!rows.length) {
    block.append(el("p", "empty", "Ещё нет прогонов с рядом прогнозов."));
    return block;
  }

  const t = el("table", "grid-table");
  t.innerHTML = "<thead><tr><th>Прогон</th><th>Условие</th><th>Точек</th>"
    + "<th>Пробитий своего интервала</th><th>Покрытие факта интервалом</th>"
    + "<th>Ширина к концу</th></tr></thead>";
  const tb = el("tbody");
  for (const r of rows.sort((a, b) => (b.breach ?? 0) - (a.breach ?? 0))) {
    const tr = el("tr");
    tr.append(el("td", null, r.run.task));
    tr.append(el("td", null, COND_RU[r.run.condition]));
    tr.append(el("td", "num", String(r.points)));
    const b = el("td", "num", r.breach == null ? "—" : `${r.breach.toFixed(0)}%`);
    if (r.breach != null) b.classList.add(r.breach > 15 ? "loss" : "win");
    tr.append(b);
    const c = el("td", "num", r.coverage == null ? "—" : `${r.coverage.toFixed(0)}%`);
    if (r.coverage != null) c.classList.add(r.coverage >= 60 ? "win" : "loss");
    tr.append(c);
    const a = el("td", "num", r.aciScale == null ? "—" : `×${r.aciScale.toFixed(2)}`);
    if (r.aciScale != null) {
      a.title = r.aciScale > 1.05
        ? "прогон расширял собственный интервал: модельная ширина была слишком оптимистична"
        : r.aciScale < 0.95
          ? "прогон сужал интервал: модельная ширина была избыточно осторожной"
          : "модельная ширина оказалась примерно верной";
    }
    tr.append(a);
    tb.append(tr);
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

// Both views carry their own picker; they stay in sync so switching view
// never silently changes which number you are looking at.
document.querySelectorAll(".metricpick").forEach((pickBox) => {
  pickBox.onclick = (e) => {
    const b = e.target.closest("button[data-metric]");
    if (!b) return;
    S.metric = b.dataset.metric;
    document.querySelectorAll(".metricpick button[data-metric]").forEach(
      (x) => x.classList.toggle("is-on", x.dataset.metric === S.metric));
    renderAnalyze();
    renderLive();
  };
});

$("#log-close").onclick = () => { $("#logpane").hidden = true; };
$("#log-tail").onclick = () => { if (LOG.stem) openLog(LOG.stem, 0); };
$("#log-search").onkeydown = (e) => {
  if (e.key !== "Enter" || !LOG.stem) return;
  const q = e.target.value.trim();
  q ? openLog(LOG.stem, 0, q) : openLog(LOG.stem, 0);
};
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") $("#logpane").hidden = true;
});

$("#stop-exp").onclick = async () => {
  if (!S.selected) return toast("Эксперимент не выбран", true);
  await post(`/api/experiments/${S.selected}/stop`);
  toast(`Сигнал остановки отправлен: ${S.selected}`);
  loadDetail();
};

$("#build-report").onclick = async () => {
  if (!S.selected) return toast("Эксперимент не выбран", true);
  const b = $("#build-report");
  b.disabled = true; b.textContent = "Собираю…";
  try {
    // build the PNG/CSV artifacts first, then hand the whole thing to the
    // browser as one download — nothing for the operator to go find on disk
    const r = await post(`/api/experiments/${S.selected}/report`);
    if (!r.ok) toast(`Графики не собрались, качаю логи как есть — ${r.output.slice(-160)}`, true);
    const a = document.createElement("a");
    a.href = `api/experiments/${S.selected}/archive`;
    a.download = `${S.selected}.zip`;
    a.click();
    if (r.ok) toast(`Архив собран: ${S.selected}.zip`);
  } catch (e) { toast(e.message, true); }
  b.disabled = false; b.textContent = "Скачать архив";
};

(async function boot() {
  try {
    S.repo = await api("/api/repo");
    renderRepo();
    await loadProblems();
    await loadExperiments();
    if (S.experiments.length) await select(S.experiments[0].name);
    tick();
  } catch (e) { toast(`Бэкенд недоступен: ${e.message}`, true); }
})();
