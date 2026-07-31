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
    await loadExperiments();
    select(res.name);
    goto("live");
  } catch (e) {
    toast(`Не удалось запустить: ${e.message}`, true);
  }
  btn.textContent = "Запустить эксперимент";
  renderLedger();
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
    meta.append(el("span", null,
      `готово ${e.done}/${e.n_runs} · задач ${e.tasks.length} · ${ago(e.created)}`));
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
  $("#live-eyebrow").textContent = d ? d.name : "Эксперимент не выбран";
  $("#live-title").textContent = d ? "Прогоны в работе" : "Наблюдение";
  const box = $("#runs");
  box.textContent = "";
  if (!d) { box.append(el("p", "empty", "Выберите эксперимент слева или запустите новый.")); return; }

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
  box.append(agentBlock(d.runs));
  box.append(decisionsBlock(d.runs));
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
          ? `Медиана модуля ошибки по ${METRIC[S.metric].label} на ${paired.length} парных задачах, замер на каждых 5% прогресса. Заливка — межквартильный размах: насколько задачи расходятся между собой.`
          : `Задача ${S.taskFilter}, модуль ошибки по ${METRIC[S.metric].label} на каждых 5% прогресса. Флажки — вмешательства агента.`)
      : "Ни у одной задачи не готовы оба условия, пары строить не из чего."));
  block.append(head);

  if (!paired.length) {
    block.append(el("p", "empty", `Задач: ${tasks.size}, но ни одна не завершена в обоих условиях.`));
    return block;
  }

  const scoped = S.taskFilter === "__all__" ? paired : paired.filter(([k]) => k === S.taskFilter);
  block.append(taskFilterBar(paired, renderAnalyze));
  const A = condSeries(scoped, "noagent"), B = condSeries(scoped, "withagent");
  const w = 980, h = 380, f = chartFrame(w, h, { l: 54, r: 158, t: 30, b: 40 });
  const all = [...A, ...B].flatMap((d) => [d.q3, d.med]).filter((v) => v != null);
  const top = Math.max(20, Math.ceil(Math.max(...all) / 10) * 10);
  const ticks = Array.from({ length: 6 }, (_, i) => Math.round((top / 5) * i));
  yAxis(f, 0, top, ticks, `модуль ошибки по ${METRIC[S.metric].label}, %`);
  xAxis(f, 5, 100, [5, 10, 15, 25, 50, 75, 100], "%");

  const X = (p) => f.x(p, 5, 100), Y = (v) => f.y(Math.min(v, top), 0, top);

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
      bindTip(c, `<b>${d.p}% прогона</b><br>${name}<br>медиана ошибки <em>${d.med.toFixed(1)}%</em><br>IQR ${d.q1.toFixed(1)}–${d.q3.toFixed(1)}% · задач ${d.n}`);
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

  // the 10% band every forecast is trying to get inside
  const y10 = Y(10);
  if (y10 > f.pad.t) {
    f.svg.appendChild(svgEl("line", { x1: f.pad.l, x2: w - f.pad.r, y1: y10, y2: y10,
                                       stroke: "var(--ink-3)", "stroke-width": 1, "stroke-dasharray": "2 4" }));
    const t = svgEl("text", { class: "tick", x: w - f.pad.r - 2, y: y10 - 6, "text-anchor": "end" });
    t.textContent = "10% — годная оценка";
    f.svg.appendChild(t);
  }

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

function agentBlock(runs) {
  const block = el("div", "block");
  const head = el("div", "block__head");
  head.append(el("h3", null, "Что именно делал агент"));
  head.append(el("p", null,
    "Каждое пробуждение на оси попыток самого прогона. Жирная насечка — сдвинул хотя бы один рычаг, тонкая — посмотрел и отказался. Наведите, чтобы увидеть триггер и обоснование."));
  block.append(head);

  const withAgent = runs.filter((r) => r.condition === "withagent");
  if (!withAgent.some((r) => r.agent_events.length)) {
    block.append(el("p", "empty", "Пробуждений пока не записано."));
    return block;
  }

  const legend = el("div", "legend");
  legend.innerHTML =
    `<span><i class="swatch" style="background:var(--wake)"></i>сдвинул рычаг</span>` +
    `<span><i class="swatch" style="background:var(--ink-3)"></i>проснулся, ничего не изменил</span>` +
    `<span><i class="swatch" style="background:var(--agent)"></i>пометил выбросы</span>`;
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
        ? ev.moved.map((k) => `${LEVER_RU[k] || k} → ${ev.levers[k]}`).join("<br>")
        : "<em>рычаги не тронуты</em>";
      bindTip(mark, `<b>попытка ${ev.attempt}</b>${ev.ts ? ` <em>${ev.ts}</em>` : ""}<br>` +
                    `${levers}${ev.outliers ? `<br>помечено выбросов: ${ev.outliers}` : ""}<br>` +
                    `<em>триггер</em> ${ev.trigger || "—"}` +
                    (ev.reason ? `<br><em>почему</em> ${ev.reason.slice(0, 260)}` : ""));
      f.svg.appendChild(mark);
    }
    lane.append(f.svg);
    const acted = run.agent_events.filter((e) => e.moved.length || e.outliers).length;
    lane.append(el("div", "lane__n", `${acted}/${run.agent_events.length} с действием`));
    block.append(lane);
  }
  return block;
}

function smallMultiples(paired) {
  const block = el("div", "block");
  const head = el("div", "block__head");
  head.append(el("h3", null, "По задачам"));
  head.append(el("p", null, "Та же кривая ошибки на каждой задаче отдельно — чтобы одиночная удача или провал не спрятались внутри медианы."));
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
    "Медиана всех прогнозов, которые прогон успел опубликовать, против того, во что он реально обошёлся. "
    + "Медиана, а не последнее значение: последняя точка — это оценка, у которой почти не осталось "
    + "неизвестного хвоста, она польстила бы модели."));
  block.append(head);

  const m = METRIC[S.metric];
  const t = el("table", "grid-table");
  t.innerHTML = `<thead><tr><th>Задача</th><th>Условие</th><th>Прогноз (медиана)</th><th>Факт</th><th>Ошибка</th><th>Пробуждений</th></tr></thead>`;
  const tb = el("tbody");
  for (const [key, c] of tasks) {
    let first = true;
    for (const cond of ["noagent", "withagent"]) {
      const run = c[cond];
      if (!run) continue;
      const med = median(run.series.map((x) => x[m.pred]));
      const actual = run[m.actual];
      const err = med != null && actual ? ((med - actual) / actual) * 100 : null;
      const tr = el("tr");
      tr.append(el("td", null, first ? key : ""));
      tr.append(el("td", null, COND_RU[cond]));
      tr.append(el("td", null, med != null ? fmt(med, m.unit) : "—"));
      tr.append(el("td", null, fmt(actual, m.unit)));
      const e = el("td", null, pct(err));
      if (err != null) e.classList.add(Math.abs(err) <= 10 ? "win" : "loss");
      tr.append(e);
      tr.append(el("td", null, run.agent_events.length ? String(run.agent_events.length) : "—"));
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
