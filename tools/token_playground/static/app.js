const $ = (selector) => document.querySelector(selector);
const state = { task: null, job: null, timer: null, catalog: null };

function compact(value) {
  const n = Number(value || 0);
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n >= 10_000_000 ? 1 : 2).replace(".", ",")} млн`;
  if (n >= 1_000) return `${Math.round(n / 1_000)} тыс.`;
  return new Intl.NumberFormat("ru-RU").format(n);
}

function plural(n, one, few, many) {
  const mod100 = Math.abs(n) % 100;
  const mod10 = mod100 % 10;
  if (mod100 > 10 && mod100 < 20) return many;
  if (mod10 === 1) return one;
  if (mod10 >= 2 && mod10 <= 4) return few;
  return many;
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
  return data;
}

function updateRange(input, output, formatter) {
  const min = Number(input.min), max = Number(input.max), value = Number(input.value);
  input.style.setProperty("--fill", `${(value - min) / (max - min) * 100}%`);
  output.value = formatter(value);
}

function renderCatalog(data) {
  state.catalog = data;
  const box = $("#tasks");
  box.textContent = "";
  for (const task of data.tasks) {
    const label = document.createElement("label");
    label.className = "task";
    label.innerHTML = `
      <input type="radio" name="task" value="${task.name}">
      <span class="task__copy">
        <span class="task__name">${task.label}</span>
        <span class="task__note">${task.note}</span>
      </span>
      ${task.cached ? '<span class="task__cache">кеш</span>' : ""}`;
    label.querySelector("input").addEventListener("change", () => {
      state.task = task.name;
      $("#measure").disabled = false;
    });
    box.append(label);
  }
  const service = $("#service-state");
  service.className = `service ${data.busy ? "is-busy" : "is-ready"}`;
  service.querySelector("span").textContent = data.busy
    ? "короткий замер уже идёт"
    : `свободен · ${data.limits.starts_remaining_today} замеров сегодня`;
}

function renderWaiting(source) {
  const sourceText = source === "cache" ? "Читаю готовый замер" : source === "shared_probe" ? "Подключаюсь к текущему замеру" : "Измеряю первые 10 попыток";
  $("#receipt").innerHTML = `
    <div class="receipt__top"><span>Расчёт выполняется</span><span class="mono"># ${state.job.slice(0, 6)}</span></div>
    <div class="probe-state">
      <div class="probe-state__meter"><i></i></div>
      <h2>${sourceText}</h2>
      <p>Страница обновится сама. Новый эксперимент здесь не создаётся: фактический прогон остановится на 10 попытках.</p>
    </div>`;
}

function renderError(message) {
  const receipt = $("#receipt");
  receipt.innerHTML = `
    <div class="receipt__top"><span>Прикидка не готова</span><span class="mono"># ———</span></div>
    <div class="probe-state error"><h2>Не получилось посчитать</h2><p></p></div>`;
  receipt.querySelector(".probe-state p").textContent = String(message);
}

function renderResult(answer) {
  const [lo, hi] = answer.interval;
  const [nLo, nHi] = answer.affordable_attempts;
  const scale = Math.max(answer.predicted_tokens, answer.budget_tokens, hi, 1);
  const estimatePct = Math.min(100, answer.predicted_tokens / scale * 100);
  const budgetPct = Math.min(100, answer.budget_tokens / scale * 100);
  const verdicts = {
    fits: ["Бюджета хватает", "fits"],
    tight: ["Бюджет впритык", "tight"],
    over: ["Бюджета не хватает", "over"],
  };
  const [verdict, cls] = verdicts[answer.verdict] || ["Нужна проверка", "tight"];
  const attemptsText = nLo === nHi
    ? `${nLo} ${plural(nLo, "попытку", "попытки", "попыток")}`
    : `${nLo}–${nHi} попыток`;
  $("#receipt").innerHTML = `
    <div class="receipt__top"><span>Смета токенов</span><span class="mono"># ${state.job.slice(0, 6)}</span></div>
    <div class="result">
      <div class="result__verdict ${cls}">${verdict}</div>
      <p class="result__number">${compact(answer.predicted_tokens)}</p>
      <p class="result__label">на ${answer.target_attempts} попыток мутации</p>
      <div class="budget-track" style="--estimate:${estimatePct}%;--budget:${budgetPct}%">
        <i class="budget-track__estimate"></i><i class="budget-track__budget"></i>
      </div>
      <div class="result__interval"><span>ожидаемый диапазон</span><strong>${compact(lo)} — ${compact(hi)}</strong></div>
      <div class="reach">
        <div class="reach__big">Имеющегося бюджета хватит на ${attemptsText}</div>
        <p>Диапазон использует медиану и верхний квартиль стоимости: левая граница — осторожный сценарий.</p>
        <span class="source">${answer.cache_hit ? "готовый кеш" : `измерено на ${answer.observed_attempts} попытках`}</span>
      </div>
    </div>`;
}

async function poll() {
  try {
    const data = await request(`api/estimate/${state.job}`);
    if (data.answer) renderResult(data.answer);
    if (data.failed || data.timed_out) {
      clearInterval(state.timer);
      renderError(data.timed_out ? "Короткий замер превысил лимит времени и был остановлен." : "Пробный прогон завершился без достаточной телеметрии.");
      $("#measure").disabled = false;
    } else if (!data.alive && data.answer) {
      clearInterval(state.timer);
      $("#measure").disabled = false;
      loadCatalog();
    }
  } catch (error) {
    clearInterval(state.timer);
    renderError(error.message);
    $("#measure").disabled = false;
  }
}

async function start() {
  if (!state.task) return;
  $("#measure").disabled = true;
  clearInterval(state.timer);
  try {
    const data = await request("api/estimate", {
      method: "POST",
      body: JSON.stringify({
        task: state.task,
        attempts: Number($("#attempts").value),
        budget_tokens: Number($("#budget").value),
      }),
    });
    state.job = data.job_id;
    renderWaiting(data.source);
    await poll();
    state.timer = setInterval(poll, 4000);
  } catch (error) {
    renderError(error.message);
    $("#measure").disabled = false;
  }
}

async function loadCatalog() {
  try {
    renderCatalog(await request("api/catalog"));
  } catch (error) {
    renderError("Сервис временно недоступен. Обновите страницу через минуту.");
    $("#service-state span").textContent = "нет соединения";
  }
}

const attempts = $("#attempts"), budget = $("#budget");
attempts.addEventListener("input", () => updateRange(attempts, $("#attempts-out"), (n) => new Intl.NumberFormat("ru-RU").format(n)));
budget.addEventListener("input", () => updateRange(budget, $("#budget-out"), compact));
$("#measure").addEventListener("click", start);
updateRange(attempts, $("#attempts-out"), (n) => new Intl.NumberFormat("ru-RU").format(n));
updateRange(budget, $("#budget-out"), compact);
loadCatalog();
