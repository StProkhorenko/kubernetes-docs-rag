"use strict";

const examples = {
  image: "Что означает ImagePullBackOff?",
  readiness: "Процесс жив, но временно не может обслуживать запросы. Как исключить его из трафика сервиса?",
  cron: "Новый запуск по расписанию должен заменить предыдущий, если тот ещё работает. Что настроить?",
  helm: "Как откатить релиз Helm на предыдущую ревизию?",
};
const labels = {bm25: "BM25", qdrant: "E5 + Qdrant", hybrid: "Гибридный", numpy: "E5 + NumPy"};
const hints = {
  image: {section: "images:89", text: "Проверь имя образа и доступ к приватному реестру. В источнике указано, что при отсутствии нужного imagePullSecret загрузка может завершиться ошибкой."},
  readiness: {section: "probes:257", text: "Проверь readiness-пробу. По документации неготовый Pod исключается из получателей трафика Service."},
};
const $ = (selector) => document.querySelector(selector);
const question = $("#question");
const status = $("#status");
const body = $("#results-body");
const timing = $("#timing");
const submit = $("#submit");
const ranking = $("#ranking");
let modes = [];
let view = "search";
let requestNumber = 0;
let controller;

function node(tag, text, className) {
  const result = document.createElement(tag);
  if (text) result.textContent = text;
  if (className) result.className = className;
  return result;
}

function sourceLink(hit, title = "Открыть источник ↗") {
  const link = node("a", title, "source-link");
  const url = new URL(hit.source_url);
  if (url.origin === "https://github.com" && url.pathname.startsWith("/kubernetes/website/blob/")) {
    link.href = url.href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  }
  return link;
}

function notice(text, kind = "") {
  status.textContent = text;
  status.className = ["notice", kind].filter(Boolean).join(" ");
}

function renderSearch(data, query) {
  timing.textContent = `${labels[data.mode]} · ${data.elapsed_ms} мс`;
  if (data.evidence === "insufficient_evidence") {
    notice("Недостаточно подтверждений в корпусе. Ниже похожие фрагменты для самостоятельной проверки.", "warning");
  } else {
    notice("Найдены фрагменты документации. Проверь вывод по первоисточнику.");
    const hintKey = Object.keys(hints).find((key) => examples[key] === query);
    const hint = hints[hintKey];
    const source = hint && data.results.find((hit) => hit.section_id === hint.section);
    if (source) {
      const card = node("aside", "", "compare-rank");
      card.append(node("h3", "Что проверить"), node("p", hint.text), sourceLink(source));
      body.append(card);
    }
  }

  const groups = new Map();
  for (const hit of data.results) {
    if (!groups.has(hit.section_id)) groups.set(hit.section_id, []);
    groups.get(hit.section_id).push(hit);
  }
  let position = 0;
  for (const hits of groups.values()) {
    const hit = hits[0];
    const card = node("article", "", "source-card");
    const header = node("div", "", "source-head");
    const title = node("div", "", "source-title");
    title.append(node("span", String(++position).padStart(2, "0"), "rank"), node("h3", hit.heading));
    header.append(title, sourceLink(hit));
    const excerpt = hit.text.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1").replace(/`/g, "").replace(/\s+/g, " ").trim();
    card.append(header, node("p", excerpt.length > 255 ? excerpt.slice(0, 252) + "…" : excerpt));
    const details = node("details");
    details.append(node("summary", `Фрагменты и версия снимка · ${hit.version.slice(0, 8)}`));
    for (const source of hits) details.append(node("pre", source.text), sourceLink(source));
    card.append(details);
    body.append(card);
  }
  if (!data.results.length) notice("В выбранной выдаче фрагментов нет. Попробуй уточнить запрос.", "warning");
}

function renderCompare(results, query) {
  const target = query === examples.cron ? "cronjobs:114" : null;
  notice("Слева направо — три метода поиска. Ниже по пять фрагментов, а позицию эталона проверяем до десятого места.");
  if (target) {
    const positions = results.map((data) => {
      const position = data.results.findIndex((hit) => hit.section_id === target);
      return `${labels[data.mode]} — ${position < 0 ? "вне топ‑10" : "№ " + (position + 1)}`;
    });
    body.append(node("p", `Эталон · Политика параллелизма CronJob. ${positions.join(" · ")}.`, "compare-rank"));
  }
  const grid = node("div", "", "compare-grid");
  for (const data of results) {
    const column = node("section");
    column.append(node("h3", labels[data.mode]), node("small", `${data.elapsed_ms} мс`));
    const list = node("ol");
    for (const hit of data.results.slice(0, 5)) {
      const item = node("li", "", hit.section_id === target ? "target" : "");
      item.append(sourceLink(hit, hit.heading), node("small", hit.section_id));
      list.append(item);
    }
    column.append(list);
    grid.append(column);
  }
  body.append(grid);
}

async function search(query, mode, signal, comparing) {
  const response = await fetch("/search", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({question: query, limit: comparing ? 10 : 5, mode, ranking: comparing ? ranking.value : undefined}),
    signal,
  });
  if (!response.ok) throw new Error("Поиск временно недоступен. Повтори запрос или проверь состояние индекса.");
  return response.json();
}

async function run() {
  const query = question.value.trim();
  if (!query) {
    notice("Введи симптом или вопрос.", "warning");
    question.focus();
    return;
  }
  controller?.abort();
  controller = new AbortController();
  const id = ++requestNumber;
  const comparing = view === "compare";
  const enabled = comparing ? ["bm25", "qdrant", "hybrid"].filter((mode) => modes.includes(mode)) : [undefined];
  body.replaceChildren();
  timing.textContent = "";
  $("#results-title").textContent = comparing ? "Сравнение выдач" : "Результаты поиска";
  $("#results").setAttribute("aria-busy", "true");
  submit.disabled = true;
  notice(comparing ? "Сравниваем выдачи…" : "Ищем фрагменты…");
  try {
    const results = await Promise.all(enabled.map((mode) => search(query, mode, controller.signal, comparing)));
    if (id !== requestNumber) return;
    if (comparing) renderCompare(results, query);
    else renderSearch(results[0], query);
  } catch (error) {
    if (id === requestNumber && error.name !== "AbortError") notice(error.message, "error");
  } finally {
    if (id === requestNumber) {
      $("#results").setAttribute("aria-busy", "false");
      submit.disabled = false;
    }
  }
}

function setView(next) {
  view = next;
  $("#search-tab").setAttribute("aria-pressed", String(next === "search"));
  $("#compare-tab").setAttribute("aria-pressed", String(next === "compare"));
  $("#compare-settings").hidden = next !== "compare";
  submit.firstChild.textContent = next === "compare" ? "Сравнить " : "Найти ";
}

$("#query-form").addEventListener("submit", (event) => { event.preventDefault(); run(); });
$("#search-tab").addEventListener("click", () => { setView("search"); run(); });
$("#compare-tab").addEventListener("click", () => { setView("compare"); run(); });
ranking.addEventListener("change", run);
for (const example of document.querySelectorAll(".example")) {
  example.addEventListener("click", () => {
    question.value = examples[example.dataset.scenario];
    run();
  });
}
$("#compare-tab").disabled = true;
fetch("/health").then((response) => response.json()).then((data) => {
  modes = data.search_modes || [];
  $("#compare-tab").disabled = modes.filter((mode) => ["bm25", "qdrant", "hybrid"].includes(mode)).length < 2;
}).catch(() => {}).finally(run);
