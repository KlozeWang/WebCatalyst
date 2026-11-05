"use strict";

const $ = (id) => document.getElementById(id);
const state = {dataset: null, current: null, filtered: [], detail: {}, renderVersion: 0, focusModel: null, lastFocus: null};
let toastTimer;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function api(path) {
  const response = await fetch(path, {cache: "no-store"});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function toast(message, error = false) {
  $("toast").textContent = message;
  $("toast").classList.toggle("error", error);
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, error ? 8000 : 3500);
}

function setLoadStatus(message, mode = "") {
  $("load-status").textContent = message;
  $("load-status").className = `load-status ${mode}`;
}

function matches(prompt) {
  const query = $("prompt-search").value.trim().toLocaleLowerCase();
  const searchMatch = !query || `${prompt.id} ${prompt.index + 1} ${prompt.prompt}`.toLocaleLowerCase().includes(query);
  return searchMatch;
}

function renderChoices() {
  state.filtered = state.dataset.prompts.filter(matches);
  const choices = [...state.filtered];
  $("prompt-select").replaceChildren(...choices.map((prompt) => {
    const option = document.createElement("option");
    option.value = prompt.key;
    const excerpt = prompt.prompt.replace(/\s+/g, " ").slice(0, 110);
    option.textContent = `${String(prompt.index + 1).padStart(3, "0")} · ID ${prompt.id ?? "—"} · ${excerpt}${prompt.prompt.length > 110 ? "…" : ""}`;
    option.selected = prompt === state.current;
    return option;
  }));
  if (!choices.length) $("prompt-select").append(new Option("没有匹配的 Prompt", ""));
  $("prompt-select").disabled = !choices.length;
  const index = state.filtered.indexOf(state.current);
  $("prompt-position").textContent = state.current ? (index >= 0 ? `${index + 1} / ${state.filtered.length}` : `剩余 ${state.filtered.length} 项`) : `0 / ${state.filtered.length}`;
  $("prev-prompt").disabled = index <= 0;
  $("next-prompt").disabled = !state.filtered.length || index === state.filtered.length - 1;
}

function applyFilters() {
  const list = state.dataset.prompts.filter(matches);
  if (!list.includes(state.current)) selectPrompt(list[0]?.key ?? null);
  else renderChoices();
}

function movePrompt(direction) {
  const index = state.filtered.indexOf(state.current);
  const next = state.filtered[index < 0 ? 0 : index + direction];
  if (next) selectPrompt(next.key);
}

function remember(key) {
  try { localStorage.setItem(`gamebench:${state.dataset.dataset_id}:prompt`, key); } catch { /* Optional navigation preference. */ }
}

async function selectPrompt(key) {
  closeFocus();
  clearStages($("comparison-board"));
  $("reference-column").replaceChildren();
  $("opponent-grid").replaceChildren();
  state.current = state.dataset.prompts.find((prompt) => prompt.key === key) || null;
  state.detail = {};
  const version = ++state.renderVersion;
  renderChoices();
  $("empty-state").hidden = Boolean(state.current);
  $("comparison-board").hidden = !state.current;
  $("prompt-text").classList.remove("expanded");
  $("toggle-prompt").textContent = "展开完整 Prompt ↓";
  $("toggle-prompt").hidden = true;
  if (!state.current) {
    $("prompt-text").textContent = "没有匹配的数据，请调整搜索条件。";
    $("prompt-id").textContent = "PROMPT";
    $("current-models").textContent = "—";
    return;
  }
  remember(key);
  $("prompt-text").textContent = state.current.prompt;
  $("prompt-id").textContent = `PROMPT ${String(state.current.index + 1).padStart(3, "0")} / ID ${state.current.id ?? "—"}`;
  $("current-models").textContent = `${state.current.models.length} 个模型有可用 HTML`;
  requestAnimationFrame(() => { $("toggle-prompt").hidden = $("prompt-text").scrollHeight <= 95; });
  try {
    const detail = await api(`/api/prompts/${key}`);
    if (version !== state.renderVersion) return;
    state.detail = detail;
    for (const model of state.dataset.models) {
      const card = createModelCard(model);
      (model === state.dataset.reference_model ? $("reference-column") : $("opponent-grid")).append(card);
    }
    const note = el("p", "card-note", "参照模型固定于此，滚动右侧即可逐个比较。");
    $("reference-column").append(note);
  } catch (error) {
    if (version !== state.renderVersion) return;
    const errorBox = el("div", "empty-state");
    errorBox.append(el("p", "", `加载失败：${error.message}`));
    const retry = el("button", "button", "重新加载");
    retry.addEventListener("click", () => selectPrompt(key));
    errorBox.append(retry);
    $("reference-column").append(errorBox);
    toast(`无法读取此 Prompt：${error.message}`, true);
  }
}

function viewport() { return $("viewport-select").value.split("x").map(Number); }

const stageObserver = new ResizeObserver((entries) => entries.forEach(({target}) => fitStage(target)));

function fitStage(stage) {
  const canvas = stage.querySelector(".game-canvas");
  if (!canvas || !stage.clientWidth || !stage.clientHeight) return;
  const [width, height] = viewport();
  const zoom = stage.dataset.focus ? $("zoom-select").value : "fit";
  const scale = zoom === "fit" ? Math.min(stage.clientWidth / width, stage.clientHeight / height) : Number(zoom);
  canvas.style.width = `${width * scale}px`;
  canvas.style.height = `${height * scale}px`;
  const frame = canvas.querySelector("iframe");
  frame.style.width = `${width}px`;
  frame.style.height = `${height}px`;
  frame.style.transform = `scale(${scale})`;
}

function clearStages(container) {
  container.querySelectorAll(".game-stage").forEach((stage) => stageObserver.unobserve(stage));
}

function makeStage(model, focus = false) {
  const stage = el("div", "game-stage");
  stage.dataset.model = model;
  if (focus) stage.dataset.focus = "true";
  if (!state.detail[model]?.available) {
    const detail = state.detail[model];
    stage.append(el("div", "frame-failure", `此模型没有可用 HTML${detail?.error ? `：${detail.error}` : ""}`));
    return stage;
  }
  const canvas = el("div", "game-canvas");
  const frame = document.createElement("iframe");
  frame.title = `${model} · Prompt ${state.current.index + 1}`;
  // Games use a separate origin so their scripts cannot access the viewer.
  // allow-same-origin lets each game use browser storage.
  frame.setAttribute("sandbox", "allow-scripts allow-same-origin allow-pointer-lock");
  frame.setAttribute("allow", "autoplay; fullscreen; gamepad");
  frame.setAttribute("referrerpolicy", "no-referrer");
  const url = new URL(location.href);
  url.port = state.dataset.preview_port;
  url.pathname = `/games/${state.current.key}/${encodeURIComponent(model)}`;
  url.search = "";
  url.hash = "";
  frame.src = url.href;
  canvas.append(frame);
  stage.append(canvas);
  stageObserver.observe(stage);
  return stage;
}

function iconButton(text, title, action) {
  const button = el("button", "icon-button", text);
  button.title = title;
  button.setAttribute("aria-label", title);
  button.addEventListener("click", action);
  return button;
}

function createModelCard(model) {
  const isReference = model === state.dataset.reference_model;
  const card = el("article", `model-card${isReference ? " reference" : ""}`);
  card.dataset.model = model;
  const header = el("div", "model-header");
  const name = el("div", "model-name-group");
  name.append(el("span", "model-kind", isReference ? "● REFERENCE / 参照模型" : `COMPARISON ${String(state.dataset.models.indexOf(model)).padStart(2, "0")}`), el("h3", "model-name", model));
  name.querySelector("h3").style.margin = "0";
  const tools = el("div", "model-tools");
  tools.append(iconButton("↻", `重新加载 ${model}`, () => reloadStage(card)), iconButton("⤢", `放大 ${model}`, () => openFocus(model)));
  header.append(name, tools);
  card.append(header, makeStage(model));
  return card;
}

function reloadStage(container) {
  container.querySelectorAll("iframe").forEach((frame) => { frame.src = frame.src; });
}

function openFocus(model) {
  state.lastFocus = document.activeElement;
  state.focusModel = model;
  $("zoom-select").value = "fit";
  const reference = state.dataset.reference_model;
  const models = model === reference ? [reference] : [reference, model];
  $("dialog-title").textContent = model === reference ? "参照模型 · 放大预览" : "并排体验";
  $("dialog-games").classList.toggle("single", models.length === 1);
  clearStages($("dialog-games"));
  $("dialog-games").replaceChildren(...models.map((name) => {
    const panel = el("div", `focus-game${name === reference ? " reference" : ""}`);
    const title = document.createElement("div");
    title.append(el("span", "pill", name === reference ? "参照模型" : "对比模型"), el("strong", "", name));
    title.append(iconButton("↻", `重新加载放大视图中的 ${name}`, () => reloadStage(panel)));
    panel.append(title, makeStage(name, true));
    return panel;
  }));
  $("dialog-description").textContent = model === reference ? "可调整缩放比例；Esc 关闭。" : "左侧为参照模型，右侧为对比模型 · Esc 关闭";
  $("preview-dialog").showModal();
  document.body.style.overflow = "hidden";
}

function closeFocus() {
  if (!$("preview-dialog").open) return;
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  $("preview-dialog").close();
  cleanFocus();
}

function cleanFocus() {
  clearStages($("dialog-games"));
  $("dialog-games").replaceChildren();
  document.body.style.overflow = "";
  state.focusModel = null;
  if (state.lastFocus?.isConnected) state.lastFocus.focus();
  state.lastFocus = null;
}

function bindEvents() {
  $("prompt-search").addEventListener("input", applyFilters);
  $("prompt-select").addEventListener("change", (event) => selectPrompt(event.target.value));
  $("prev-prompt").addEventListener("click", () => movePrompt(-1));
  $("next-prompt").addEventListener("click", () => movePrompt(1));
  $("toggle-prompt").addEventListener("click", () => { const expanded = $("prompt-text").classList.toggle("expanded"); $("toggle-prompt").textContent = expanded ? "收起 Prompt ↑" : "展开完整 Prompt ↓"; });
  $("viewport-select").addEventListener("change", () => document.querySelectorAll(".game-stage").forEach(fitStage));
  $("zoom-select").addEventListener("change", () => $("dialog-games").querySelectorAll(".game-stage").forEach(fitStage));
  $("close-dialog").addEventListener("click", closeFocus);
  $("preview-dialog").addEventListener("cancel", (event) => { event.preventDefault(); closeFocus(); });
  $("dialog-fullscreen").addEventListener("click", async () => {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else await $("preview-dialog").requestFullscreen(); }
    catch { toast("浏览器暂不支持全屏，请使用当前放大窗口。"); }
  });
}

async function init() {
  try {
    if (location.protocol === "file:") throw new Error("请先运行 python3 server.py，再打开 http://127.0.0.1:8765；不要直接双击 HTML 文件。");
    state.dataset = await api("/api/dataset");
    $("model-count").textContent = `${state.dataset.models.length} 个模型`;
    setLoadStatus(`${state.dataset.prompts.length} 个 Prompt · ${state.dataset.models.length} 个模型`);
    bindEvents();
    let remembered;
    try { remembered = localStorage.getItem(`gamebench:${state.dataset.dataset_id}:prompt`); } catch { /* Optional navigation preference. */ }
    const prompt = state.dataset.prompts.find((item) => item.key === remembered) || state.dataset.prompts[0];
    await selectPrompt(prompt?.key ?? null);
  } catch (error) {
    setLoadStatus("连接失败", "error");
    $("fatal-error").hidden = false;
    $("fatal-error").textContent = `无法加载游戏展示：${error.message}\n请确认 server.py 正在运行，然后刷新页面。`;
    $("prompt-text").textContent = "数据加载失败。请查看下方错误信息。";
    document.querySelectorAll("button, input, select").forEach((node) => { node.disabled = true; });
  }
}

init();
