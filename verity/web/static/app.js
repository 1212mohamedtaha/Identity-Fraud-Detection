"use strict";

/* Verity chat UI: plain JavaScript, no build step, no dependencies. */

const log = document.getElementById("log");
const choicesEl = document.getElementById("choices");
const promptEl = document.getElementById("prompt");
const noticeEl = document.getElementById("notice");
const statusEl = document.getElementById("status");
const restartBtn = document.getElementById("restart");
const answerForm = document.getElementById("answer-form");
const answerInput = document.getElementById("answer");

const STORAGE_KEY = "verity.session";
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const TYPING_DELAY_MS = reducedMotion ? 0 : 450;
const SHIELD_PATH = "M12 2.5 4.5 5.5v5.7c0 4.6 3.1 8.8 7.5 10.3 4.4-1.5 7.5-5.7 7.5-10.3V5.5L12 2.5Zm-1.2 13.6-3.1-3.1 1.4-1.4 1.7 1.7 4-4 1.4 1.4-5.4 5.4Z";

const VERDICTS = {
  supported: { tone: "ok", title: "The claims hold up" },
  refuted: { tone: "bad", title: "At least one claim did not hold up" },
  uncertain: { tone: "unsure", title: "Not enough evidence yet" },
};
const STATUS_LABEL = { supported: "Supported", refuted: "Refuted", uncertain: "Uncertain" };

let sessionId = null;
let busy = false;

/* ---------- small helpers ---------- */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function botAvatar() {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  const path = document.createElementNS(ns, "path");
  path.setAttribute("d", SHIELD_PATH);
  svg.append(path);
  const avatar = el("span", "avatar small");
  avatar.setAttribute("aria-hidden", "true");
  avatar.append(svg);
  return avatar;
}

function scrollToEnd() {
  log.scrollTop = log.scrollHeight;
}

function addMessage(role, content) {
  const bubble = el("div", "bubble");
  bubble.dir = "auto";
  if (typeof content === "string") bubble.textContent = content;
  else bubble.append(content);
  const row = el("div", `message ${role}`);
  if (role === "bot") row.append(botAvatar());
  row.append(bubble);
  log.append(row);
  scrollToEnd();
  return row;
}

function addTyping() {
  const dots = el("div", "bubble typing");
  dots.setAttribute("aria-label", "Typing");
  dots.append(el("span"), el("span"), el("span"));
  const row = el("div", "message bot");
  row.append(botAvatar(), dots);
  log.append(row);
  scrollToEnd();
  return row;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function showNotice(message) {
  noticeEl.textContent = message || "";
  noticeEl.hidden = !message;
}

function remember(id) {
  sessionId = id;
  try {
    if (id) localStorage.setItem(STORAGE_KEY, id);
    else localStorage.removeItem(STORAGE_KEY);
  } catch (e) {
    /* storage may be unavailable (private mode); the app still works */
  }
}

function storedSession() {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch (e) {
    return null;
  }
}

async function api(path, method = "GET", body) {
  const response = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.detail || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return data;
}

/* ---------- composer ---------- */

function clearComposer() {
  choicesEl.replaceChildren();
  promptEl.hidden = true;
  answerForm.hidden = true;
  answerInput.value = "";
}

function showQuestionInput(question) {
  clearComposer();
  if (question.free_text) {
    answerForm.hidden = false;
    answerInput.focus({ preventScroll: true });
    return;
  }
  choicesEl.replaceChildren(
    ...question.choices.map((choice) => {
      const button = el("button", "choice");
      button.type = "button";
      if (choice.text === "Not Sure") button.classList.add("muted");
      const text = el("span", "text", choice.text);
      text.dir = "auto";
      button.append(el("span", "key", choice.id), text);
      button.addEventListener("click", () => sendAnswer(choice.id, choice.text));
      return button;
    }),
  );
  promptEl.hidden = false;
  choicesEl.querySelector("button")?.focus({ preventScroll: true });
}

function setComposerDisabled(disabled) {
  choicesEl.querySelectorAll("button").forEach((b) => (b.disabled = disabled));
  answerForm.querySelectorAll("textarea, button").forEach((x) => (x.disabled = disabled));
}

/* ---------- rendering a session ---------- */

function feedbackNode(turn) {
  const wrap = el("div", "feedback");
  const score = el("span", "score", `Score ${Math.round(turn.score * 100)}%. `);
  wrap.append(score, document.createTextNode(turn.feedback || ""));
  return wrap;
}

function renderVerdict(view) {
  const verdict = view.verdict;
  const config = VERDICTS[verdict.status];
  const card = el("section", `result ${config.tone}`);
  card.setAttribute("role", "status");
  card.append(el("h2", null, config.title));
  card.append(el("p", null, `Asked ${view.asked} question${view.asked === 1 ? "" : "s"}.`));

  const list = el("ul", "claims");
  for (const claim of verdict.claims) {
    const item = el("li", "claim");
    const head = el("div", "claim-head");
    const text = el("span", "claim-text", claim.text);
    text.dir = "auto";
    head.append(text, el("span", `pill ${claim.status}`, STATUS_LABEL[claim.status]));
    const bar = el("div", "bar");
    const fill = el("span");
    fill.style.setProperty("--value", `${Math.round(claim.probability * 100)}%`);
    bar.append(fill);
    bar.setAttribute("aria-label", `${Math.round(claim.probability * 100)}% likely true`);
    item.append(head, bar, el("small", null, claim.explanation));
    list.append(item);
  }
  card.append(list);
  for (const note of verdict.notes) card.append(el("p", "notes", note));

  const again = el("button", "btn", "New check");
  again.type = "button";
  again.addEventListener("click", showStart);
  card.append(again);
  log.append(card);
  scrollToEnd();
  again.focus({ preventScroll: true });
}

async function showNext(view, animate) {
  clearComposer();
  if (view.finished) {
    statusEl.textContent = "Complete";
    renderVerdict(view);
    return;
  }
  statusEl.textContent = `Question ${view.asked + 1} of up to ${view.max_questions}`;
  if (animate && TYPING_DELAY_MS) {
    const typing = addTyping();
    await sleep(TYPING_DELAY_MS);
    typing.remove();
  }
  addMessage("bot", view.question.text);
  showQuestionInput(view.question);
}

function renderTranscript(view) {
  log.replaceChildren();
  for (const turn of view.history) {
    addMessage("bot", turn.question);
    addMessage("user", turn.answer);
    if (view.show_feedback && turn.feedback !== null) addMessage("bot", feedbackNode(turn));
  }
}

/* ---------- actions ---------- */

async function sendAnswer(answer, shownText) {
  if (busy || !answer.trim()) return;
  busy = true;
  showNotice("");
  setComposerDisabled(true);
  addMessage("user", shownText || answer);
  try {
    const view = await api(`/api/sessions/${sessionId}/answer`, "POST", { answer });
    const last = view.history[view.history.length - 1];
    if (view.show_feedback && last && last.feedback !== null) addMessage("bot", feedbackNode(last));
    await showNext(view, true);
  } catch (error) {
    if (error.status === 404) {
      showNotice("This session has expired. Start a new check.");
      clearComposer();
      remember(null);
    } else {
      showNotice(error.message);
      setComposerDisabled(false);
    }
  } finally {
    busy = false;
    setComposerDisabled(false);
  }
}

async function startSession(pack, form) {
  if (busy) return;
  busy = true;
  showNotice("");
  const inputs = {};
  for (const field of pack.fields) inputs[field.name] = form.elements[field.name].value;
  const policy = form.elements.policy.value;
  const submit = form.querySelector("button[type=submit]");
  submit.disabled = true;
  submit.textContent = "Preparing questions…";
  try {
    const view = await api("/api/sessions", "POST", { pack: pack.name, inputs, policy });
    remember(view.id);
    log.replaceChildren();
    restartBtn.hidden = false;
    addMessage("bot", `Let's check: ${pack.title}. Answer as well as you can; “Not Sure” is fine.`);
    await showNext(view, true);
  } catch (error) {
    showNotice(error.message);
    submit.disabled = false;
    submit.textContent = "Start";
  } finally {
    busy = false;
  }
}

function showForm(pack) {
  const form = el("form", "form");
  form.append(el("h2", null, pack.title), el("p", "intro", pack.description));
  for (const field of pack.fields) {
    const label = el("label", "field", field.label);
    const input = el(field.type === "textarea" ? "textarea" : "input");
    input.name = field.name;
    input.required = Boolean(field.required);
    input.placeholder = field.placeholder || "";
    input.dir = "auto";
    label.append(input);
    form.append(label);
  }
  const policyLabel = el("label", "field", "Question strategy");
  const select = el("select");
  select.name = "policy";
  for (const name of pack.policies) select.append(new Option(name, name));
  policyLabel.append(select);
  form.append(policyLabel);

  const actions = el("div", "form-actions");
  const start = el("button", "btn", "Start");
  start.type = "submit";
  const back = el("button", "btn btn-ghost", "Back");
  back.type = "button";
  back.addEventListener("click", showStart);
  actions.append(start, back);
  form.append(actions);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    startSession(pack, form);
  });

  log.replaceChildren(form);
  form.querySelector("textarea, input, select")?.focus();
}

async function showStart() {
  remember(null);
  clearComposer();
  showNotice("");
  restartBtn.hidden = true;
  statusEl.textContent = "Choose what to verify";
  log.replaceChildren(el("p", "intro", "Loading…"));
  try {
    const packs = await api("/api/packs");
    const list = el("div", "packs");
    for (const pack of packs) {
      const card = el("button", "pack");
      card.type = "button";
      card.append(el("strong", null, pack.title), el("span", null, pack.description));
      card.addEventListener("click", () => showForm(pack));
      list.append(card);
    }
    log.replaceChildren(el("p", "intro", "What would you like to verify?"), list);
  } catch (error) {
    log.replaceChildren();
    showNotice(`Could not load: ${error.message}`);
  }
}

async function init() {
  const stored = storedSession();
  if (!stored) return showStart();
  try {
    const view = await api(`/api/sessions/${stored}`);
    sessionId = stored;
    restartBtn.hidden = false;
    renderTranscript(view);
    await showNext(view, false);
  } catch (error) {
    showStart();
  }
}

/* ---------- events ---------- */

restartBtn.addEventListener("click", showStart);

answerForm.addEventListener("submit", (event) => {
  event.preventDefault();
  sendAnswer(answerInput.value);
});

answerInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    answerForm.requestSubmit();
  }
});

const SHORTCUTS = { a: 0, b: 1, c: 2, d: 3, 1: 0, 2: 1, 3: 2, 4: 3 };
document.addEventListener("keydown", (event) => {
  if (busy || event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.target.closest && event.target.closest("textarea, input, select")) return;
  const index = SHORTCUTS[event.key.toLowerCase()];
  const button = index === undefined ? null : choicesEl.querySelectorAll("button:not(:disabled)")[index];
  if (button) {
    event.preventDefault();
    button.click();
  }
});

init();
