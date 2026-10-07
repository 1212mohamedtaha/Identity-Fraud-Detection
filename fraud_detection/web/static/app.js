"use strict";

const log = document.getElementById("log");
const choicesEl = document.getElementById("choices");
const promptEl = document.getElementById("prompt");
const noticeEl = document.getElementById("notice");
const statusEl = document.getElementById("status");
const restartBtn = document.getElementById("restart");

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const TYPING_DELAY_MS = reducedMotion ? 0 : 500;
const SHIELD_PATH = "M12 2.5 4.5 5.5v5.7c0 4.6 3.1 8.8 7.5 10.3 4.4-1.5 7.5-5.7 7.5-10.3V5.5L12 2.5Zm-1.2 13.6-3.1-3.1 1.4-1.4 1.7 1.7 4-4 1.4 1.4-5.4 5.4Z";

const RESULTS = {
  non_fraud: {
    tone: "ok",
    title: "Identity verified",
    text: (t) => `Your answers matched our records (${t.matched} of ${t.answered}). You're all set.`,
  },
  fraud: {
    tone: "bad",
    title: "We couldn't verify your identity",
    text: (t) => `Only ${t.matched} of ${t.answered} answers matched our records. Your application has been flagged for manual review.`,
  },
};

let busy = false;

/* ---------- helpers ---------- */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function botAvatar() {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
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

function addBotMessage(text) {
  const bubble = el("div", "bubble", text);
  bubble.dir = "auto";
  const row = el("div", "message bot");
  row.append(botAvatar(), bubble);
  log.append(row);
  scrollToEnd();
  return row;
}

function addUserMessage(text) {
  const bubble = el("div", "bubble", text);
  bubble.dir = "auto";
  const row = el("div", "message user");
  row.append(bubble);
  log.append(row);
  scrollToEnd();
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
  noticeEl.textContent = message;
  noticeEl.hidden = !message;
}

function setStatus(text) {
  statusEl.textContent = text;
}

async function api(path, method = "GET", body) {
  const response = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
    credentials: "same-origin",
  });
  if (response.status === 204) return null;
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.error || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return data;
}

/* ---------- rendering ---------- */

function clearChoices() {
  choicesEl.replaceChildren();
  promptEl.hidden = true;
}

function renderChoices(choices) {
  choicesEl.replaceChildren(
    ...choices.map((choice) => {
      const button = el("button", "choice");
      button.type = "button";
      button.dataset.id = choice.id;
      if (choice.text === "Not Sure") button.classList.add("muted");
      const text = el("span", "text", choice.text);
      text.dir = "auto";
      button.append(el("span", "key", choice.id), text);
      button.addEventListener("click", () => choose(choice));
      return button;
    }),
  );
  promptEl.hidden = false;
}

function renderResult(turn) {
  const config = RESULTS[turn.status];
  const card = el("section", `result ${config.tone}`);
  card.setAttribute("role", "status");
  const again = el("button", "btn", "Start over");
  again.type = "button";
  again.addEventListener("click", startDialogue);
  card.append(el("h2", null, config.title), el("p", null, config.text(turn)), again);
  log.append(card);
  scrollToEnd();
  again.focus({ preventScroll: true });
}

async function showTurn(turn, { animate = true } = {}) {
  clearChoices();
  if (turn.status === "question") {
    setStatus(`Question ${turn.answered + 1}`);
    if (animate && TYPING_DELAY_MS) {
      const typing = addTyping();
      await sleep(TYPING_DELAY_MS);
      typing.remove();
    }
    addBotMessage(turn.question);
    renderChoices(turn.choices);
    choicesEl.querySelector("button")?.focus({ preventScroll: true });
  } else {
    setStatus("Complete");
    renderResult(turn);
  }
}

function renderHistory(history) {
  for (const item of history) {
    addBotMessage(item.question);
    const picked = item.choices.find((c) => c.id === item.selected);
    addUserMessage(picked ? picked.text : item.selected);
  }
}

/* ---------- actions ---------- */

async function choose(choice) {
  if (busy) return;
  busy = true;
  showNotice("");
  choicesEl.querySelectorAll("button").forEach((b) => (b.disabled = true));
  addUserMessage(choice.text);
  try {
    const { turn } = await api("/api/dialogue/answer", "POST", { choice: choice.id });
    await showTurn(turn);
  } catch (error) {
    if (error.status === 404) {
      showNotice("Your session has expired. Please start over.");
      clearChoices();
    } else {
      showNotice(error.message);
      choicesEl.querySelectorAll("button").forEach((b) => (b.disabled = false));
    }
  } finally {
    busy = false;
  }
}

function greet() {
  addBotMessage("Hi! I'll ask you a few short questions to verify your identity. Pick the answer that fits best, or “Not Sure” if you don't know.");
}

async function startDialogue() {
  if (busy) return;
  busy = true;
  restartBtn.disabled = true;
  showNotice("");
  clearChoices();
  log.replaceChildren();
  setStatus("Getting ready…");
  greet();
  try {
    const { turn } = await api("/api/dialogue", "POST");
    await showTurn(turn);
  } catch (error) {
    setStatus("Unavailable");
    showNotice(`Couldn't start the verification: ${error.message}`);
  } finally {
    busy = false;
    restartBtn.disabled = false;
  }
}

async function resumeOrStart() {
  try {
    const current = await api("/api/dialogue");
    if (!current) return startDialogue();
    const { turn, history } = current;
    greet();
    renderHistory(history);
    restartBtn.disabled = false;
    await showTurn(turn, { animate: false });
  } catch (error) {
    setStatus("Unavailable");
    showNotice(error.message);
  }
}

restartBtn.addEventListener("click", startDialogue);

const SHORTCUTS = { a: 0, b: 1, c: 2, d: 3, 1: 0, 2: 1, 3: 2, 4: 3 };

document.addEventListener("keydown", (event) => {
  if (busy || event.ctrlKey || event.metaKey || event.altKey) return;
  const index = SHORTCUTS[event.key.toLowerCase()];
  const button = index === undefined ? null : choicesEl.querySelectorAll("button:not(:disabled)")[index];
  if (button) {
    event.preventDefault();
    button.click();
  }
});

resumeOrStart();
