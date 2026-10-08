"""The LLM interface every provider implements, plus small helpers to use it."""
import json
import logging
import re
import time
from pathlib import Path

log = logging.getLogger("verity.llm")


class LLMError(Exception):
    """The model call failed or returned something unusable."""


class LLM:
    """Send a system prompt and a user prompt, get text back. That is the whole interface."""

    provider = "base"
    model = ""

    def complete(self, system, prompt, max_tokens=4000):
        raise NotImplementedError

    def __repr__(self):
        return f"<{type(self).__name__} {self.provider}:{self.model}>"


def load_prompt(path, **values):
    """Read a prompt file and fill ``{{name}}`` placeholders.

    Prompts live in .md files next to the code that uses them, so they can be reviewed
    and versioned like code. The file name carries the version, e.g. grade_answer.v1.md.
    """
    text = Path(path).read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


def split_prompt(text):
    """Prompt files hold a system part and a user part separated by a line '---'."""
    system, _, user = text.partition("\n---\n")
    return system.strip(), user.strip()


def extract_json(text):
    """Pull the first JSON object or array out of a model reply (tolerates ``` fences)."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise LLMError("No JSON found in the model reply.")
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"Invalid JSON in the model reply: {exc}") from exc


def ask_json(llm, prompt_path, **values):
    """Run a prompt file and parse the JSON reply; retries once if the JSON is broken."""
    system, user = split_prompt(load_prompt(prompt_path, **values))
    name = Path(prompt_path).name
    for attempt in (1, 2):
        started = time.monotonic()
        reply = llm.complete(system, user)
        log.info("llm call prompt=%s model=%s attempt=%d seconds=%.1f",
                 name, llm.model, attempt, time.monotonic() - started)
        try:
            return extract_json(reply)
        except LLMError:
            if attempt == 2:
                raise
            user = user + "\n\nYour previous reply was not valid JSON. Reply with JSON only."
