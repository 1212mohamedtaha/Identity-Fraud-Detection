"""A scripted LLM for tests and demos: no network, fully predictable."""
from .base import LLM


class FakeLLM(LLM):
    """Returns canned replies.

    ``replies`` is either a list (returned in order) or a function
    ``reply(system, prompt) -> str`` for replies that depend on the prompt.
    Every call is recorded in ``calls`` so tests can inspect the prompts.
    """

    provider = "fake"
    model = "fake"

    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def complete(self, system, prompt, max_tokens=4000):
        self.calls.append((system, prompt))
        if callable(self.replies):
            return self.replies(system, prompt)
        return self.replies.pop(0)
