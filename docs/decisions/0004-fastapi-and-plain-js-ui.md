# 0004: FastAPI for the API, plain HTML/JS for the UI

**Status:** accepted

## Context
The first version used Flask with HTML forms. We need a JSON API that other programs and AI
agents can call, typed request/response models, and a UI anyone can change.

## Decision
- **FastAPI** with Pydantic models: validation and an OpenAPI schema (`/openapi.json`,
  `/docs`) for free, so tools and agents can discover the API.
- **Plain HTML, CSS and JavaScript** for the chat UI: no build step, no npm, no framework.
- Sessions live in memory with expiry (single server process).

## Consequences
- Easy to run and to change; the API is self-documenting.
- Multiple server processes or restarts lose sessions; a shared store (e.g. Redis) would be
  needed for production.
