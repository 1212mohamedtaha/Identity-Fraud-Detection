# Spec: HTTP API and web UI

Code: `verity/api/app.py` (FastAPI), `verity/web/` (UI). The live, machine-readable
schema is at `/openapi.json` and the interactive docs at `/docs`.

Start: `verity serve` (or `uvicorn verity.api.app:create_app --factory`).

## Endpoints

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/` | – | the chat UI |
| GET | `/api/health` | – | `{"status": "ok", "llm": "<provider:model or none>"}` |
| GET | `/api/packs` | – | `[Pack]` |
| POST | `/api/sessions` | `{"pack": str, "inputs": {name: str}, "policy": str?}` | `Session` (201) |
| GET | `/api/sessions/{id}` | – | `Session` |
| POST | `/api/sessions/{id}/answer` | `{"answer": str}` | `Session` |

`answer` is a choice id (`"A"`…) for multiple-choice questions or free text otherwise.

### Pack
`{name, title, description, fields: [{name, label, type, required, placeholder}], policies: [str]}`

### Session
```json
{
  "id": "…", "pack": "cv", "policy": "greedy",
  "finished": false, "show_feedback": true,
  "asked": 1, "max_questions": 12,
  "question": {"id": "sql/2", "text": "…", "claim": "Knows SQL", "free_text": true, "choices": []},
  "history": [{"question": "…", "claim": "Knows SQL", "answer": "…", "score": 0.5, "feedback": "…"}],
  "verdict": null
}
```
When finished, `question` is `null` and `verdict` is
`{status, probability, claims: [{id, text, status, probability, questions, explanation, level, claimed_level}], notes: [str]}`.
`level` (most likely real level) and `claimed_level` are level names for leveled claims, `null`
for yes/no claims (and `level` is `null` for untested claims).

For packs with `show_feedback = false` (identity), `score` and `feedback` are always
`null`, so the API never reveals which answers were right.

### Errors
`{"detail": "<message>"}` with:
- 400: invalid inputs (e.g. empty CV), unknown policy, invalid answer
- 404: unknown pack or session (sessions expire after 1 hour without use)
- 409: answering a finished session
- 502: the LLM provider failed

## Sessions

Kept in memory (`SessionStore`): 1-hour expiry, at most 1000, one request at a time per
session. Run a single server process; restarting the server ends all sessions.
Session ids are random 128-bit tokens; whoever has the id can continue the session.

## Security headers

All responses: `X-Content-Type-Options: nosniff`, `Referrer-Policy: same-origin`.
Everything except `/docs`: `Content-Security-Policy: default-src 'self'; …` (no inline
scripts or styles, no third-party assets). API responses: `Cache-Control: no-store`.

## Web UI

Plain HTML, CSS and JavaScript (`verity/web/index.html`, `static/app.js`, `static/style.css`);
no build step and no dependencies.

1. **Start:** pack cards from `/api/packs`; choosing one shows its `fields` and a
   "Question strategy" select (`policies`).
2. **Chat:** questions as bot messages. Multiple choice: buttons (keys A–D / 1–4).
   Free text: a text box (Enter sends, Shift+Enter adds a line). With `show_feedback`, the
   grader's score and feedback appear after each answer.
3. **Verdict:** overall result, then per claim a status pill, probability bar and explanation,
   plus notes (e.g. the original RL model's own decision).

The session id is kept in `localStorage`, so reloading the page resumes the conversation.
Light and dark themes follow the system setting; layout works from 360 px wide; text
direction is automatic per message (Arabic and English mix correctly).
