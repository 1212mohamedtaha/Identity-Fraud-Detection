"""FastAPI application: JSON API under /api and the chat UI at /."""
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..core.engine import Session, SessionError
from ..llm import LLMError, get_llm
from ..packs import PACKS, get_pack
from .store import SessionStore

WEB = Path(__file__).resolve().parent.parent / "web"


# ---------------------------------------------------------------- request / response models
class StartRequest(BaseModel):
    pack: str
    inputs: dict = {}
    policy: Optional[str] = None


class AnswerRequest(BaseModel):
    answer: str


class ChoiceOut(BaseModel):
    id: str
    text: str


class QuestionOut(BaseModel):
    id: str
    text: str
    claim: str
    free_text: bool
    choices: list[ChoiceOut]


class TurnOut(BaseModel):
    question: str
    claim: str
    answer: str
    score: Optional[float] = None       # hidden for packs that do not show feedback
    feedback: Optional[str] = None


class ClaimOut(BaseModel):
    id: str
    text: str
    status: str
    probability: float
    questions: int
    explanation: str


class VerdictOut(BaseModel):
    status: str
    probability: float
    claims: list[ClaimOut]
    notes: list[str]


class SessionOut(BaseModel):
    id: str
    pack: str
    policy: str
    finished: bool
    show_feedback: bool
    asked: int
    max_questions: int
    question: Optional[QuestionOut] = None
    history: list[TurnOut]
    verdict: Optional[VerdictOut] = None


class PackOut(BaseModel):
    name: str
    title: str
    description: str
    fields: list[dict]
    policies: list[str]


# ---------------------------------------------------------------- views
def claim_text(session, claim_id):
    return next((c.text for c in session.state.claims if c.id == claim_id), claim_id)


def answer_text(probe, answer):
    if probe.is_free_text:
        return answer
    return next((c.text for c in probe.choices if c.id == answer), answer)


def session_out(session_id, session):
    pack = session.pack
    probe = session.current
    question = None
    if probe:
        question = QuestionOut(
            id=probe.id, text=probe.question, claim=claim_text(session, probe.claim_id),
            free_text=probe.is_free_text, choices=[ChoiceOut(id=c.id, text=c.text) for c in probe.choices])
    history = []
    for turn in session.history:
        obs = turn.observation
        history.append(TurnOut(
            question=turn.probe.question,
            claim=claim_text(session, turn.probe.claim_id),
            answer=answer_text(turn.probe, obs.answer),
            score=obs.score if pack.show_feedback else None,
            feedback=obs.feedback if pack.show_feedback else None,
        ))
    verdict = None
    if session.verdict:
        v = session.verdict
        verdict = VerdictOut(status=v.status, probability=v.probability, notes=v.notes, claims=[
            ClaimOut(id=r.claim.id, text=r.claim.text, status=r.status, probability=r.probability,
                     questions=r.questions, explanation=r.explanation) for r in v.claims])
    return SessionOut(id=session_id, pack=pack.name, policy=session.policy.name,
                      finished=session.finished, show_feedback=pack.show_feedback,
                      asked=len(session.history), max_questions=session.state.max_questions,
                      question=question, history=history, verdict=verdict)


# ---------------------------------------------------------------- app
def create_app(llm="auto", store=None):
    """Build the app. ``llm="auto"`` reads the provider from the environment (see verity.llm)."""
    llm = get_llm() if llm == "auto" else llm
    packs = {name: get_pack(name, llm=llm) for name in PACKS}
    store = store or SessionStore()

    app = FastAPI(title="Verity", description="Verify claims with adaptive questions.", version="0.1.0")
    app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        if not request.url.path.startswith(("/docs", "/redoc", "/openapi.json")):
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def find(session_id):
        entry = store.get(session_id)
        if entry is None:
            raise HTTPException(404, "Session not found; it may have expired.")
        return entry

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "llm": repr(llm) if llm else "none"}

    @app.get("/api/packs", response_model=list[PackOut])
    def list_packs():
        return [pack.describe() for pack in packs.values()]

    @app.post("/api/sessions", response_model=SessionOut, status_code=201)
    def start_session(body: StartRequest):
        pack = packs.get(body.pack)
        if pack is None:
            raise HTTPException(404, f"Unknown pack {body.pack!r}.")
        try:
            policy = pack.policy(body.policy) if body.policy else None
            session = Session(pack, {k: str(v) for k, v in body.inputs.items()}, policy=policy)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(502, f"The language model failed: {exc}") from exc
        return session_out(store.add(session), session)

    @app.get("/api/sessions/{session_id}", response_model=SessionOut)
    def get_session(session_id: str):
        entry = find(session_id)
        with entry.lock:
            return session_out(session_id, entry.session)

    @app.post("/api/sessions/{session_id}/answer", response_model=SessionOut)
    def answer(session_id: str, body: AnswerRequest):
        entry = find(session_id)
        with entry.lock:
            try:
                entry.session.answer(body.answer)
            except SessionError as exc:
                raise HTTPException(409, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            except LLMError as exc:
                raise HTTPException(502, f"The language model failed: {exc}") from exc
            return session_out(session_id, entry.session)

    return app
