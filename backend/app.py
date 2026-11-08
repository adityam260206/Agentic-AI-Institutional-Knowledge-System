"""
backend/app.py
---------------
FastAPI application — EduMind Institutional Knowledge Engine v3.0

Endpoints:
  GET  /                          — serve frontend index.html
  GET  /styles.css, /app.js       — serve static assets

  POST /api/auth/login            — JWT login
  POST /api/chat                  — public chat (no auth, role=Public)
  POST /api/chat/auth             — authenticated chat (saves history)
  GET  /api/chat/stream           — SSE streaming, public
  GET  /api/chat/auth/stream      — SSE streaming, authenticated

  GET  /api/history               — list sessions (auth)
  GET  /api/history/{id}          — get messages in session (auth)
  DELETE /api/history/{id}        — delete session (auth)

  POST /api/upload                — Admin doc upload → real ingestion pipeline
  GET  /api/users                 — list users (Admin)

  GET  /api/admin/stats           — doc + vector stats (Admin)
  GET  /api/admin/documents       — full document list (Admin)
  GET  /api/admin/logs            — ingestion audit log (Admin)
  GET  /api/admin/chroma          — ChromaDB vector count (Admin)

  POST /api/agent/chat            — LangGraph agentic chat with memory (auth)
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.auth import create_access_token, get_current_user, verify_password
from backend.database import ChatMessage, User, get_db, init_db

logger = logging.getLogger(__name__)

app = FastAPI(title="EduMind AI", version="3.0.0", description="VIT Institutional Knowledge Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ── Serve frontend ─────────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


# Static frontend assets are served with `Cache-Control: no-cache` so the browser
# always revalidates (via ETag) and picks up new JS/CSS immediately. Without this,
# browsers heuristically cache app.js and can keep running a stale version — which
# is how inline-handler functions like openDoc end up "undefined" after an update.
_NO_CACHE = {"Cache-Control": "no-cache"}


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return HTMLResponse(
        content=(FRONTEND_DIR / "index.html").read_text(encoding="utf-8"),
        headers=_NO_CACHE,
    )


@app.get("/styles.css")
def serve_css():
    return FileResponse(FRONTEND_DIR / "styles.css", media_type="text/css", headers=_NO_CACHE)


@app.get("/app.js")
def serve_js():
    return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript", headers=_NO_CACHE)


# ── Schemas ────────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    use_agent: bool = False        # set True to route through LangGraph agent


class AgentChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    memory: Optional[list] = None  # client may pass conversation memory


# ── Auth ───────────────────────────────────────────────────────────────────────
@app.post("/api/auth/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token({"sub": user.username, "role": user.role})
    return {
        "access_token": token,
        "token_type":   "bearer",
        "role":         user.role,
        "username":     user.username,
    }


# ── Public chat (no auth) ──────────────────────────────────────────────────────
@app.post("/api/chat")
def chat(payload: ChatRequest, db: Session = Depends(get_db)):
    """Public chat — always uses role=Public."""
    from backend.rag_integration import query as rag_query
    sid    = payload.session_id or str(uuid.uuid4())
    result = rag_query(question=payload.query, role="Public")
    return {**result, "session_id": sid}


# ── Authenticated chat (saves history) ────────────────────────────────────────
@app.post("/api/chat/auth")
def chat_authenticated(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Authenticated chat — uses the user's RBAC role, saves history."""
    from backend.rag_integration import query as rag_query
    sid = payload.session_id or str(uuid.uuid4())

    db.add(ChatMessage(
        username=current_user.username, session_id=sid,
        content=payload.query, is_user=True, sources=None,
    ))

    result = rag_query(question=payload.query, role=current_user.role)

    db.add(ChatMessage(
        username=current_user.username, session_id=sid,
        content=result["answer"], is_user=False,
        sources=json.dumps(result["source_documents"]),
    ))
    db.commit()

    return {**result, "session_id": sid}


# ── SSE Streaming chat ─────────────────────────────────────────────────────────
def _sse_public_generator(question: str, role: str):
    """
    True SSE streaming for public (unauthenticated) chat — no history saved.

    Streams tokens live and emits a final [META] event with citations so the
    frontend can render sources for guest sessions exactly as it does for
    authenticated users. Uses the same structured stream as the auth path.
    """
    from backend.rag_integration import stream_structured

    final_meta: dict = {}
    try:
        for kind, payload in stream_structured(question=question, role=role):
            if kind == "token":
                safe = payload.replace("\n", "\\n")
                yield f"data: {safe}\n\n"
            elif kind == "meta":
                final_meta = payload
    except Exception as exc:
        yield f"data: [ERROR: {str(exc)[:200]}]\n\n"
        yield "data: [DONE]\n\n"
        return

    meta = json.dumps({
        "source_documents":   final_meta.get("source_documents", []),
        "citations":          final_meta.get("citations", []),
        "confidence":         final_meta.get("confidence", "UNKNOWN"),
        "confidence_score":   final_meta.get("confidence_score", 0.0),
        "retrieval_mode":     final_meta.get("retrieval_mode", ""),
        "processing_time_ms": final_meta.get("processing_time_ms", 0.0),
    })
    yield f"data: [META]{meta}\n\n"
    yield "data: [DONE]\n\n"


def _sse_auth_generator(question: str, role: str, username: str, session_id: str):
    """
    Authenticated SSE: streams Qwen tokens LIVE, then saves history and emits
    citations/confidence as a final metadata event.

    Previously this blocked on a full non-streaming pipeline run before sending
    anything — on slow hardware that exceeded Ollama's read timeout, so the
    client only ever saw a blinking cursor that never resolved.  We now consume
    the pipeline's structured stream so the first token paints in seconds and
    the read timeout is reset by every chunk.

    Protocol (unchanged):
      data: <text chunk>              — tokens, as they arrive
      data: [META]{"answer":...}      — final metadata JSON (citations, confidence)
      data: [DONE]                    — stream complete
    """
    from backend.rag_integration import stream_structured
    from backend.database import SessionLocal, ChatMessage

    final_meta: dict = {}
    try:
        for kind, payload in stream_structured(question=question, role=role):
            if kind == "token":
                safe = payload.replace("\n", "\\n")
                yield f"data: {safe}\n\n"
            elif kind == "meta":
                final_meta = payload
    except Exception as exc:
        yield f"data: [ERROR: {str(exc)[:200]}]\n\n"
        yield "data: [DONE]\n\n"
        return

    # Persist to history DB (full accumulated answer from the meta payload)
    try:
        db = SessionLocal()
        db.add(ChatMessage(
            username=username, session_id=session_id,
            content=question, is_user=True, sources=None,
        ))
        db.add(ChatMessage(
            username=username, session_id=session_id,
            content=final_meta.get("answer", ""), is_user=False,
            sources=json.dumps(final_meta.get("source_documents", [])),
        ))
        db.commit()
        db.close()
    except Exception as exc:
        logger.error("[SSE] history save failed: %s", exc)

    # Emit metadata
    meta = json.dumps({
        "source_documents":   final_meta.get("source_documents", []),
        "citations":          final_meta.get("citations", []),
        "confidence":         final_meta.get("confidence", "UNKNOWN"),
        "confidence_score":   final_meta.get("confidence_score", 0.0),
        "retrieval_mode":     final_meta.get("retrieval_mode", ""),
        "processing_time_ms": final_meta.get("processing_time_ms", 0.0),
        "session_id":         session_id,
    })
    yield f"data: [META]{meta}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/api/chat/stream")
def chat_stream(q: str = Query(..., description="Query text")):
    """Public SSE streaming endpoint — no auth, no history saved.

    Always retrieves as role=Public. The role is never taken from the request,
    so an unauthenticated caller cannot ask for Student/Faculty/Admin content.
    """
    return StreamingResponse(
        _sse_public_generator(question=q, role="Public"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/chat/auth/stream")
def chat_stream_auth(
    q: str = Query(..., description="Query text"),
    session_id: str = Query(default=""),
    current_user: User = Depends(get_current_user),
):
    """Authenticated SSE streaming — saves history, emits metadata at end."""
    sid = session_id or str(uuid.uuid4())
    return StreamingResponse(
        _sse_auth_generator(
            question=q, role=current_user.role,
            username=current_user.username, session_id=sid,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── LangGraph agent chat ───────────────────────────────────────────────────────
@app.post("/api/agent/chat")
def agent_chat(
    payload: AgentChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Agentic chat via the multi-agent LangGraph workflow:
    Query Analyzer → Planner → Retrieval Agent → Context Validator →
    Response Generator → Citation Formatter → Confidence Evaluator →
    Reflection Agent (conditional retry) → Final Response.

    Response shape is a superset of the legacy agent, so existing clients keep
    working unchanged.
    """
    from agents.multi_agent_graph import run_multi_agent
    sid = payload.session_id or str(uuid.uuid4())

    db.add(ChatMessage(
        username=current_user.username, session_id=sid,
        content=payload.query, is_user=True, sources=None,
    ))

    result = run_multi_agent(
        question=payload.query,
        role=current_user.role,
        memory=payload.memory or [],
    )

    db.add(ChatMessage(
        username=current_user.username, session_id=sid,
        content=result["answer"], is_user=False,
        sources=json.dumps(result["source_documents"]),
    ))
    db.commit()

    return {**result, "session_id": sid}


# ── Chat history ───────────────────────────────────────────────────────────────
@app.get("/api/history")
def get_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(ChatMessage)
        .filter(ChatMessage.username == current_user.username)
        .order_by(ChatMessage.timestamp.desc())
        .all()
    )

    sessions: dict[str, dict] = {}
    for row in rows:
        if row.session_id not in sessions:
            sessions[row.session_id] = {
                "session_id":    row.session_id,
                "preview":       "",
                "message_count": 0,
                "timestamp":     row.timestamp.isoformat(),
            }
        sessions[row.session_id]["message_count"] += 1
        if row.is_user and not sessions[row.session_id]["preview"]:
            sessions[row.session_id]["preview"] = row.content[:80]

    return sorted(sessions.values(), key=lambda x: x["timestamp"], reverse=True)


@app.get("/api/history/{session_id}")
def get_conversation(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.username   == current_user.username,
            ChatMessage.session_id == session_id,
        )
        .order_by(ChatMessage.timestamp.asc())
        .all()
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return [
        {
            "content":   r.content,
            "is_user":   r.is_user,
            "sources":   json.loads(r.sources) if r.sources else [],
            "timestamp": r.timestamp.isoformat(),
        }
        for r in rows
    ]


@app.delete("/api/history/{session_id}")
def delete_conversation(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    db.query(ChatMessage).filter(
        ChatMessage.username   == current_user.username,
        ChatMessage.session_id == session_id,
    ).delete()
    db.commit()
    return {"status": "deleted"}


# ── Document upload (Admin) ────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """Admin-only: upload a .docx document → triggers full ingestion pipeline."""
    if current_user.role != "Admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only Admin users can upload documents",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File is empty.")

    from backend.document_manager import ingest_uploaded_file
    result = ingest_uploaded_file(filename=file.filename, content_bytes=content)

    if result.status == "duplicate":
        raise HTTPException(
            status_code=409,
            detail=f"Duplicate: {result.error or 'identical document already indexed'}",
        )
    if result.status == "failed":
        raise HTTPException(status_code=400, detail=result.error or "Ingestion failed.")

    return {
        "status":          result.status,
        "filename":        result.filename,
        "doc_id":          result.doc_id,
        "department":      result.department,
        "version":         result.version,
        "chunks_created":  result.chunks_created,
        "vectors_added":   result.vectors_added,
        "processing_ms":   result.processing_ms,
        "superseded":      result.status == "superseded",
    }


# ── Users list (Admin) ─────────────────────────────────────────────────────────
@app.get("/api/users")
def list_users(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role != "Admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    users = db.query(User).all()
    return [{"username": u.username, "role": u.role} for u in users]


# ── Citation document viewer ────────────────────────────────────────────────────
def _authorize_document(doc_id: str, current_user: User):
    """
    Shared RBAC + resolution for the document endpoints.

    Enforces, in order:
      1. Public role can never open documents          → 403
      2. Document must exist and have a servable file   → 404
      3. Role must satisfy the document's access level  → 403

    Returns the resolved DocumentRef. Filesystem paths are never exposed.
    """
    from backend import document_service as ds

    if not ds.role_can_open(current_user.role):
        raise HTTPException(
            status_code=403,
            detail="Document viewing is not available for public users. Please sign in.",
        )

    docref = ds.load_document(doc_id)
    if docref is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    if not ds.can_access(current_user.role, docref.access_level):
        raise HTTPException(
            status_code=403,
            detail=f"Your role ({current_user.role}) cannot access this document.",
        )
    return docref


@app.get("/api/documents/{doc_id}/meta")
def document_meta(doc_id: str, current_user: User = Depends(get_current_user)):
    """Lightweight metadata for a document the user is allowed to open."""
    docref = _authorize_document(doc_id, current_user)
    return {
        "doc_id":       docref.doc_id,
        "title":        docref.title,
        "department":   docref.department,
        "version":      docref.version,
        "category":     docref.category,
        "access_level": docref.access_level,
        "kind":         docref.kind,
    }


@app.get("/api/documents/{doc_id}/view", response_class=HTMLResponse)
def document_view(
    doc_id: str,
    chunk_index: Optional[int] = Query(default=None),
    current_user: User = Depends(get_current_user),
):
    """
    Returns a self-contained HTML viewer for the document with the source chunk
    highlighted. RBAC-enforced; public users are rejected.
    """
    from backend import document_service as ds
    docref = _authorize_document(doc_id, current_user)
    logger.info(
        "[DOCVIEW] view doc_id=%s chunk=%s by=%s(%s)",
        doc_id[:12], chunk_index, current_user.username, current_user.role,
    )
    return HTMLResponse(content=ds.render_viewer_html(docref, chunk_index))


@app.get("/api/documents/{doc_id}/file")
def document_file(doc_id: str, current_user: User = Depends(get_current_user)):
    """
    Streams the original document bytes (RBAC-enforced). Streams from disk in
    blocks — the filesystem path is never exposed to the client.
    """
    from backend import document_service as ds
    docref = _authorize_document(doc_id, current_user)
    filename = docref.path.name
    logger.info(
        "[DOCVIEW] file doc_id=%s by=%s(%s)",
        doc_id[:12], current_user.username, current_user.role,
    )
    return StreamingResponse(
        ds.iter_file_bytes(docref.path),
        media_type=ds.media_type_for(docref.path),
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


# ── Admin Dashboard APIs ───────────────────────────────────────────────────────
def _require_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "Admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


@app.get("/api/admin/stats")
def admin_stats(_: User = Depends(_require_admin)):
    """Document counts, chunk counts, vector counts from ledger + ChromaDB."""
    from backend.document_manager import get_document_stats, get_chroma_vector_count
    stats = get_document_stats()
    stats["chroma_vector_count"] = get_chroma_vector_count()
    return stats


@app.get("/api/admin/documents")
def admin_documents(_: User = Depends(_require_admin)):
    """Full document registry from SQLite ledger."""
    from backend.document_manager import get_document_list
    docs = get_document_list()
    # Return a clean subset of fields
    return [
        {
            "doc_id":       d.get("doc_id", ""),
            "source_file":  d.get("source_file", ""),
            "title":        d.get("title", ""),
            "department":   d.get("department", ""),
            "version":      d.get("version", ""),
            "category":     d.get("category", ""),
            "status":       d.get("status", ""),
            "total_chunks": d.get("total_chunks", 0),
            "upload_date":  d.get("upload_date", ""),
            "ingested_at":  d.get("ingested_at", ""),
        }
        for d in docs
    ]


@app.get("/api/admin/logs")
def admin_logs(
    limit: int = Query(50, ge=1, le=500),
    _: User = Depends(_require_admin),
):
    """Recent ingestion audit log events."""
    from backend.document_manager import get_ingestion_logs
    return get_ingestion_logs(limit=limit)


@app.get("/api/admin/chroma")
def admin_chroma(_: User = Depends(_require_admin)):
    """ChromaDB collection info."""
    from backend.document_manager import get_chroma_vector_count
    from backend.document_manager import get_document_stats
    stats = get_document_stats()
    return {
        "collection":    "vit_institutional_kb",
        "vector_count":  get_chroma_vector_count(),
        "total_chunks":  stats["total_chunks"],
        "embedded":      stats["embedded_vectors"],
        "coverage_pct":  round(
            100 * stats["embedded_vectors"] / max(stats["total_chunks"], 1), 1
        ),
    }
