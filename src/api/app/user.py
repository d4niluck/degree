from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

from .agent import create_backend_v5_agent
from .auth import create_token, get_current_user, now_iso, verify_password
from .db import connect
from .dialog_router import route_dialog_turn
from .schemas import (
    AddSourceRequest,
    AskRequest,
    ConversationAskResponse,
    ConversationCreateRequest,
    ConversationDetailResponse,
    ConversationMessageResponse,
    ConversationResponse,
    ConversationSummaryResponse,
    ConversationTokenUsageResponse,
    DeleteDocumentRequest,
    LoginRequest,
    UserKnowledgeBaseCreateRequest,
)


router = APIRouter(tags=["user"])
_TOKEN_COUNTER = None
_CONVERSATION_MESSAGE_LIMIT = 50
_CONVERSATION_TAIL_KEEP = 12
_AGENT_TOP_K_BM25 = 3


def _display_document_name(source_path: str) -> str:
    name = Path(source_path).name
    prefix, separator, suffix = name.partition("_")
    if separator and len(prefix) == 32:
        return suffix
    return name


def _require_kb_access(request: Request, user_id: str, kb_id: str) -> None:
    conn = connect(request.app.state.app_db_path)
    row = conn.execute(
        "SELECT 1 FROM user_kb_access WHERE user_id = ? AND kb_id = ?",
        (user_id, kb_id),
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=403, detail="No access to knowledge base")


def _require_kb_owner(request: Request, user_id: str, kb_id: str) -> None:
    conn = connect(request.app.state.app_db_path)
    row = conn.execute(
        """
        SELECT 1 FROM user_kb_access
        WHERE user_id = ? AND kb_id = ? AND access_role = 'owner'
        """,
        (user_id, kb_id),
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=403, detail="Need owner access to knowledge base")


def _estimate_tokens(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    global _TOKEN_COUNTER
    if _TOKEN_COUNTER is None:
        try:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")
            _TOKEN_COUNTER = lambda value: len(encoding.encode(value))
        except Exception:
            try:
                tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
                _TOKEN_COUNTER = lambda value: len(tokenizer.encode(value, add_special_tokens=False))
            except Exception:
                _TOKEN_COUNTER = lambda value: max(1, (len(value) + 3) // 4)
    return int(_TOKEN_COUNTER(text))


def _default_conversation_title(title: str | None, fallback: str = "Новый диалог") -> str:
    if title and title.strip():
        return title.strip()[:120]
    return fallback


def _require_conversation_access(
    request: Request,
    user_id: str,
    kb_id: str,
    conversation_id: str,
):
    conn = connect(request.app.state.app_db_path)
    row = conn.execute(
        """
        SELECT c.conversation_id, c.kb_id, c.title, c.created_at, c.updated_at
        FROM conversations c
        JOIN user_kb_access a ON a.kb_id = c.kb_id
        WHERE c.conversation_id = ? AND c.kb_id = ? AND a.user_id = ?
        """,
        (conversation_id, kb_id, user_id),
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return row


def _conversation_messages(conn, conversation_id: str):
    return conn.execute(
        """
        SELECT message_id, role, content, created_at
        FROM conversation_messages
        WHERE conversation_id = ?
        ORDER BY rowid
        """,
        (conversation_id,),
    ).fetchall()


def _serialize_messages(rows) -> list[ConversationMessageResponse]:
    return [
        ConversationMessageResponse(
            message_id=row["message_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
        )
        for row in rows
    ]


def _upsert_conversation_title(conn, conversation_id: str, title: str) -> None:
    conn.execute(
        """
        UPDATE conversations
        SET title = ?, updated_at = ?
        WHERE conversation_id = ?
        """,
        (title[:120], now_iso(), conversation_id),
    )


def _touch_conversation(conn, conversation_id: str) -> None:
    conn.execute(
        "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
        (now_iso(), conversation_id),
    )


def _rows_to_message_dicts(rows) -> list[dict]:
    return [
        {
            "message_id": row["message_id"],
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def _summarize_messages(llm, messages: list[dict]) -> str:
    class SummarySchema(BaseModel):
        summary: str

    transcript = "\n\n".join(
        f"{str(message.get('role', 'assistant')).upper()}:\n{str(message.get('content', '')).strip()}"
        for message in messages
        if str(message.get("content", "")).strip()
    )
    if not transcript.strip():
        return ""
    prompt = f"""
Суммаризируй диалог пользователя и ассистента.

Требования:
- Сохрани только полезный контекст для продолжения разговора.
- Кратко перечисли уже установленные факты.
- Кратко укажи незавершенные вопросы, если они есть.
- Не добавляй ничего от себя.
- Ответ должен быть коротким и пригодным для дальнейшего использования в диалоге.

Диалог:
{transcript}
""".strip()
    result = llm.parse(prompt, SummarySchema, temperature=0)
    return result.summary.strip()


def _compress_conversation_if_needed(request: Request, conversation_id: str) -> list[dict]:
    conn = connect(request.app.state.app_db_path)
    rows = _conversation_messages(conn, conversation_id)
    messages = _rows_to_message_dicts(rows)
    if len(messages) <= _CONVERSATION_MESSAGE_LIMIT:
        conn.close()
        return messages

    keep_tail = min(_CONVERSATION_TAIL_KEEP, len(messages))
    old_messages = messages[:-keep_tail]
    recent_messages = messages[-keep_tail:]
    summary = _summarize_messages(request.app.state.llm, old_messages)
    with conn:
        conn.execute("DELETE FROM conversation_messages WHERE conversation_id = ?", (conversation_id,))
        if summary:
            conn.execute(
                """
                INSERT INTO conversation_messages (message_id, conversation_id, role, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (uuid4().hex, conversation_id, "assistant", summary, now_iso()),
            )
        for message in recent_messages:
            conn.execute(
                """
                INSERT INTO conversation_messages (message_id, conversation_id, role, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    uuid4().hex,
                    conversation_id,
                    message["role"],
                    message["content"],
                    message["created_at"],
                ),
            )
        _touch_conversation(conn, conversation_id)
    rows = _conversation_messages(conn, conversation_id)
    conn.close()
    return _rows_to_message_dicts(rows)


def _store_conversation_turn(
    request: Request,
    conversation: dict,
    conversation_id: str,
    user_message: str,
    assistant_message: str,
) -> str:
    conn = connect(request.app.state.app_db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO conversation_messages (message_id, conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (uuid4().hex, conversation_id, "user", user_message, now_iso()),
        )
        conn.execute(
            """
            INSERT INTO conversation_messages (message_id, conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (uuid4().hex, conversation_id, "assistant", assistant_message, now_iso()),
        )
        if conversation["title"] == "Новый диалог":
            _upsert_conversation_title(conn, conversation_id, user_message)
        _touch_conversation(conn, conversation_id)
    conn.close()
    if conversation["title"] == "Новый диалог":
        return user_message[:120]
    return str(conversation["title"])


@router.post("/login")
def login(payload: LoginRequest, request: Request):
    conn = connect(request.app.state.app_db_path)
    row = conn.execute(
        "SELECT user_id, password_hash, is_active FROM users WHERE login = ?",
        (payload.login,),
    ).fetchone()
    conn.close()
    if not row or not row["is_active"] or not verify_password(payload.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": create_token(row["user_id"]), "token_type": "bearer"}


@router.get("/knowledge-bases")
def list_knowledge_bases(request: Request, user=Depends(get_current_user)):
    conn = connect(request.app.state.app_db_path)
    rows = conn.execute(
        """
        SELECT kb.kb_id, kb.owner_user_id, kb.name, kb.storage_path
        FROM knowledge_bases kb
        JOIN user_kb_access a ON a.kb_id = kb.kb_id
        WHERE a.user_id = ?
        ORDER BY kb.created_at
        """,
        (user["user_id"],),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


@router.post("/knowledge-bases")
def create_knowledge_base(
    payload: UserKnowledgeBaseCreateRequest,
    request: Request,
    user=Depends(get_current_user),
):
    conn = connect(request.app.state.app_db_path)
    kb_id = uuid4().hex
    storage_path = str(Path(request.app.state.index_storage_root) / kb_id)
    with conn:
        duplicate = conn.execute(
            """
            SELECT 1 FROM knowledge_bases
            WHERE owner_user_id = ? AND name = ?
            """,
            (user["user_id"], payload.name),
        ).fetchone()
        if duplicate:
            raise HTTPException(
                status_code=409,
                detail="Knowledge base with this name already exists for this user",
            )
        conn.execute(
            """
            INSERT INTO knowledge_bases (kb_id, owner_user_id, name, storage_path, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (kb_id, user["user_id"], payload.name, storage_path, now_iso()),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO user_kb_access (user_id, kb_id, access_role, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (user["user_id"], kb_id, "owner", now_iso()),
        )
    conn.close()
    return {"kb_id": kb_id, "owner_user_id": user["user_id"], "name": payload.name, "storage_path": storage_path}


@router.delete("/knowledge-bases/{kb_id}")
def delete_knowledge_base(kb_id: str, request: Request, user=Depends(get_current_user)):
    _require_kb_owner(request, user["user_id"], kb_id)
    conn = connect(request.app.state.app_db_path)
    row = conn.execute(
        "SELECT storage_path FROM knowledge_bases WHERE kb_id = ?",
        (kb_id,),
    ).fetchone()
    with conn:
        conversation_ids = conn.execute(
            "SELECT conversation_id FROM conversations WHERE kb_id = ?",
            (kb_id,),
        ).fetchall()
        for conversation_id_row in conversation_ids:
            conn.execute(
                "DELETE FROM conversation_messages WHERE conversation_id = ?",
                (conversation_id_row["conversation_id"],),
            )
        conn.execute("DELETE FROM conversations WHERE kb_id = ?", (kb_id,))
        conn.execute("DELETE FROM user_kb_access WHERE kb_id = ?", (kb_id,))
        conn.execute("DELETE FROM knowledge_bases WHERE kb_id = ?", (kb_id,))
    conn.close()
    request.app.state.index_manager.close_index(kb_id)
    if row:
        shutil.rmtree(row["storage_path"], ignore_errors=True)
    return {"ok": True}


@router.get("/knowledge-bases/{kb_id}/documents")
def list_documents(kb_id: str, request: Request, user=Depends(get_current_user)):
    _require_kb_access(request, user["user_id"], kb_id)
    index = request.app.state.index_manager.get_index(kb_id)
    documents = []
    for doc_id in index.list_documents():
        document = index.get_document(doc_id=doc_id)
        documents.append(
            {
                "doc_id": doc_id,
                "source_path": document.source_path if document else None,
                "name": _display_document_name(document.source_path) if document else doc_id,
                "pages": len(document.pages) if document else None,
            }
        )
    return documents


@router.post("/knowledge-bases/{kb_id}/documents")
def add_document(
    kb_id: str,
    payload: AddSourceRequest,
    request: Request,
    user=Depends(get_current_user),
):
    _require_kb_access(request, user["user_id"], kb_id)
    index = request.app.state.index_manager.get_index(kb_id)
    doc_ids = index.add_sources([payload.file_path], save_vectorestore=True)
    return {"doc_ids": doc_ids}


@router.delete("/knowledge-bases/{kb_id}/documents")
def delete_document(
    kb_id: str,
    payload: DeleteDocumentRequest,
    request: Request,
    user=Depends(get_current_user),
):
    _require_kb_access(request, user["user_id"], kb_id)
    index = request.app.state.index_manager.get_index(kb_id)
    if payload.doc_id:
        index.delete_documents(doc_ids=[payload.doc_id], save_vectorestore=True)
    elif payload.source_path:
        index.delete_documents(source_paths=[payload.source_path], save_vectorestore=True)
    else:
        raise HTTPException(status_code=400, detail="Need doc_id or source_path")
    return {"ok": True}


@router.post("/knowledge-bases/{kb_id}/search")
def search(
    kb_id: str,
    payload: AskRequest,
    request: Request,
    user=Depends(get_current_user),
):
    _require_kb_access(request, user["user_id"], kb_id)
    index = request.app.state.index_manager.get_index(kb_id)
    results = index.search(
        payload.question,
        top_k=payload.top_k,
        source_paths=payload.source_paths,
    )
    return [
        {
            "doc_id": item.doc_id,
            "chunk_id": item.chunk_id,
            "score": item.score,
            "chunk": item.chunk,
        }
        for item in results
    ]


@router.post("/knowledge-bases/{kb_id}/ask")
def ask(
    kb_id: str,
    payload: AskRequest,
    request: Request,
    user=Depends(get_current_user),
):
    _require_kb_access(request, user["user_id"], kb_id)
    index = request.app.state.index_manager.get_index(kb_id)
    agent = create_backend_v5_agent(
        index=index,
        llm=request.app.state.llm,
        logger=request.app.state.logger,
        max_top_k_dense=min(payload.top_k, 10),
        max_top_k_bm25=_AGENT_TOP_K_BM25,
        source_paths=payload.source_paths,
    )
    return {"answer": agent.answer(payload.question)}


@router.get("/knowledge-bases/{kb_id}/conversations")
def list_conversations(kb_id: str, request: Request, user=Depends(get_current_user)):
    _require_kb_access(request, user["user_id"], kb_id)
    conn = connect(request.app.state.app_db_path)
    rows = conn.execute(
        """
        SELECT c.conversation_id, c.kb_id, c.title, c.created_at, c.updated_at,
               COUNT(m.message_id) AS message_count
        FROM conversations c
        LEFT JOIN conversation_messages m ON m.conversation_id = c.conversation_id
        WHERE c.kb_id = ?
        GROUP BY c.conversation_id, c.kb_id, c.title, c.created_at, c.updated_at
        ORDER BY c.updated_at DESC, c.created_at DESC
        """,
        (kb_id,),
    ).fetchall()
    conn.close()
    return [
        ConversationResponse(
            conversation_id=row["conversation_id"],
            kb_id=row["kb_id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            message_count=int(row["message_count"] or 0),
        )
        for row in rows
    ]


@router.post("/knowledge-bases/{kb_id}/conversations")
def create_conversation(
    kb_id: str,
    payload: ConversationCreateRequest,
    request: Request,
    user=Depends(get_current_user),
):
    _require_kb_access(request, user["user_id"], kb_id)
    conversation_id = uuid4().hex
    created_at = now_iso()
    title = _default_conversation_title(payload.title)
    conn = connect(request.app.state.app_db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO conversations (conversation_id, kb_id, owner_user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (conversation_id, kb_id, user["user_id"], title, created_at, created_at),
        )
    conn.close()
    return ConversationResponse(
        conversation_id=conversation_id,
        kb_id=kb_id,
        title=title,
        created_at=created_at,
        updated_at=created_at,
        message_count=0,
    )


@router.get("/knowledge-bases/{kb_id}/conversations/{conversation_id}")
def get_conversation(
    kb_id: str,
    conversation_id: str,
    request: Request,
    user=Depends(get_current_user),
):
    conversation = _require_conversation_access(request, user["user_id"], kb_id, conversation_id)
    conn = connect(request.app.state.app_db_path)
    messages = _conversation_messages(conn, conversation_id)
    conn.close()
    return ConversationDetailResponse(
        conversation_id=conversation["conversation_id"],
        kb_id=conversation["kb_id"],
        title=conversation["title"],
        created_at=conversation["created_at"],
        updated_at=conversation["updated_at"],
        messages=_serialize_messages(messages),
    )


@router.delete("/knowledge-bases/{kb_id}/conversations/{conversation_id}")
def delete_conversation(
    kb_id: str,
    conversation_id: str,
    request: Request,
    user=Depends(get_current_user),
):
    _require_conversation_access(request, user["user_id"], kb_id, conversation_id)
    conn = connect(request.app.state.app_db_path)
    with conn:
        conn.execute("DELETE FROM conversation_messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
    conn.close()
    return {"ok": True}


@router.get(
    "/knowledge-bases/{kb_id}/conversations/{conversation_id}/token-usage",
    response_model=ConversationTokenUsageResponse,
)
def conversation_token_usage(
    kb_id: str,
    conversation_id: str,
    request: Request,
    user=Depends(get_current_user),
):
    _require_conversation_access(request, user["user_id"], kb_id, conversation_id)
    conn = connect(request.app.state.app_db_path)
    messages = _conversation_messages(conn, conversation_id)
    conn.close()
    char_count = sum(len(row["content"] or "") for row in messages)
    return ConversationTokenUsageResponse(
        conversation_id=conversation_id,
        message_count=len(messages),
        char_count=char_count,
        estimated_tokens=_estimate_tokens(" ".join((row["content"] or "") for row in messages)),
    )


@router.post("/knowledge-bases/{kb_id}/conversations/{conversation_id}/clear")
def clear_conversation(
    kb_id: str,
    conversation_id: str,
    request: Request,
    user=Depends(get_current_user),
):
    _require_conversation_access(request, user["user_id"], kb_id, conversation_id)
    conn = connect(request.app.state.app_db_path)
    with conn:
        conn.execute("DELETE FROM conversation_messages WHERE conversation_id = ?", (conversation_id,))
        _upsert_conversation_title(conn, conversation_id, "Новый диалог")
    conn.close()
    return {"ok": True}


@router.post(
    "/knowledge-bases/{kb_id}/conversations/{conversation_id}/summarize",
    response_model=ConversationSummaryResponse,
)
def summarize_conversation(
    kb_id: str,
    conversation_id: str,
    request: Request,
    user=Depends(get_current_user),
):
    _require_conversation_access(request, user["user_id"], kb_id, conversation_id)
    conn = connect(request.app.state.app_db_path)
    rows = _conversation_messages(conn, conversation_id)
    if not rows:
        conn.close()
        return ConversationSummaryResponse(conversation_id=conversation_id, summary="")
    summary = _summarize_messages(request.app.state.llm, _rows_to_message_dicts(rows))
    with conn:
        conn.execute("DELETE FROM conversation_messages WHERE conversation_id = ?", (conversation_id,))
        if summary:
            conn.execute(
                """
                INSERT INTO conversation_messages (message_id, conversation_id, role, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (uuid4().hex, conversation_id, "assistant", summary, now_iso()),
            )
        _touch_conversation(conn, conversation_id)
    conn.close()
    return ConversationSummaryResponse(conversation_id=conversation_id, summary=summary)


@router.post(
    "/knowledge-bases/{kb_id}/conversations/{conversation_id}/messages",
    response_model=ConversationAskResponse,
)
def ask_in_conversation(
    kb_id: str,
    conversation_id: str,
    payload: AskRequest,
    request: Request,
    user=Depends(get_current_user),
):
    conversation = _require_conversation_access(request, user["user_id"], kb_id, conversation_id)
    history_messages = _compress_conversation_if_needed(request, conversation_id)
    router_decision = route_dialog_turn(
        request.app.state.llm,
        messages=history_messages,
        user_message=payload.question,
    )
    if router_decision.action == "respond_from_context":
        answer = (router_decision.assistant_response or "").strip()
        title = _store_conversation_turn(
            request,
            conversation,
            conversation_id,
            payload.question,
            answer,
        )
        return ConversationAskResponse(
            answer=answer,
            conversation_id=conversation_id,
            title=title,
        )

    index = request.app.state.index_manager.get_index(kb_id)
    agent = create_backend_v5_agent(
        index=index,
        llm=request.app.state.llm,
        logger=request.app.state.logger,
        max_top_k_dense=min(payload.top_k, 10),
        max_top_k_bm25=_AGENT_TOP_K_BM25,
        source_paths=payload.source_paths,
    )
    search_question = (router_decision.search_question or payload.question).strip()
    answer = agent.answer(search_question)
    title = _store_conversation_turn(
        request,
        conversation,
        conversation_id,
        payload.question,
        answer,
    )
    return ConversationAskResponse(
        answer=answer,
        conversation_id=conversation_id,
        title=title,
    )


@router.post("/knowledge-bases/{kb_id}/conversations/{conversation_id}/messages/stream")
def ask_in_conversation_stream(
    kb_id: str,
    conversation_id: str,
    payload: AskRequest,
    request: Request,
    user=Depends(get_current_user),
):
    conversation = _require_conversation_access(request, user["user_id"], kb_id, conversation_id)
    history_messages = _compress_conversation_if_needed(request, conversation_id)
    router_decision = route_dialog_turn(
        request.app.state.llm,
        messages=history_messages,
        user_message=payload.question,
    )

    def event_iter() -> Iterator[str]:
        def push_event(event_type: str, payload_data: dict) -> str:
            return json.dumps({"type": event_type, **payload_data}, ensure_ascii=False) + "\n"

        yield push_event("started", {"conversation_id": conversation_id})
        try:
            if router_decision.action == "respond_from_context":
                answer = (router_decision.assistant_response or "").strip()
                title = _store_conversation_turn(
                    request,
                    conversation,
                    conversation_id,
                    payload.question,
                    answer,
                )
                yield push_event(
                    "final_answer",
                    {
                        "answer": answer,
                        "conversation_id": conversation_id,
                        "title": title,
                    },
                )
                return

            index = request.app.state.index_manager.get_index(kb_id)
            agent = create_backend_v5_agent(
                index=index,
                llm=request.app.state.llm,
                logger=request.app.state.logger,
                max_top_k_dense=min(payload.top_k, 10),
                max_top_k_bm25=_AGENT_TOP_K_BM25,
                source_paths=payload.source_paths,
            )
            answer = ""
            search_question = (router_decision.search_question or payload.question).strip()
            for event in agent.iter_events(search_question):
                event_type = str(event.get("type"))
                if event_type == "final_answer":
                    answer = str(event.get("answer", "")).strip()
                    title = _store_conversation_turn(
                        request,
                        conversation,
                        conversation_id,
                        payload.question,
                        answer,
                    )
                    yield push_event(
                        "final_answer",
                        {
                            "answer": answer,
                            "conversation_id": conversation_id,
                            "title": title,
                        },
                    )
                else:
                    yield push_event(
                        event_type,
                        {key: value for key, value in event.items() if key != "type"},
                    )
        except Exception as exc:
            yield push_event("error", {"error": str(exc)})

    return StreamingResponse(event_iter(), media_type="application/x-ndjson")
