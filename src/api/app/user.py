from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request

from ...agents import V1Agent
from .auth import create_token, get_current_user, now_iso, verify_password
from .db import connect
from .schemas import (
    AddSourceRequest,
    AskRequest,
    DeleteDocumentRequest,
    LoginRequest,
    UserKnowledgeBaseCreateRequest,
)


router = APIRouter(tags=["user"])


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
    agent = V1Agent(index=index, llm=request.app.state.llm, top_k=payload.top_k)
    return {"answer": agent.answer(payload.question, source_paths=payload.source_paths)}
