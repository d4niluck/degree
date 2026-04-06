from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request

from .auth import get_current_user, hash_password, now_iso, require_admin
from .db import connect
from .schemas import (
    KnowledgeBaseAccessRequest,
    KnowledgeBaseCreateRequest,
    UserCreateRequest,
)


router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/users")
def list_users(request: Request, user=Depends(get_current_user)):
    require_admin(user)
    conn = connect(request.app.state.app_db_path)
    rows = conn.execute(
        "SELECT user_id, login, role, is_active FROM users ORDER BY created_at"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


@router.post("/users")
def create_user(payload: UserCreateRequest, request: Request, user=Depends(get_current_user)):
    require_admin(user)
    conn = connect(request.app.state.app_db_path)
    user_id = uuid4().hex
    with conn:
        conn.execute(
            """
            INSERT INTO users (user_id, login, password_hash, role, is_active, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, payload.login, hash_password(payload.password), payload.role, 1, now_iso()),
        )
    conn.close()
    return {"user_id": user_id, "login": payload.login, "role": payload.role, "is_active": True}


@router.delete("/users/{target_user_id}")
def delete_user(target_user_id: str, request: Request, user=Depends(get_current_user)):
    require_admin(user)
    conn = connect(request.app.state.app_db_path)
    with conn:
        conn.execute("DELETE FROM user_kb_access WHERE user_id = ?", (target_user_id,))
        conn.execute("DELETE FROM knowledge_bases WHERE owner_user_id = ?", (target_user_id,))
        conn.execute("DELETE FROM users WHERE user_id = ?", (target_user_id,))
    conn.close()
    return {"ok": True}


@router.post("/knowledge-bases")
def create_knowledge_base(
    payload: KnowledgeBaseCreateRequest,
    request: Request,
    user=Depends(get_current_user),
):
    require_admin(user)
    conn = connect(request.app.state.app_db_path)
    kb_id = uuid4().hex
    storage_path = str(Path(request.app.state.index_storage_root) / kb_id)
    with conn:
        owner = conn.execute(
            "SELECT 1 FROM users WHERE user_id = ?",
            (payload.owner_user_id,),
        ).fetchone()
        if not owner:
            raise HTTPException(status_code=404, detail="Owner not found")
        conn.execute(
            """
            INSERT INTO knowledge_bases (kb_id, owner_user_id, name, storage_path, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (kb_id, payload.owner_user_id, payload.name, storage_path, now_iso()),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO user_kb_access (user_id, kb_id, access_role, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (payload.owner_user_id, kb_id, "owner", now_iso()),
        )
    conn.close()
    return {"kb_id": kb_id, "owner_user_id": payload.owner_user_id, "name": payload.name, "storage_path": storage_path}


@router.post("/knowledge-bases/{kb_id}/access")
def grant_kb_access(
    kb_id: str,
    payload: KnowledgeBaseAccessRequest,
    request: Request,
    user=Depends(get_current_user),
):
    require_admin(user)
    conn = connect(request.app.state.app_db_path)
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO user_kb_access (user_id, kb_id, access_role, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (payload.user_id, kb_id, payload.access_role, now_iso()),
        )
    conn.close()
    return {"ok": True}
