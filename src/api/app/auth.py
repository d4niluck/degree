from __future__ import annotations

import hashlib
import secrets
from datetime import datetime
from typing import Dict
from uuid import uuid4

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .db import connect


TOKENS: Dict[str, str] = {}
security = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    salt, digest_hex = password_hash.split("$", 1)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return secrets.compare_digest(digest.hex(), digest_hex)


def create_token(user_id: str) -> str:
    token = uuid4().hex
    TOKENS[token] = user_id
    return token


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
):
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = credentials.credentials.strip()
    user_id = TOKENS.get(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    conn = connect(request.app.state.app_db_path)
    row = conn.execute(
        "SELECT user_id, login, role, is_active FROM users WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    conn.close()
    if not row or not row["is_active"]:
        raise HTTPException(status_code=401, detail="Inactive user")
    return dict(row)


def require_admin(user=...):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
