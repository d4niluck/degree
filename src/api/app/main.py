from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI

from ...llm import OpenAILLM
from .admin import router as admin_router
from .auth import hash_password, now_iso
from .db import connect
from .db import init_db
from .index_manager import IndexManager
from .user import router as user_router

load_dotenv()


def ensure_default_user(
    app_db_path: Path,
    *,
    user_id: str,
    login: str,
    password: str,
    role: str,
) -> None:
    conn = connect(app_db_path)
    existing = conn.execute(
        "SELECT user_id FROM users WHERE login = ?",
        (login,),
    ).fetchone()
    with conn:
        if existing:
            conn.execute(
                """
                UPDATE users
                SET password_hash = ?, role = ?, is_active = 1
                WHERE login = ?
                """,
                (hash_password(password), role, login),
            )
        else:
            conn.execute(
                """
                INSERT INTO users (user_id, login, password_hash, role, is_active, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, login, hash_password(password), role, 1, now_iso()),
            )
    conn.close()


def ensure_default_accounts(app_db_path: Path) -> None:
    ensure_default_user(
        app_db_path,
        user_id="admin",
        login=os.getenv("APP_ADMIN_LOGIN", "admin"),
        password=os.getenv("APP_ADMIN_PASSWORD", "admin"),
        role="admin",
    )
    ensure_default_user(
        app_db_path,
        user_id="user",
        login=os.getenv("APP_USER_LOGIN", "user"),
        password=os.getenv("APP_USER_PASSWORD", "user"),
        role="user",
    )


def create_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("app_api")

    root = Path(os.getenv("APP_STORAGE_ROOT", "data/app_storage")).resolve()
    app_db_path = root / "app.db"
    index_storage_root = root / "indexes"
    init_db(app_db_path)
    ensure_default_accounts(app_db_path)

    client = OpenAI(
        api_key=os.getenv("OPEN_AI_KEY"),
        base_url=os.getenv("BASE_OPEN_AI_URL"),
    )

    app = FastAPI(title="Main App API")
    app.state.logger = logger
    app.state.app_db_path = str(app_db_path)
    app.state.index_storage_root = str(index_storage_root)
    app.state.index_manager = IndexManager(
        storage_root=index_storage_root,
        logger=logger,
        embedder_url=os.getenv("EMBEDDER_API_URL", "http://127.0.0.1:8001"),
        reader_url=os.getenv("READER_API_URL", "http://127.0.0.1:8002"),
    )
    app.state.llm = OpenAILLM(
        client=client,
        model_name=os.getenv("APP_LLM_MODEL", "gpt-4o-mini"),
    )

    @app.get("/health")
    def health():
        return {"ok": True}

    app.include_router(user_router)
    app.include_router(admin_router)
    return app


app = create_app()
