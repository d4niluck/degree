from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path) -> None:
    conn = connect(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                login TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('admin', 'user')),
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_bases (
                kb_id TEXT PRIMARY KEY,
                owner_user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (owner_user_id) REFERENCES users(user_id)
            )
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_bases_owner_name
            ON knowledge_bases(owner_user_id, name)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_kb_access (
                user_id TEXT NOT NULL,
                kb_id TEXT NOT NULL,
                access_role TEXT NOT NULL CHECK (access_role IN ('owner', 'reader')),
                created_at TEXT NOT NULL,
                PRIMARY KEY (user_id, kb_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                FOREIGN KEY (kb_id) REFERENCES knowledge_bases(kb_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                kb_id TEXT NOT NULL,
                owner_user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (kb_id) REFERENCES knowledge_bases(kb_id),
                FOREIGN KEY (owner_user_id) REFERENCES users(user_id)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversations_kb_id
            ON conversations(kb_id)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversation_messages_conversation_id
            ON conversation_messages(conversation_id, created_at)
            """
        )
    conn.close()
