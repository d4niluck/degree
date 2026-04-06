from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class LoginRequest(BaseModel):
    login: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreateRequest(BaseModel):
    login: str
    password: str
    role: str = "user"


class UserResponse(BaseModel):
    user_id: str
    login: str
    role: str
    is_active: bool


class KnowledgeBaseCreateRequest(BaseModel):
    owner_user_id: str
    name: str


class KnowledgeBaseAccessRequest(BaseModel):
    user_id: str
    access_role: str = "reader"


class KnowledgeBaseResponse(BaseModel):
    kb_id: str
    owner_user_id: str
    name: str
    storage_path: str


class AddSourceRequest(BaseModel):
    file_path: str


class DeleteDocumentRequest(BaseModel):
    doc_id: Optional[str] = None
    source_path: Optional[str] = None


class AskRequest(BaseModel):
    question: str
    top_k: int = 10
    source_paths: Optional[List[str]] = None
