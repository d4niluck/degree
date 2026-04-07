from pydantic import BaseModel


class UserCreateForm(BaseModel):
    login: str
    password: str
    role: str = "user"


class KnowledgeBaseCreateForm(BaseModel):
    owner_user_id: str
    name: str


class UserKnowledgeBaseCreateForm(BaseModel):
    name: str
