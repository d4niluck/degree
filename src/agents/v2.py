# Hybrid search agent

from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional
from pydantic import BaseModel, Field
from .base import BaseAgent
if TYPE_CHECKING:
    from ..indexing.index import Index
    from ..llm.llm import BaseLLM


class V2Answer(BaseModel):
    reasoning: List[str] = Field(..., description="3-5 предложений по обдумыванию контекста")
    answer: str = Field(..., description="Финальный ответ")


class V2Agent(BaseAgent):
    def __init__(
        self,
        index: "Index",
        llm: "BaseLLM",
        top_k_dense: int = 5,
        top_k_bm25: int = 5,
        system_prompt: str = (
            "Ты отвечаешь на вопросы пользователя по строительной документации. "
            "Отвечай только на основе доступного контекста. "
            "Запрещено утверждать или предполагать специфичные факты, которых нет в найденной информации. "
        ),
    ) -> None:
        self.index = index
        self.llm = llm
        self.top_k_dense = top_k_dense
        self.top_k_bm25 = top_k_bm25
        self.system_prompt = system_prompt
        self.last = {
            "question": None,
            "context": None,
            "answer": None,
        }

    def answer(self, question: str, source_paths: Optional[List[str]] = None) -> str:
        extracted = self.index.search_hybrid(
            query=question,
            top_k_dense=self.top_k_dense,
            top_k_bm25=self.top_k_bm25,
            source_paths=source_paths,
        )
        context = "\n\n".join(
            [f"{i + 1}. {result.chunk['text']}" for i, result in enumerate(extracted)]
        )
        prompt = (
            f"{self.system_prompt}\n\n"
            f"Запрос: {question}\n\n"
            f"Контекст: {context}"
        )
        response = self.llm.parse(prompt, V2Answer)
        answer = response.answer
        self._upd_last(question, context, answer)
        return answer

    def _upd_last(self, q: str, c: str, a: str) -> None:
        self.last = {
            "question": q,
            "context": c,
            "answer": a,
        }
