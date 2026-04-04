from __future__ import annotations

from typing import TYPE_CHECKING, List

from pydantic import BaseModel, Field

from .base import BaseAgent

if TYPE_CHECKING:
    from ..indexing.index import Index
    from ..llm.llm import BaseLLM


class V1Answer(BaseModel):
    reasoning: List[str] = Field(..., description="3-5 предложений по обдумыванию контекста")
    answer: str = Field(..., description="Финальный ответ")


class V1Agent(BaseAgent):
    def __init__(
        self,
        index: "Index",
        llm: "BaseLLM",
        top_k: int = 10,
        system_prompt: str = (
            "Ты отвечаешь на вопросы пользователя по строительной документации. "
            "Отвечай только на основе доступного контекста. "
            "Запрещено утверждать или предполагать специфичные факты, которых нет в найденной инйормации. "
        ),
    ) -> None:
        self.index = index
        self.llm = llm
        self.top_k = top_k
        self.system_prompt = system_prompt
        
        self.last = {
            "question": None,
            "context": None,
            "answer": None,
        }

    def answer(self, question: str) -> str:
        extracted = self.index.search(question, top_k=self.top_k)
        context = "\n\n".join(
            [f"{i + 1}. {result.chunk['text']}" for i, result in enumerate(extracted)]
        )
        prompt = (
            f"{self.system_prompt}\n\n"
            f"Запрос: {question}\n\n"
            f"Контекст: {context}"
        )
        response = self.llm.parse(prompt, V1Answer)
        answer = response.answer
        self._upd_last(question, context, answer)
        return answer
    
    def _upd_last(self, q: str, c: str, a: str):
        self.last = {
            "question": q,
            "context": c,
            "answer": a,
        }

        
