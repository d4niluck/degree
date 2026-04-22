from __future__ import annotations
import logging
from typing import TYPE_CHECKING, List, Optional
from pydantic import BaseModel, Field
from .base import BaseAgent
if TYPE_CHECKING:
    from ..indexing.index import Index
    from ..llm.llm import BaseLLM


class DecompositionPlan(BaseModel):
    need_decompose: bool
    decomposition_queries: List[str] = Field(default_factory=list, max_length=10)


class V3SubAnswer(BaseModel):
    answer: str = Field(..., description="Краткий ответ на декомпозиционный подзапрос")


class V3FinalAnswer(BaseModel):
    reasoning: List[str] = Field(..., description="3-5 предложений по обдумыванию контекста")
    answer: str = Field(..., description="Финальный ответ")


class V3Agent(BaseAgent):
    def __init__(
        self,
        index: "Index",
        llm: "BaseLLM",
        top_k_dense: int = 5,
        top_k_bm25: int = 5,
        max_decomposition_queries: int = 10,
        logger: Optional[logging.Logger] = None,
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
        self.max_decomposition_queries = max_decomposition_queries
        self.logger = logger
        self.system_prompt = system_prompt
        self.last = {
            "question": None,
            "plan": None,
            "sub_answers": None,
            "answer": None,
        }

    def answer(self, question: str, source_paths: Optional[List[str]] = None) -> str:
        plan = self._plan_decomposition(question)
        if not self._should_decompose(plan):
            answer = self._answer_single_query(question, source_paths=source_paths)
            self._update_last(question, plan, [], answer)
            self._log_answer(question, plan, answer)
            return answer

        sub_answers = self._run_decomposition_queries(
            queries=plan.decomposition_queries,
            source_paths=source_paths,
        )
        answer = self._aggregate_decomposition_answers(
            question=question,
            sub_answers=sub_answers,
        )
        self._update_last(question, plan, sub_answers, answer)
        self._log_answer(question, plan, answer)
        return answer

    def _plan_decomposition(self, question: str) -> DecompositionPlan:
        prompt = (
            "Определи, нужно ли декомпозировать запрос пользователя на несколько атомарных подзапросов. "
            "Если вопрос простой и отвечает на один факт или одну локальную задачу, декомпозиция не нужна. "
            "Если вопрос составной, многошаговый, требует нескольких независимых фактов или сравнения, декомпозиция нужна.\n\n"
            f"Запрос пользователя: {question}"
        )
        plan = self.llm.parse(prompt, DecompositionPlan)
        cleaned_queries = [
            query.strip()
            for query in plan.decomposition_queries[: self.max_decomposition_queries]
            if query.strip()
        ]
        return DecompositionPlan(
            need_decompose=plan.need_decompose,
            decomposition_queries=cleaned_queries,
        )

    def _should_decompose(self, plan: DecompositionPlan) -> bool:
        return plan.need_decompose and len(plan.decomposition_queries) >= 2

    def _answer_single_query(self, question: str, source_paths: Optional[List[str]] = None) -> str:
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
        response = self.llm.parse(prompt, V3SubAnswer)
        return response.answer

    def _run_decomposition_queries(
        self,
        queries: List[str],
        source_paths: Optional[List[str]] = None,
    ) -> List[dict]:
        results = []
        for query in queries:
            answer = self._answer_single_query(query, source_paths=source_paths)
            results.append(
                {
                    "query": query,
                    "answer": answer,
                }
            )
        return results

    def _aggregate_decomposition_answers(
        self,
        question: str,
        sub_answers: List[dict],
    ) -> str:
        decomposition_block = "\n\n".join(
            [
                f"{i + 1}. {item['query']}\nAnswer: {item['answer']}"
                for i, item in enumerate(sub_answers)
            ]
        )
        prompt = (
            f"{self.system_prompt}\n\n"
            f"Запрос пользователя:\n{question}\n\n"
            f"Декомпозиция запроса:\n{decomposition_block}\n\n"
            "Финальный ответ на запрос пользователя на основе полученных декомпозиционных ответов:"
        )
        response = self.llm.parse(prompt, V3FinalAnswer)
        return response.answer

    def _update_last(
        self,
        question: str,
        plan: DecompositionPlan,
        sub_answers: List[dict],
        answer: str,
    ) -> None:
        self.last = {
            "question": question,
            "plan": plan.model_dump(),
            "sub_answers": sub_answers,
            "answer": answer,
        }

    def _log_answer(self, question: str, plan: DecompositionPlan, answer: str) -> None:
        if self.logger is None:
            return

        lines = [question, "Decompose:"]
        if self._should_decompose(plan):
            lines.extend([f"  - {query}" for query in plan.decomposition_queries])
        else:
            lines.append("  - decomposition not used")
        lines.append(f"Answer: {answer}")
        self.logger.info("\n".join(lines))
