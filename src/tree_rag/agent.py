from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Type, Union

from pydantic import BaseModel, Field, create_model

from src.llm.llm import OpenAILLM
from src.tree_rag.index import TreeIndex


DEFAULT_TREE_AGENT_SYSTEM_PROMPT = """
Твоя задача - ответить на вопрос пользователя по документам из индекса.

Как работать:
- Сначала сформируй короткий план из 2-5 пунктов с метками [todo] и [done].
- В current_goal указывай короткий поисковый запрос для текущего шага.
- current_goal используется для поиска семантически близких текстов в индексе.
- Поэтому current_goal должен быть короткой смысловой формулировкой того, что нужно найти в тексте.
- current_goal должен быть похож на естественный поисковый запрос, а не на инструкцию.
- Пиши в current_goal тему или условие поиска, например: "осадки при оттаивании грунтов основания" или "грунты кроме слабых и просадочных".
- Не пиши в current_goal навигационные команды вроде "открыть документ", "просмотреть раздел", "найти узел".
- Не пиши в current_goal названия конкретных файлов, номера документов, индексы узлов и технические команды.
- Не меняй current_goal без необходимости. Если уже найден хороший релевантный текст, удерживай goal стабильным и используй его для короткой верификации.
- В facts записывай только подтвержденные результаты просмотра текста.
- Нельзя записывать в facts гипотезы навигации, догадки, фразы "может содержать", "раздел выглядит релевантным" и подобные.
- Нельзя помечать пункт плана как [done], если в facts нет подтвержденной информации, закрывающей этот пункт.
- Сам факт открытия документа или узла не означает выполнение пункта плана.

Формат facts:
- Для факта из узла используй формат:
  <имя документа>, узел <idx>: <факт>
- Для безрезультатного первичного поиска по документу используй формат:
  <имя документа>, документ: выполнен первичный поиск, полезной информации не найдено

Про view:
- В общем списке документов метка [attention] означает, что документ попал в top результатов поиска по current_goal.
- В содержании документа метка [attention] означает, что этот leaf или соседние leaf выглядят потенциально полезными для current_goal.
- Метка [visited] означает, что узел или документ уже просматривался.
- Метка [attention] не гарантирует наличие ответа.
- При выборе документа сначала смотри на описание документа и список candidate_documents.
- При выборе узла сначала смотри на название раздела.

Про стратегию поиска:
- Не зацикливайся на одном документе.
- Если в документе просмотрены несколько узлов, но полезных фактов по вопросу не найдено, переходи к следующему candidate document.
- Если документ уже отмечен как безрезультатный, не возвращайся к нему без очень сильной причины.
- Если есть candidate documents, сначала проверяй их, а уже потом любые другие документы.
- Если уже найден прямой ответ в одном сильном узле, не запускай новый широкий поиск без необходимости.
- После первого сильного факта переходи в режим короткой верификации: проверь не более 1-2 других candidate documents, чтобы убедиться, что нет более прямой или противоречащей информации.
- Если дополнительные candidate documents не дают более релевантной информации и не противоречат найденному факту, можно формировать final_answer.

Про final_answer:
- Заполняй final_answer только когда facts достаточно для ответа.
- Если остаются непроверенные candidate documents или attention-узлы, не делай ранний final_answer.
- Если final_answer заполнен, поле view заполнять нельзя.
- Если точного ответа в документах нет, честно скажи, что подтвержденной информации не найдено.
- Не добавляй ничего, чего нет в facts.
- Финальный ответ должен быть чистым пользовательским ответом, а не отчетом о поиске.
- Не упоминай во final_answer внутреннюю механику поиска: узлы, candidate documents, attention, visited, current_goal, верификацию, первичный поиск.
- Не пиши во final_answer ссылки вида "документ X", "узел Y", "в документе проверено", "дополнительно проверен".
- Не добавляй блоки "Основание", "Источник", "Дополнительно проверен", если пользователь прямо этого не просил.
- Если вопрос просит перечисление, дай только само перечисление и краткую формулировку ответа без служебных пояснений.
""".strip()


DEFAULT_TREE_AGENT_SYNTHESIS_PROMPT = """
Тебе дан вопрос пользователя и список подтвержденных фактов.
Сформируй краткий итоговый ответ.

Правила:
- Используй только information из facts.
- Убери повторы и служебные формулировки.
- Если facts не содержат прямого ответа, честно скажи, что подтвержденной информации для ответа не найдено.
- Не добавляй догадок.
- Сформируй чистый ответ для пользователя, а не внутренний отчет агента.
- Не упоминай узлы, индексы, candidate documents, attention, visited, поиск, верификацию, "дополнительно проверен" и другие служебные детали.
- Не добавляй блоки "Основание" или "Источник", если пользователь этого не просил.
""".replace("information", "информацию").strip()


class AddFacts(BaseModel):
    action: Literal["add"] = "add"
    content: str


class EditFacts(BaseModel):
    action: Literal["edit"] = "edit"
    content: str


class OpenDocumentAction(BaseModel):
    action: Literal["open_document"] = "open_document"
    document_idx: int


class OpenLeafAction(BaseModel):
    action: Literal["open_leaf"] = "open_leaf"
    leaf_idx: int


class OpenAllDocsAction(BaseModel):
    action: Literal["open_all_docs"] = "open_all_docs"


class BackToDocumentAction(BaseModel):
    action: Literal["back_to_document"] = "back_to_document"


@dataclass
class TreeAgentSession:
    question: str
    current_plan: str = ""
    current_goal: str = ""
    facts: str = ""
    view_state: Literal["all_docs", "doc", "node"] = "all_docs"
    current_document_idx: Optional[int] = None
    current_leaf_idx: Optional[int] = None
    visited_nodes: Dict[int, Set[int]] = field(default_factory=dict)
    attention_hits: Dict[int, List[int]] = field(default_factory=dict)
    last_search_goal: str = ""
    positive_facts_by_doc: Dict[int, int] = field(default_factory=dict)
    no_info_recorded_docs: Set[int] = field(default_factory=set)
    exhausted_docs: Set[int] = field(default_factory=set)
    verification_docs: Set[int] = field(default_factory=set)


class TreeAgent:
    def __init__(
        self,
        client,
        model_name: str,
        tree_index: TreeIndex,
        system_prompt: str = DEFAULT_TREE_AGENT_SYSTEM_PROMPT,
        max_iters: int = 30,
        top_k: int = 3,
        logger: Optional[logging.Logger] = None,
        max_logged_view_lines: int = 40,
        max_logged_text_chars: int = 2000,
        synthesis_prompt: str = DEFAULT_TREE_AGENT_SYNTHESIS_PROMPT,
    ) -> None:
        self.llm = OpenAILLM(client=client, model_name=model_name)
        self.tree_index = tree_index
        self.system_prompt = system_prompt
        self.synthesis_prompt = synthesis_prompt
        self.max_iters = max_iters
        self.top_k = top_k
        self.logger = logger or logging.getLogger(__name__)
        self.max_logged_view_lines = max_logged_view_lines
        self.max_logged_text_chars = max_logged_text_chars

    def answer(self, question: str) -> str:
        session = TreeAgentSession(question=question)

        for iteration in range(self.max_iters):
            if session.current_goal and session.current_goal != session.last_search_goal:
                session.attention_hits = self.tree_index.search(
                    session.current_goal,
                    top_k=self.top_k,
                )
                session.last_search_goal = session.current_goal
                self.logger.info("Search goal: %s", session.current_goal)
                self.logger.info(
                    "Attention hits: %s",
                    self._format_attention_hits(session.attention_hits),
                )

            show_view = iteration > 0
            tree_view = self._render_view(session) if show_view else ""
            schema = self._make_iteration_schema(session, include_view=show_view)
            prompt = self._build_prompt(session, tree_view=tree_view, include_view=show_view)

            self.logger.info(self._format_context(iteration, session, tree_view, show_view))
            old_plan = session.current_plan
            old_facts = session.facts
            response = self.llm.parse(
                f"{self.system_prompt}\n\n{prompt}",
                schema,
                temperature=0,
            )

            self._apply_iteration_response(session, response, include_view=show_view)
            self.logger.info(self._format_response(response))
            self._log_plan_warning(old_plan, session.current_plan, old_facts, session.facts)

            final_answer = getattr(response, "final_answer", None)
            if final_answer and self._should_accept_final_answer(session):
                return final_answer.strip()
            if final_answer:
                self.logger.warning(
                    "Rejected early final_answer because candidate documents or attention nodes remain"
                )
                self._prepare_next_step_after_rejected_final_answer(session)

        if session.facts.strip():
            return self._synthesize_answer(session)
        return "Подтвержденной информации не найдено."

    def _make_iteration_schema(
        self,
        session: TreeAgentSession,
        include_view: bool,
    ) -> Type[BaseModel]:
        fields = {
            "current_plan": (Optional[str], Field(default=None)),
            "current_goal": (Optional[str], Field(default=None)),
            "facts": (Optional[Union[AddFacts, EditFacts]], Field(default=None)),
            "final_answer": (Optional[str], Field(default=None)),
        }

        if include_view:
            if session.view_state == "all_docs":
                max_idx = max(1, self.tree_index.max_document_idx())
                view_model = create_model(
                    "AllDocsViewAction",
                    __base__=OpenDocumentAction,
                    document_idx=(int, Field(ge=1, le=max_idx)),
                )
            elif session.view_state == "doc":
                document = self.tree_index.get_document(session.current_document_idx or 1)
                max_idx = max(1, document.max_leaf_idx())
                open_leaf_model = create_model(
                    "DocOpenLeafAction",
                    __base__=OpenLeafAction,
                    leaf_idx=(int, Field(ge=1, le=max_idx)),
                )
                view_model = Union[open_leaf_model, OpenAllDocsAction]
            else:
                view_model = BackToDocumentAction
            fields["view"] = (Optional[view_model], Field(default=None))

        return create_model("TreeAgentIteration", **fields)

    def _build_prompt(
        self,
        session: TreeAgentSession,
        tree_view: str,
        include_view: bool,
    ) -> str:
        parts = [
            f"Question:\n{session.question}",
            f"Current plan:\n{session.current_plan or '<empty>'}",
            f"Current goal:\n{session.current_goal or '<empty>'}",
            f"Facts:\n{session.facts or '<empty>'}",
            f"Candidate documents:\n{self._render_candidate_documents(session)}",
        ]

        if include_view:
            parts.append(f"View state:\n{self._view_label(session)}")
            if session.current_document_idx is not None:
                visited = len(session.visited_nodes.get(session.current_document_idx, set()))
                positive = session.positive_facts_by_doc.get(session.current_document_idx, 0)
                exhausted = "yes" if session.current_document_idx in session.exhausted_docs else "no"
                parts.append(
                    "Current document stats:\n"
                    f"visited_nodes={visited}\n"
                    f"positive_facts={positive}\n"
                    f"exhausted={exhausted}"
                )
            parts.append(f"Tree view:\n{tree_view}")
        else:
            parts.append(
                "View state:\nНа первой итерации view еще не открыт. Сначала сформируй plan и goal."
            )

        return "\n\n".join(parts)

    def _render_candidate_documents(self, session: TreeAgentSession) -> str:
        if not session.attention_hits:
            return "<empty>"
        lines: List[str] = []
        for document_idx in self._candidate_document_indices(session):
            document = self.tree_index.get_document(document_idx)
            status: List[str] = []
            if document_idx in session.exhausted_docs:
                status.append("exhausted")
            elif document_idx in session.visited_nodes:
                status.append("visited")
            else:
                status.append("pending")
            leafs = session.attention_hits.get(document_idx, [])
            leaf_preview = ", ".join(str(leaf_idx) for leaf_idx in leafs[:5])
            description = document.document_description or "<no description>"
            lines.append(
                f"[{document_idx}] [{', '.join(status)}] {document.source_name}: "
                f"attention_leafs={leaf_preview or '-'}: {description}"
            )
        return "\n".join(lines)

    def _candidate_document_indices(self, session: TreeAgentSession) -> List[int]:
        ordered = [document_idx for document_idx in session.attention_hits if document_idx not in session.exhausted_docs]
        pending = [document_idx for document_idx in ordered if document_idx not in session.visited_nodes]
        revisitable = [document_idx for document_idx in ordered if document_idx in session.visited_nodes]
        return pending + revisitable

    def _render_view(self, session: TreeAgentSession) -> str:
        if session.view_state == "all_docs":
            return self.tree_index.llm_view(labels=self._doc_labels(session))
        if session.view_state == "doc":
            document_idx = session.current_document_idx
            if document_idx is None:
                return self.tree_index.llm_view(labels=self._doc_labels(session))
            return self.tree_index.llm_view_document(
                document_idx,
                labels=self._leaf_labels(session, document_idx),
            )
        if session.current_document_idx is None or session.current_leaf_idx is None:
            return ""
        return self.tree_index.get_text(session.current_document_idx, session.current_leaf_idx)

    def _doc_labels(self, session: TreeAgentSession) -> Dict[int, List[str]]:
        labels: Dict[int, List[str]] = {}
        for document_idx in session.attention_hits:
            labels.setdefault(document_idx, []).append("attention")
        for document_idx, leafs in session.visited_nodes.items():
            if leafs:
                labels.setdefault(document_idx, []).append("visited")
        return labels

    def _leaf_labels(self, session: TreeAgentSession, document_idx: int) -> Dict[int, List[str]]:
        labels: Dict[int, List[str]] = {}
        for leaf_idx in session.attention_hits.get(document_idx, []):
            labels.setdefault(leaf_idx, []).append("attention")
        for leaf_idx in session.visited_nodes.get(document_idx, set()):
            labels.setdefault(leaf_idx, []).append("visited")
        return labels

    def _apply_iteration_response(
        self,
        session: TreeAgentSession,
        response: BaseModel,
        include_view: bool,
    ) -> None:
        previous_document_idx = session.current_document_idx
        current_plan = getattr(response, "current_plan", None)
        current_goal = getattr(response, "current_goal", None)
        facts_action = getattr(response, "facts", None)

        if current_plan is not None:
            session.current_plan = current_plan.strip()
        if current_goal is not None:
            session.current_goal = self._sanitize_goal(current_goal.strip(), session)
        if facts_action is not None:
            self._apply_facts_action(session, facts_action)

        if not include_view:
            return
        if getattr(response, "final_answer", None):
            return

        view_action = getattr(response, "view", None)
        if view_action is None:
            return

        if isinstance(view_action, OpenDocumentAction):
            chosen_document_idx = self._resolve_document_choice(session, view_action.document_idx)
            if previous_document_idx is not None and previous_document_idx != chosen_document_idx:
                self._record_no_info_document_fact(session, previous_document_idx)
                if session.facts.strip():
                    session.verification_docs.add(previous_document_idx)
            session.view_state = "doc"
            session.current_document_idx = chosen_document_idx
            session.current_leaf_idx = None
            return

        if isinstance(view_action, OpenAllDocsAction):
            if previous_document_idx is not None:
                self._record_no_info_document_fact(session, previous_document_idx)
                if session.facts.strip():
                    session.verification_docs.add(previous_document_idx)
            session.view_state = "all_docs"
            session.current_document_idx = None
            session.current_leaf_idx = None
            return

        if isinstance(view_action, OpenLeafAction):
            if session.current_document_idx is None:
                return
            if self._document_should_be_abandoned(session, session.current_document_idx):
                self._record_no_info_document_fact(session, session.current_document_idx)
                session.view_state = "all_docs"
                session.current_document_idx = None
                session.current_leaf_idx = None
                self.logger.warning(
                    "Forced switch to all_docs because document %s produced no positive facts",
                    previous_document_idx,
                )
                return
            session.view_state = "node"
            session.current_leaf_idx = view_action.leaf_idx
            session.visited_nodes.setdefault(session.current_document_idx, set()).add(
                view_action.leaf_idx
            )
            return

        if isinstance(view_action, BackToDocumentAction) and session.current_document_idx is not None:
            session.view_state = "doc"
            session.current_leaf_idx = None

    def _resolve_document_choice(self, session: TreeAgentSession, requested_idx: int) -> int:
        candidate_docs = self._candidate_document_indices(session)
        pending_candidates = [
            document_idx for document_idx in candidate_docs if document_idx not in session.visited_nodes
        ]
        if requested_idx in session.exhausted_docs and candidate_docs:
            chosen = candidate_docs[0]
            self.logger.warning(
                "Redirected exhausted document %s to candidate document %s",
                requested_idx,
                chosen,
            )
            return chosen
        if pending_candidates and requested_idx not in pending_candidates:
            chosen = pending_candidates[0]
            self.logger.warning(
                "Redirected document %s to next pending candidate document %s",
                requested_idx,
                chosen,
            )
            return chosen
        if candidate_docs and requested_idx not in candidate_docs:
            chosen = candidate_docs[0]
            self.logger.warning(
                "Redirected non-candidate document %s to top candidate document %s",
                requested_idx,
                chosen,
            )
            return chosen
        return requested_idx

    def _apply_facts_action(
        self,
        session: TreeAgentSession,
        facts_action: Union[AddFacts, EditFacts],
    ) -> None:
        old_lines = self._split_lines(session.facts)
        content = self._normalize_fact_lines(
            self._format_facts_content(session, facts_action.content)
        )
        content = self._filter_non_evidence_lines(content)
        if not content:
            return
        if isinstance(facts_action, AddFacts):
            session.facts = self._merge_lines(session.facts, content)
        else:
            session.facts = content
        new_lines = self._split_lines(session.facts)
        added_lines = [
            line for line in new_lines
            if self._line_key(line) not in {self._line_key(old_line) for old_line in old_lines}
        ]
        self._update_positive_fact_stats(session, added_lines)

    def _view_label(self, session: TreeAgentSession) -> str:
        if session.view_state == "all_docs":
            return "all_docs"
        if session.view_state == "doc":
            return f"doc(document_idx={session.current_document_idx})"
        return (
            f"node(document_idx={session.current_document_idx}, "
            f"leaf_idx={session.current_leaf_idx})"
        )

    def _format_context(
        self,
        iteration: int,
        session: TreeAgentSession,
        tree_view: str,
        include_view: bool,
    ) -> str:
        parts = [
            f"\n==================== Iter {iteration} ====================",
            f"Question:\n{session.question}",
            f"Plan:\n{session.current_plan or '<empty>'}",
            f"Goal:\n{session.current_goal or '<empty>'}",
            f"Facts:\n{session.facts or '<empty>'}",
            f"Candidate Documents:\n{self._render_candidate_documents(session)}",
        ]
        if include_view:
            parts.append(f"View State:\n{self._view_label(session)}")
            if session.current_document_idx is not None:
                visited = len(session.visited_nodes.get(session.current_document_idx, set()))
                positive = session.positive_facts_by_doc.get(session.current_document_idx, 0)
                exhausted = "yes" if session.current_document_idx in session.exhausted_docs else "no"
                parts.append(
                    "Document Stats:\n"
                    f"visited_nodes={visited}\n"
                    f"positive_facts={positive}\n"
                    f"exhausted={exhausted}"
                )
            parts.append(f"View:\n{self._compact_view_for_log(tree_view, session.view_state)}")
        return "\n\n".join(parts)

    def _format_response(self, response: BaseModel) -> str:
        payload = response.model_dump()
        lines = ["\n\nLLM Step:"]
        for key in ("current_plan", "current_goal", "facts", "final_answer", "view"):
            value = payload.get(key)
            if value is None:
                continue
            if key == "current_plan":
                lines.append(f"- current_plan:\n{value}")
                continue
            if key == "final_answer" and isinstance(value, str):
                lines.append(f"- final_answer: {self._truncate_text(value, 500)}")
                continue
            if key == "facts" and isinstance(value, dict):
                lines.append(
                    f"- facts: {value.get('action')} | "
                    f"{self._truncate_text(value.get('content', ''), 300)}"
                )
                continue
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _format_attention_hits(self, hits: Dict[int, List[int]]) -> str:
        if not hits:
            return "{}"
        parts = []
        for document_idx, leafs in sorted(hits.items()):
            preview = ", ".join(str(leaf) for leaf in leafs[:5])
            suffix = "..." if len(leafs) > 5 else ""
            parts.append(f"{document_idx}:[{preview}{suffix}]")
        return "{ " + "; ".join(parts) + " }"

    def _compact_view_for_log(self, tree_view: str, view_state: str) -> str:
        if not tree_view:
            return "<empty>"
        if view_state == "node":
            return self._truncate_text(tree_view, self.max_logged_text_chars)
        lines = tree_view.splitlines()
        if len(lines) <= self.max_logged_view_lines:
            return tree_view
        head = "\n".join(lines[: self.max_logged_view_lines])
        return f"{head}\n... ({len(lines) - self.max_logged_view_lines} more lines)"

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars].rstrip()}... ({len(text) - max_chars} more chars)"

    def _should_accept_final_answer(self, session: TreeAgentSession) -> bool:
        if not session.facts.strip():
            return False
        return self._count_positive_fact_lines(session.facts) > 0

    @staticmethod
    def _count_done_items(plan: str) -> int:
        return sum(1 for line in plan.splitlines() if line.strip().startswith("[done]"))

    def _log_plan_warning(
        self,
        old_plan: str,
        new_plan: str,
        old_facts: str,
        new_facts: str,
    ) -> None:
        if not new_plan:
            return
        old_done = self._count_done_items(old_plan or "")
        new_done = self._count_done_items(new_plan or "")
        if new_done > old_done and (new_facts or "").strip() == (old_facts or "").strip():
            self.logger.warning("Plan marks additional [done] items without new evidence")

    def _normalize_fact_lines(self, text: str) -> str:
        lines = self._split_lines(text)
        unique: List[str] = []
        seen: Set[str] = set()
        for line in lines:
            normalized = self._line_key(line)
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(line)
        return "\n".join(unique).strip()

    def _merge_lines(self, current: str, new_content: str) -> str:
        current_lines = self._split_lines(current)
        new_lines = self._split_lines(new_content)
        merged: List[str] = []
        seen: Set[str] = set()
        for line in current_lines + new_lines:
            normalized = self._line_key(line)
            if normalized in seen:
                continue
            seen.add(normalized)
            merged.append(line)
        return "\n".join(merged).strip()

    def _synthesize_answer(self, session: TreeAgentSession) -> str:
        class FinalSynthesis(BaseModel):
            answer: str

        prompt = (
            f"{self.synthesis_prompt}\n\n"
            f"Question:\n{session.question}\n\n"
            f"Facts:\n{session.facts.strip()}"
        )
        try:
            result = self.llm.parse(prompt, FinalSynthesis, temperature=0)
            answer = result.answer.strip()
            if answer:
                return answer
        except Exception as exc:
            self.logger.warning("Failed to synthesize final answer: %s", exc)
        return session.facts.strip()

    def _format_facts_content(self, session: TreeAgentSession, content: str) -> str:
        content = content.strip()
        if not content:
            return content
        if session.view_state != "node":
            return content
        if session.current_document_idx is None or session.current_leaf_idx is None:
            return content
        prefix = (
            f"{self._document_trace_name(session.current_document_idx)}, "
            f"узел {session.current_leaf_idx}: "
        )
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        formatted: List[str] = []
        for line in lines:
            if ", узел " in line or ", документ:" in line:
                formatted.append(line)
            else:
                formatted.append(f"{prefix}{line}")
        return "\n".join(formatted)

    def _filter_non_evidence_lines(self, content: str) -> str:
        navigation_markers = [
            "релевант",
            "может содерж",
            "проверить",
            "в оглавлении",
            "найден как",
            "выглядит",
            "возможно",
            "поискать",
        ]
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        filtered: List[str] = []
        for line in lines:
            lowered = line.lower()
            if any(marker in lowered for marker in navigation_markers):
                continue
            filtered.append(line)
        return self._normalize_fact_lines("\n".join(filtered))

    def _is_negative_fact(self, line: str) -> bool:
        lowered = line.lower()
        negative_markers = [
            "не содержит",
            "не указ",
            "не найден",
            "нет информации",
            "полезной информации не найдено",
            "не содержит информации",
        ]
        return any(marker in lowered for marker in negative_markers)

    def _update_positive_fact_stats(self, session: TreeAgentSession, added_lines: List[str]) -> None:
        if session.current_document_idx is None:
            return
        positive_count = 0
        for line in added_lines:
            if ", документ:" in line:
                continue
            if self._is_negative_fact(line):
                continue
            positive_count += 1
        if positive_count > 0:
            session.positive_facts_by_doc[session.current_document_idx] = (
                session.positive_facts_by_doc.get(session.current_document_idx, 0) + positive_count
            )

    def _document_should_be_abandoned(self, session: TreeAgentSession, document_idx: int) -> bool:
        visited = len(session.visited_nodes.get(document_idx, set()))
        positive = session.positive_facts_by_doc.get(document_idx, 0)
        return visited >= 2 and positive == 0

    def _record_no_info_document_fact(self, session: TreeAgentSession, document_idx: int) -> None:
        if document_idx in session.no_info_recorded_docs:
            return
        visited = len(session.visited_nodes.get(document_idx, set()))
        positive = session.positive_facts_by_doc.get(document_idx, 0)
        if visited == 0 or positive > 0:
            return
        line = (
            f"{self._document_trace_name(document_idx)}, документ: "
            "выполнен первичный поиск, полезной информации не найдено"
        )
        session.facts = self._merge_lines(session.facts, line)
        session.no_info_recorded_docs.add(document_idx)
        session.exhausted_docs.add(document_idx)
        if session.facts.strip():
            session.verification_docs.add(document_idx)

    def _prepare_next_step_after_rejected_final_answer(self, session: TreeAgentSession) -> None:
        if session.view_state == "node":
            session.view_state = "all_docs"
            session.current_leaf_idx = None
            session.current_document_idx = None
            return
        if session.view_state == "doc":
            session.view_state = "all_docs"
            session.current_document_idx = None
            session.current_leaf_idx = None

    @staticmethod
    def _split_lines(text: str) -> List[str]:
        return [line.strip() for line in text.splitlines() if line.strip()]

    @staticmethod
    def _line_key(line: str) -> str:
        return line.rstrip(".").strip()

    def _document_trace_name(self, document_idx: int) -> str:
        document = self.tree_index.get_document(document_idx)
        source_name = (document.source_name or "").strip()
        if source_name.lower().endswith(".pdf"):
            return source_name
        return f"{document_idx}:{document.root.title}"

    def _sanitize_goal(self, goal: str, session: TreeAgentSession) -> str:
        if not goal:
            return session.current_goal
        lowered = goal.lower()
        navigation_markers = [
            "открыть",
            "просмотреть",
            "документ",
            "раздел",
            "узел",
            ".pdf",
            "сп ",
            "свод правил",
        ]
        if any(marker in lowered for marker in navigation_markers):
            return session.current_goal or self._question_to_goal(session.question)
        if len(goal) > 140:
            return session.current_goal or self._question_to_goal(session.question)
        return goal

    @staticmethod
    def _question_to_goal(question: str) -> str:
        return question.strip().rstrip("?.!")

    def _count_positive_fact_lines(self, facts: str) -> int:
        count = 0
        for line in self._split_lines(facts):
            if ", документ:" in line:
                continue
            if self._is_negative_fact(line):
                continue
            count += 1
        return count
