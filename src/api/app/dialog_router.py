from __future__ import annotations

from typing import Iterable, Optional

from pydantic import BaseModel, Field

from ...llm.llm import BaseLLM


DEFAULT_DIALOG_ROUTER_PROMPT = """
Ты маршрутизатор диалога для ассистента по документам базы знаний.

Твоя задача: по истории диалога и последнему сообщению пользователя решить, можно ли ответить уже из контекста диалога, или нужно искать ответ в документах.

Правила:
1. Если пользователь не задал содержательный вопрос, а только поздоровался, написал короткую реплику без запроса или явно ждёт начала работы, ответь сам.
2. В таком случае сообщи, что ты ассистент для поиска по документам текущей базы знаний, и попроси сформулировать вопрос.
3. Если в истории диалога уже есть достаточная информация для ответа на последний вопрос пользователя, ответь сам и не отправляй запрос в поиск.
4. Если в истории недостаточно данных, сформируй конкретный поисковый вопрос для ассистента поиска.
5. Не придумывай факты, которых нет в истории диалога.
6. Если выбираешь search_required, search_question должен быть коротким, конкретным и самодостаточным.
7. Если выбираешь respond_from_context, assistant_response должен быть готовым ответом для пользователя.
""".strip()


class DialogRouterDecision(BaseModel):
    action: str = Field(..., pattern="^(respond_from_context|search_required)$")
    assistant_response: Optional[str] = None
    search_question: Optional[str] = None


def _format_messages(messages: Iterable[dict]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "assistant")).upper()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts).strip()


def route_dialog_turn(
    llm: BaseLLM,
    *,
    messages: list[dict],
    user_message: str,
) -> DialogRouterDecision:
    transcript = _format_messages(messages)
    prompt = f"""
{DEFAULT_DIALOG_ROUTER_PROMPT}

История диалога:
{transcript or "<empty>"}

Последнее сообщение пользователя:
{user_message}
""".strip()
    decision = llm.parse(prompt, DialogRouterDecision, temperature=0)
    if decision.action == "respond_from_context":
        response = (decision.assistant_response or "").strip()
        if not response:
            return DialogRouterDecision(
                action="respond_from_context",
                assistant_response="Я ассистент для поиска по документам текущей базы знаний. Сформулируйте, пожалуйста, ваш вопрос.",
            )
        return DialogRouterDecision(
            action="respond_from_context",
            assistant_response=response,
        )
    search_question = (decision.search_question or "").strip()
    if not search_question:
        search_question = user_message.strip()
    return DialogRouterDecision(
        action="search_required",
        search_question=search_question,
    )
