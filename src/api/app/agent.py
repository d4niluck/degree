from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field, create_model

from ...indexing import Index
from ...llm.llm import BaseLLM


DEFAULT_BACKEND_V5_SYSTEM_PROMPT = """
Твоя задача - ответить на вопрос пользователя (user_query).
Ты работаешь как исследователь и оркестратор: сначала строишь план, затем решаешь подзадачи по одной.

Правила работы:
1. Если вопрос сложный, сначала декомпозируй его на 2-5 маленьких подзадач.
2. Для каждой подзадачи сначала используй IndexSearch с коротким и точным вопросом.
3. Используй IndexChunkLookup только если IndexSearch вернул partial_need_neighbors и дал ссылки на полезные чанки.
4. Не отправляй в поиск один большой сложный запрос, если его можно разбить на несколько простых.
5. Финальный ответ формируй только после того, как по всем ключевым подзадачам есть достаточно данных или явно установлено отсутствие данных.
6. Один вызов IndexSearch должен отвечать только за один аспект поиска.
7. Если нужно узнать несколько аспектов, делай несколько отдельных вызовов IndexSearch.

Требования к reasoning_steps:
- Это должен быть структурированный план исследования.
- Пиши пункты в формате [done]/[todo]/[next].
- Отмечай, какая подзадача уже решена, какая требует уточнения, и какой следующий шаг самый полезный.

Требования к next_steps:
- Каждый шаг должен быть маленьким и конкретным.
- Один шаг должен пытаться решить только одну подзадачу или одно уточнение.
- Для IndexSearch один шаг должен покрывать только одну процедуру и только один аспект.

Аспекты, которые нельзя смешивать в одном запросе IndexSearch:
- цель работ
- контроль и наблюдения
- фиксация в документации
- технологические параметры

Плохо:
- "для объекта X найти цель, контроль, документацию и параметры"
- "что известно про объект X по всем аспектам"

Хорошо:
- "какова цель работ для объекта X"
- "какие контроль и наблюдения требуются для объекта X"
- "что фиксируется в документации для объекта X"
- "какие технологические параметры прямо указаны для объекта X"

Если задача сравнивает две процедуры по четырем аспектам, сначала собери минимум восемь атомарных поисковых наблюдений: по одному на каждую пару процедура+аспект.

Требования к финальному ответу:
1. Отвечай конкретно на поставленный вопрос и не добавляй детали, о которых пользователь не спрашивал.
2. В финальном ответе может быть только та информация, которая реально подтверждена документами из контекста.
3. Если пользователь просит сравнение по конкретным критериям, выдай список, в котором каждый пункт посвящен одному требуемому критерию и внутри него сопоставлены сравниваемые объекты именно по этому критерию.
4. После такого списка не добавляй общее заключение, итог или вывод, если пользователь прямо этого не просил.
5. Если вопрос требует перечисления, перед выдачей ответа проверь, что перечислены все подтвержденные пункты из документов и не добавлено ничего сверх них.
6. Если точного ответа в документах нет, прямо скажи об отсутствии подтвержденной информации и не заполняй пробелы догадками.

task_completed - флаг завершения задачи, если достаточно информации и проделанных шагов, чтобы на основе контекста ответить на запрос пользователя.
final_answer - финальный ответ, если задача решена.
Ответ должен формироваться только на основе информации, которая предоставлена в документах. Запрещается выдумывать информацию, которой в документах нет.
""".strip()

INDEX_SEARCH_TOOL_DESCRIPTION = """
Search the index for one narrow sub-question and return a compact synthesis.
Use this tool for exactly one atomic information need.
One call must target only one procedure/topic and only one aspect.
Never combine multiple aspects in one query.
The result contains:
- status=answer_found if the retrieved chunks contain a direct answer
- status=partial_need_neighbors if there is related evidence but adjacent chunks should be checked
- status=not_found if relevant evidence was not found
- answer: a concise synthesis strictly grounded in the retrieved chunks
- useful_chunks: document_name + chunk_index references that support the answer or need follow-up
""".strip()

INDEX_CHUNK_LOOKUP_TOOL_DESCRIPTION = """
Retrieve exact chunks by document_name and chunk_indices.
Use this only after IndexSearch points to specific useful chunks that need closer inspection.
""".strip()


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Tool(BaseModel):
    name: str
    description: str
    input_model: Type[ToolInput]
    handler: Callable[[ToolInput], Any]

    def make_step_model(self) -> Type[BaseModel]:
        return create_model(
            f"{self.name}Step",
            __config__=ConfigDict(extra="forbid"),
            tool_name=(Literal[self.name], ...),
            tool_args=(self.input_model, ...),
            tool_reason=(str, ...),
        )


class IndexSearchInput(ToolInput):
    query: str = Field(..., min_length=5, max_length=500)


class IndexChunkLookupInput(ToolInput):
    document_name: str
    chunk_indices: List[int] = Field(..., min_length=1, max_length=10)


class ChunkReference(BaseModel):
    document_name: str
    chunk_index: int
    page_number: Optional[int] = None
    note: str = Field(..., max_length=40)


class SearchSynthesisResult(BaseModel):
    status: Literal["answer_found", "partial_need_neighbors", "not_found"]
    answer: str = Field(..., max_length=400)
    useful_chunks: List[ChunkReference] = Field(default_factory=list, max_length=5)


class ChunkText(BaseModel):
    document_name: str
    chunk_index: int
    page_number: Optional[int] = None
    text: str


class ChunkLookupResult(BaseModel):
    status: Literal["chunks_found", "not_found"]
    chunks: List[ChunkText] = Field(default_factory=list, max_length=10)


class StepResult(BaseModel):
    step: Any
    tool_result: Any
    success: bool


def make_plan_model(tools: List[Tool]) -> Type[BaseModel]:
    step_models = [tool.make_step_model() for tool in tools]
    return create_model(
        "BackendPlan",
        reasoning_steps=(List[str], Field(..., min_length=2, max_length=5)),
        next_steps=(List[Union[tuple(step_models)]], Field(..., min_length=1, max_length=3)),
        task_completed=(bool, Field(...)),
        final_answer=(str | None, Field(None)),
    )


def _format_search_results(results: List[Any]) -> str:
    if not results:
        return "No search results found."
    payloads: List[str] = []
    for i, result in enumerate(results):
        payloads.append(
            f"{i}. Document: {Path(result.chunk['source_path']).name}, "
            f"page: {result.chunk['page_number']}, "
            f"chunk_index: {result.chunk['index']}\n"
            f"Text: {result.chunk['text']}"
        )
    return "\n\n".join(payloads)


def _serialize_for_context(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return {
            key: _serialize_for_context(item)
            for key, item in value.model_dump(mode="json").items()
        }
    if isinstance(value, dict):
        return {key: _serialize_for_context(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_for_context(item) for item in value]
    return value


def _summarize_tool_result(result: Any) -> Any:
    if isinstance(result, SearchSynthesisResult):
        return result.model_dump(mode="json")
    if isinstance(result, ChunkLookupResult):
        chunks = []
        for chunk in result.chunks:
            text = chunk.text[:500] + ("..." if len(chunk.text) > 500 else "")
            chunks.append(
                {
                    "document_name": chunk.document_name,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "text": text,
                }
            )
        return {"status": result.status, "chunks": chunks}
    return _serialize_for_context(result)


def _build_search_synthesis_prompt(query: str, search_results: str) -> str:
    return f"""
Ты получаешь узкий поисковый вопрос и результаты поиска по документам.
Твоя задача - сделать компактный структурированный вывод строго по найденным фрагментам.

Вопрос:
{query}

Результаты поиска:
{search_results}

Правила:
1. Если найден прямой ответ, верни status=answer_found.
2. Если есть частично релевантные фрагменты, но нужно посмотреть соседние чанки для точного ответа, верни status=partial_need_neighbors.
3. Если релевантной информации нет, верни status=not_found.
4. В answer дай максимально короткую выжимку по существу: 1-3 коротких предложения без лишних деталей.
5. Не цитируй длинные фрагменты, не пересказывай весь поиск и не перечисляй все найденные детали.
6. Если точного полного ответа нет, кратко скажи чего именно не хватает.
7. В useful_chunks укажи только реально полезные чанки из результатов поиска.
8. Поле note должно быть очень коротким: не более 5 слов.
9. Не придумывай document_name, chunk_index и page_number: бери только из результатов поиска.
""".strip()


def _normalize_chunk_refs(
    chunk_refs: List[ChunkReference],
    available_chunks: Dict[tuple[str, int], Dict[str, Any]],
) -> List[ChunkReference]:
    normalized: List[ChunkReference] = []
    seen: set[tuple[str, int]] = set()
    for chunk_ref in chunk_refs:
        key = (chunk_ref.document_name, chunk_ref.chunk_index)
        available_chunk = available_chunks.get(key)
        if available_chunk is None or key in seen:
            continue
        seen.add(key)
        normalized.append(
            ChunkReference(
                document_name=chunk_ref.document_name,
                chunk_index=chunk_ref.chunk_index,
                page_number=available_chunk["page_number"],
                note=chunk_ref.note,
            )
        )
    return normalized


def build_index_search_tool(
    index: Index,
    llm: BaseLLM,
    logger: Optional[logging.Logger] = None,
    top_k_dense: int = 3,
    top_k_bm25: int = 3,
    source_paths: Optional[List[str]] = None,
) -> Tool:
    def handler(args: IndexSearchInput) -> SearchSynthesisResult:
        results = index.search_hybrid(
            query=args.query,
            top_k_dense=top_k_dense,
            top_k_bm25=top_k_bm25,
            source_paths=source_paths,
        )
        if not results:
            return SearchSynthesisResult(
                status="not_found",
                answer="Релевантные документы по этому подзапросу не найдены.",
                useful_chunks=[],
            )
        if logger is not None:
            bm25_hits = sum(1 for result in results if getattr(result, "extra", None) == "bm25")
            dense_hits = len(results) - bm25_hits
            logger.info(
                "INDEX SEARCH\nQuery: %s\nHits total: %s | dense: %s | bm25: %s",
                args.query,
                len(results),
                dense_hits,
                bm25_hits,
            )
        synthesis = llm.parse(
            _build_search_synthesis_prompt(args.query, _format_search_results(results)),
            SearchSynthesisResult,
        )
        available = {
            (Path(result.chunk["source_path"]).name, int(result.chunk["index"])): {
                "page_number": result.chunk["page_number"],
            }
            for result in results
        }
        return SearchSynthesisResult(
            status=synthesis.status,
            answer=synthesis.answer,
            useful_chunks=_normalize_chunk_refs(synthesis.useful_chunks, available),
        )

    return Tool(
        name="IndexSearch",
        description=INDEX_SEARCH_TOOL_DESCRIPTION,
        input_model=IndexSearchInput,
        handler=handler,
    )


def build_index_chunk_lookup_tool(
    index: Index,
    source_paths: Optional[List[str]] = None,
) -> Tool:
    allowed_paths = {str(Path(path).resolve()) for path in source_paths or []}

    def handler(args: IndexChunkLookupInput) -> ChunkLookupResult:
        source_path = index.resolve_source_path(args.document_name)
        if source_path is None:
            return ChunkLookupResult(status="not_found", chunks=[])
        if allowed_paths and str(Path(source_path).resolve()) not in allowed_paths:
            return ChunkLookupResult(status="not_found", chunks=[])
        chunks = index.get_chunks(
            source_path=source_path,
            chunk_indices=sorted(set(args.chunk_indices)),
        )
        if not chunks:
            return ChunkLookupResult(status="not_found", chunks=[])
        return ChunkLookupResult(
            status="chunks_found",
            chunks=[
                ChunkText(
                    document_name=Path(chunk["source_path"]).name,
                    chunk_index=int(chunk["index"]),
                    page_number=chunk["page_number"],
                    text=chunk["text"],
                )
                for chunk in chunks
            ],
        )

    return Tool(
        name="IndexChunkLookup",
        description=INDEX_CHUNK_LOOKUP_TOOL_DESCRIPTION,
        input_model=IndexChunkLookupInput,
        handler=handler,
    )


def build_backend_v5_tools(
    index: Index,
    llm: BaseLLM,
    max_top_k_dense: int = 3,
    max_top_k_bm25: int = 3,
    source_paths: Optional[List[str]] = None,
) -> List[Tool]:
    return [
        build_index_search_tool(
            index,
            llm,
            logger=index.logger,
            top_k_dense=max_top_k_dense,
            top_k_bm25=max_top_k_bm25,
            source_paths=source_paths,
        ),
        build_index_chunk_lookup_tool(index, source_paths=source_paths),
    ]


class BackendV5Agent:
    def __init__(
        self,
        llm: BaseLLM,
        tools: List[Tool],
        max_iters: int = 20,
        history_limit: int = 50,
        logger: Optional[logging.Logger] = None,
        system_prompt: str = DEFAULT_BACKEND_V5_SYSTEM_PROMPT,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools
        self.tools_map: Dict[str, Tool] = {tool.name: tool for tool in tools}
        self.max_iters = max_iters
        self.history_limit = history_limit
        self.logger = logger or logging.getLogger(__name__)
        self.schema = make_plan_model(tools)
        self._reset_context()

    def answer(
        self,
        question: str,
    ) -> str:
        answer = "No info"
        for event in self.iter_events(question):
            if event["type"] == "final_answer":
                answer = str(event["answer"])
        return answer

    def iter_events(self, question: str) -> Generator[Dict[str, Any], None, None]:
        self._reset_context()
        self._user_query = question
        try:
            for iteration in range(1, self.max_iters + 1):
                yield {"type": "iteration", "iteration": iteration}
                reasoning = self._reasoning()
                if reasoning.task_completed:
                    answer = reasoning.final_answer or "No info"
                    yield {"type": "final_answer", "answer": answer}
                    return
                if reasoning.next_steps:
                    self._action(reasoning.next_steps[0])
            yield {"type": "final_answer", "answer": "No info"}
        finally:
            self._reset_context()

    def _reset_context(self) -> None:
        self._history: List[StepResult] = []
        self._user_query: Optional[str] = None

    def _push_history(self, entry: StepResult) -> None:
        self._history.append(entry)
        if len(self._history) > self.history_limit:
            self._history = self._history[-self.history_limit :]

    def _build_context(self) -> str:
        payload = {
            "system_instructions": self.system_prompt,
            "tools": [{"name": tool.name, "description": tool.description} for tool in self.tools],
            "user_query": self._user_query,
            "actions_history": _serialize_for_context(self._history),
        }
        return json.dumps(payload, ensure_ascii=False, default=str)

    def _reasoning(self) -> BaseModel:
        return self.llm.parse(self._build_context(), self.schema)

    def _action(self, step: BaseModel) -> None:
        tool = self.tools_map.get(step.tool_name)
        if tool is None:
            result = {"error": f"Unknown tool: {step.tool_name}"}
            success = False
        else:
            try:
                result = tool.handler(step.tool_args)
                success = True
            except Exception as exc:
                self.logger.error(exc)
                result = {"error": f"Tool error: {exc}"}
                success = False
        self._push_history(
            StepResult(
                step=_serialize_for_context(step),
                tool_result=_summarize_tool_result(result),
                success=success,
            )
        )


def create_backend_v5_agent(
    *,
    index: Index,
    llm: BaseLLM,
    logger: Optional[logging.Logger] = None,
    system_prompt: str = DEFAULT_BACKEND_V5_SYSTEM_PROMPT,
    max_iters: int = 20,
    history_limit: int = 50,
    max_top_k_dense: int = 3,
    max_top_k_bm25: int = 3,
    source_paths: Optional[List[str]] = None,
) -> BackendV5Agent:
    tools = build_backend_v5_tools(
        index=index,
        llm=llm,
        max_top_k_dense=max_top_k_dense,
        max_top_k_bm25=max_top_k_bm25,
        source_paths=source_paths,
    )
    return BackendV5Agent(
        llm=llm,
        tools=tools,
        max_iters=max_iters,
        history_limit=history_limit,
        logger=logger or logging.getLogger(__name__),
        system_prompt=system_prompt,
    )
