from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union

from opensearchpy import OpenSearch
from pydantic import BaseModel, ConfigDict, Field, create_model

from .base import BaseAgent
from ..indexing import ChunkStore, DataStore, FlatVectorStore, Index, OpenSearchBM25Store
from ..llm.llm import BaseLLM, OpenAILLM
from ..preprocessing import HTTPEmbedder, HTTPReader, ParagraphChunker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_STORE_DIR = PROJECT_ROOT / "data/index/documents"
DEFAULT_CHUNK_STORE_DIR = PROJECT_ROOT / "data/index/chunks"
DEFAULT_VECTOR_STORE_DIR = PROJECT_ROOT / "data/index/vectors"
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "data/index/db"

DEFAULT_V4_SYSTEM_PROMPT = """
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

Allowed examples:
- "какова цель работ для объекта X?"
- "какие контроль и наблюдения требуются для объекта X?"
- "что фиксируется в документации для объекта X?"
- "какие технологические параметры прямо указаны для объекта X?"

Forbidden examples:
- "какова цель, контроль, документация и параметры для объекта X?"
- "собери все сведения по объекту X"
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
    """Search the index for one narrow question."""

    query: str = Field(
        ...,
        description="One narrow sub-question to search in the documents",
        min_length=5,
        max_length=500,
    )


class IndexChunkLookupInput(ToolInput):
    """Get exact chunks by document name and chunk indices."""

    document_name: str = Field(
        ...,
        description="Document file name or source path from the index",
    )
    chunk_indices: List[int] = Field(
        ...,
        description="Chunk indices to retrieve",
        min_length=1,
        max_length=10,
    )


class ChunkReference(BaseModel):
    document_name: str = Field(..., description="Document file name")
    chunk_index: int = Field(..., description="Chunk index inside the document")
    page_number: Optional[int] = Field(None, description="Page number if known")
    note: str = Field(..., description="Very short reason why this chunk is useful", max_length=40)


class SearchSynthesisResult(BaseModel):
    status: Literal["answer_found", "partial_need_neighbors", "not_found"]
    answer: str = Field(
        ...,
        description="Compact grounded answer or explanation of what is still missing",
        max_length=400,
    )
    useful_chunks: List[ChunkReference] = Field(
        default_factory=list,
        description="Chunk references that support the answer or should be inspected next",
        max_length=5,
    )


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
        "Plan",
        reasoning_steps=(
            List[str],
            Field(
                ...,
                description="Размышление над задачей для выбора следующего действия",
                min_length=2,
                max_length=5,
            ),
        ),
        next_steps=(
            List[Union[tuple(step_models)]],
            Field(
                ...,
                description="Список следующих шагов",
                min_length=1,
                max_length=3,
            ),
        ),
        task_completed=(bool, Field(..., description="Флаг, завершена ли задача")),
        final_answer=(str | None, Field(None, description="Финальный ответ, если task_completed")),
    )


def _format_search_results(results: List[Any]) -> str:
    if not results:
        return "No search results found."

    formatted_results: List[str] = []
    for i, result in enumerate(results):
        source_doc = Path(result.chunk["source_path"]).name
        payload = (
            f"{i}. Document: {source_doc}, "
            f"page: {result.chunk['page_number']}, "
            f"chunk_index: {result.chunk['index']}\n"
            f"Text: {result.chunk['text']}"
        )
        formatted_results.append(payload)
    return "\n\n".join(formatted_results)


def _format_search_hits_for_log(results: List[Any]) -> str:
    if not results:
        return "No search hits"

    lines: List[str] = []
    for i, result in enumerate(results):
        lines.append(
            f"{i}. document={Path(result.chunk['source_path']).name}, "
            f"page={result.chunk['page_number']}, "
            f"chunk_index={result.chunk['index']}, "
            f"score={result.score:.4f}"
        )
    return "\n".join(lines)


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
        summarized_chunks = []
        for chunk in result.chunks:
            excerpt = chunk.text[:500]
            if len(chunk.text) > 500:
                excerpt += "..."
            summarized_chunks.append(
                {
                    "document_name": chunk.document_name,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "text": excerpt,
                }
            )
        return {
            "status": result.status,
            "chunks": summarized_chunks,
        }
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
9. Предпочитай заметки вида "цель работ", "контроль", "журналы работ", "параметры раствора".
10. Не придумывай document_name, chunk_index и page_number: бери только из результатов поиска.
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
) -> Tool:
    def handler(args: IndexSearchInput) -> SearchSynthesisResult:
        search_result = index.search_hybrid(
            query=args.query,
            top_k_dense=top_k_dense,
            top_k_bm25=top_k_bm25,
        )
        if not search_result:
            if logger is not None:
                logger.info(
                    "INDEX SEARCH\nQuery: %s\nHits: no relevant documents found",
                    args.query,
                )
            return SearchSynthesisResult(
                status="not_found",
                answer="Релевантные документы по этому подзапросу не найдены.",
                useful_chunks=[],
            )

        if logger is not None:
            logger.info(
                "INDEX SEARCH\nQuery: %s\nHits:\n%s",
                args.query,
                _format_search_hits_for_log(search_result),
            )

        formatted_results = _format_search_results(search_result)
        synthesis = llm.parse(
            _build_search_synthesis_prompt(args.query, formatted_results),
            SearchSynthesisResult,
        )
        available_chunks = {
            (Path(result.chunk["source_path"]).name, int(result.chunk["index"])): {
                "page_number": result.chunk["page_number"],
            }
            for result in search_result
        }
        return SearchSynthesisResult(
            status=synthesis.status,
            answer=synthesis.answer,
            useful_chunks=_normalize_chunk_refs(
                synthesis.useful_chunks,
                available_chunks,
            ),
        )

    return Tool(
        name="IndexSearch",
        description=INDEX_SEARCH_TOOL_DESCRIPTION,
        input_model=IndexSearchInput,
        handler=handler,
    )


def build_index_chunk_lookup_tool(index: Index) -> Tool:
    def handler(args: IndexChunkLookupInput) -> ChunkLookupResult:
        source_path = index.resolve_source_path(args.document_name)
        if source_path is None:
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


def build_v4_tools(
    index: Index,
    llm: BaseLLM,
    max_top_k_dense: int = 3,
    max_top_k_bm25: int = 3,
) -> List[Tool]:
    return [
        build_index_search_tool(
            index,
            llm,
            logger=index.logger,
            top_k_dense=max_top_k_dense,
            top_k_bm25=max_top_k_bm25,
        ),
        build_index_chunk_lookup_tool(index),
    ]


def build_default_index(
    logger: logging.Logger,
    bm25_index_name: str = "test_index",
    os_host: str = "localhost",
    os_port: int = 9201,
    data_store_dir: Path = DEFAULT_DATA_STORE_DIR,
    chunk_store_dir: Path = DEFAULT_CHUNK_STORE_DIR,
    vector_store_dir: Path = DEFAULT_VECTOR_STORE_DIR,
    sqlite_path: Path = DEFAULT_SQLITE_PATH,
) -> Index:
    embedder = HTTPEmbedder()
    reader = HTTPReader()
    chunker = ParagraphChunker(
        logger=logger,
        max_length=490,
        overlap_sentences=1,
    )
    dimensions = embedder.get_embeddings(["test"]).shape[-1]
    os_client = OpenSearch(hosts=[{"host": os_host, "port": os_port}])
    bm25store = OpenSearchBM25Store(
        client=os_client,
        index_name=bm25_index_name,
        logger=logger,
    )
    return Index(
        datastore=DataStore(str(data_store_dir), logger),
        vectorstore=FlatVectorStore(str(vector_store_dir), dimensions, logger),
        chunkstore=ChunkStore(str(chunk_store_dir), logger),
        chunker=chunker,
        embedder=embedder,
        reader=reader,
        sqlite_path=str(sqlite_path),
        bm25store=bm25store,
        logger=logger,
    )


class V4Agent(BaseAgent):
    def __init__(
        self,
        llm: BaseLLM,
        tools: List[Tool],
        max_iters: int = 20,
        history_limit: int = 50,
        logger: Optional[logging.Logger] = None,
        system_prompt: str = DEFAULT_V4_SYSTEM_PROMPT,
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

    def answer(self, question: str, source_paths: Optional[List[str]] = None) -> str:
        del source_paths
        self._reset_context()
        self._user_query = question

        for iteration in range(self.max_iters):
            self.logger.info("Iter %s", iteration)
            reasoning = self._reasoning()
            if reasoning.task_completed:
                answer = reasoning.final_answer or "No info"
                self._reset_context()
                return answer

            if reasoning.next_steps:
                self._action(reasoning.next_steps[0])

        self._reset_context()
        return "No info"

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
            "tools": [
                {"name": tool.name, "description": tool.description}
                for tool in self.tools
            ],
            "user_query": self._user_query,
            "actions_history": _serialize_for_context(self._history),
        }
        return json.dumps(payload, ensure_ascii=False, default=str)

    def _reasoning(self) -> BaseModel:
        context = self._build_context()
        reasoning = self.llm.parse(context, self.schema)
        log_msg = (
            "🧠 REASONING \n"
            + "\n".join(reasoning.reasoning_steps)
            + "\nNext steps:\n"
            + "\n".join(
                [f"{i}. {step}" for i, step in enumerate(reasoning.next_steps)]
            )
        )
        self.logger.info(log_msg)
        return reasoning

    def _action(self, step: BaseModel) -> None:
        tool_name = step.tool_name
        tool = self.tools_map.get(tool_name)

        if tool is None:
            result = {"error": f"Unknown tool: {tool_name}"}
            success = False
        else:
            try:
                result = tool.handler(step.tool_args)
                success = True
            except Exception as exc:
                result = {"error": f"Tool error: {exc}"}
                self.logger.error(exc)
                success = False

        context_payload = StepResult(
            step=_serialize_for_context(step),
            tool_result=_summarize_tool_result(result),
            success=success,
        )
        self._push_history(context_payload)

        log_msg = (
            "\n 🛠️ ACTION \n"
            f"Tool: {tool_name}\n"
            f"Args: {step.tool_args}\n"
            f"Result: {_serialize_for_context(result)}\n"
        )
        self.logger.info(log_msg)

    def log_context(self) -> None:
        self.logger.info("\n === CONTEXT === \n%s", self._build_context())


def create_v4_agent(
    client: Any,
    model_name: str = "gpt-5.4",
    logger: Optional[logging.Logger] = None,
    system_prompt: str = DEFAULT_V4_SYSTEM_PROMPT,
    max_iters: int = 20,
    history_limit: int = 50,
    bm25_index_name: str = "test_index",
    max_top_k_dense: int = 3,
    max_top_k_bm25: int = 3,
) -> V4Agent:
    active_logger = logger or logging.getLogger(__name__)
    index = build_default_index(
        logger=active_logger,
        bm25_index_name=bm25_index_name,
    )
    llm = OpenAILLM(client, model_name=model_name)
    tools = build_v4_tools(
        index,
        llm,
        max_top_k_dense=max_top_k_dense,
        max_top_k_bm25=max_top_k_bm25,
    )
    return V4Agent(
        llm=llm,
        tools=tools,
        max_iters=max_iters,
        history_limit=history_limit,
        logger=active_logger,
        system_prompt=system_prompt,
    )


__all__ = [
    "ChunkLookupResult",
    "ChunkReference",
    "ChunkText",
    "DEFAULT_V4_SYSTEM_PROMPT",
    "INDEX_CHUNK_LOOKUP_TOOL_DESCRIPTION",
    "INDEX_SEARCH_TOOL_DESCRIPTION",
    "IndexChunkLookupInput",
    "IndexSearchInput",
    "SearchSynthesisResult",
    "Tool",
    "ToolInput",
    "V4Agent",
    "build_default_index",
    "build_index_chunk_lookup_tool",
    "build_index_search_tool",
    "build_v4_tools",
    "create_v4_agent",
    "make_plan_model",
]
