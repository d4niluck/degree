from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field
from tqdm import tqdm

from src.tree_rag.tree import DocumentTree, TreeNode


class TOCEntry(BaseModel):
    level: int = Field(ge=1, le=6)
    title: str
    page_number: int = Field(ge=1)


class TOCExtraction(BaseModel):
    has_toc: bool
    toc_pages: list[int] = Field(default_factory=list)
    entries: list[TOCEntry] = Field(default_factory=list)


class TOCPageDetection(BaseModel):
    has_toc: bool
    entries: list[TOCEntry] = Field(default_factory=list)


class PageMatch(BaseModel):
    matched: bool
    page_number: Optional[int] = None


class SplitPart(BaseModel):
    title: str
    anchor_start: str


class SplitDecision(BaseModel):
    can_split: bool
    parts: list[SplitPart] = Field(default_factory=list)


class ChunkTitle(BaseModel):
    title: str


class DocumentDescription(BaseModel):
    description: str


@dataclass
class SectionSpan:
    level: int
    title: str
    start_page: int
    end_page: int


class TreePreprocessor:
    def __init__(
        self,
        llm: Any,
        logger: Optional[logging.Logger] = None,
        max_text_chars: int = 5000,
        min_text_chars: int = 500,
        max_llm_split_chars: int = 15000,
        max_split_depth: int = 10,
        max_title_chars: int = 80,
        max_title_words: int = 10,
    ):
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self.max_text_chars = max_text_chars
        self.min_text_chars = min_text_chars
        self.max_llm_split_chars = max_llm_split_chars
        self.max_split_depth = max_split_depth
        self.max_title_chars = max_title_chars
        self.max_title_words = max_title_words

    def normalize_text(self, text: str) -> str:
        text = text.lower().replace("ё", "е")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip()

    def pages_to_text(
        self,
        doc: Any,
        start_page: int,
        end_page: int,
        with_tags: bool = True,
        skip_pages: Optional[set[int]] = None,
    ) -> str:
        skip_pages = skip_pages or set()
        parts = []
        for page in doc.pages[start_page - 1 : end_page]:
            if page.number in skip_pages:
                continue
            if with_tags:
                parts.append(f"<page_{page.number}>\n{page.text}\n</page_{page.number}>")
            else:
                parts.append(page.text)
        return "\n\n".join(parts)

    def detect_toc_on_page(self, page_number: int, page_text: str) -> TOCPageDetection:
        prompt = f"""
Тебе дана одна страница PDF-документа.
Нужно понять, есть ли на ней фрагмент оглавления.

Если на странице есть оглавление, извлеки все видимые пункты.

Правила:
- Оглавление - это последовательный список разделов документа с номерами страниц.
- Приложения тоже являются частью оглавления.
- Любая строка вида "Приложение А ...", "Приложение Б ..." и так далее должна быть извлечена как отдельный пункт.
- Все приложения размечай как level=1.
- Строку "Библиография" тоже извлекай как отдельный пункт с level=1.
- Если title перенесён или склеен OCR, восстанови один цельный title.
- Если номер страницы прилип к тексту, всё равно выдели правильный page_number.
- Игнорируй колонтитулы, коды документа и мусор.
- Если на странице нет оглавления, верни has_toc=false и пустой entries.
- Не придумывай пункты, которых нет на странице.

Номер страницы документа: {page_number}

Текст страницы:
{page_text}
"""
        return self.llm.parse(prompt, TOCPageDetection, temperature=0)

    def merge_toc_entries(self, page_entries: list[list[TOCEntry]]) -> list[TOCEntry]:
        raw_lists = []
        for index, entries in enumerate(page_entries, start=1):
            raw_lists.append(
                {
                    "chunk_index": index,
                    "entries": [entry.model_dump() for entry in entries],
                }
            )

        prompt = f"""
Тебе дано несколько подряд идущих списков оглавления, извлечённых с соседних страниц документа.
Объедини их в один итоговый список.

Правила:
- Сохрани исходный порядок пунктов.
- Удали только явные дубликаты и OCR-мусор.
- Приложения являются частью оглавления и должны остаться в итоговом списке.
- Все приложения должны иметь level=1.
- Строку "Библиография" тоже нужно сохранить как пункт оглавления с level=1.
- Если приложение или библиография были разбиты OCR на части, склей их в один title.
- Не добавляй новые пункты, которых нет во входных списках.

Списки для объединения:
{json.dumps(raw_lists, ensure_ascii=False, indent=2)}
"""
        result = self.llm.parse(prompt, TOCPageDetection, temperature=0)
        return result.entries

    def recover_appendix_entries(self, toc_text: str) -> list[TOCEntry]:
        prompt = f"""
Тебе дан сырой текст оглавления документа.
Извлеки из него только приложения и библиографию.

Правила:
- Извлекай только строки, начинающиеся с "Приложение", и строку "Библиография".
- Приложения являются частью оглавления.
- Все приложения должны иметь level=1.
- Библиография тоже должна иметь level=1.
- Если строка перенесена или склеена OCR, восстанови один цельный title и page_number.
- Не возвращай основные разделы документа.
- Если приложений и библиографии нет, верни has_toc=false и пустой entries.

Текст оглавления:
{toc_text}
"""
        result = self.llm.parse(prompt, TOCPageDetection, temperature=0)
        return result.entries

    def extract_toc(self, doc: Any, max_pages: int = 20) -> TOCExtraction:
        toc_page_numbers: list[int] = []
        toc_chunks: list[list[TOCEntry]] = []
        toc_started = False

        for page in tqdm(doc.pages[:max_pages]):
            page_result = self.detect_toc_on_page(page.number, page.text)
            if page_result.has_toc and page_result.entries:
                toc_started = True
                toc_page_numbers.append(page.number)
                toc_chunks.append(page_result.entries)
                continue
            if toc_started:
                break

        if not toc_chunks:
            return TOCExtraction(has_toc=False, toc_pages=[], entries=[])

        merged_entries = self.merge_toc_entries(toc_chunks)
        toc_text = self.pages_to_text(
            doc,
            toc_page_numbers[0],
            toc_page_numbers[-1],
            with_tags=False,
        )
        has_appendix_in_text = "приложение" in toc_text.lower()
        has_appendix_in_entries = any(
            entry.title.lower().startswith("приложение") for entry in merged_entries
        )
        has_bibliography_in_text = "библиография" in toc_text.lower()
        has_bibliography_in_entries = any(
            entry.title.lower().startswith("библиография") for entry in merged_entries
        )

        if (has_appendix_in_text and not has_appendix_in_entries) or (
            has_bibliography_in_text and not has_bibliography_in_entries
        ):
            appendix_entries = self.recover_appendix_entries(toc_text)
            known_keys = {
                (entry.title.strip().lower(), entry.page_number) for entry in merged_entries
            }
            for entry in appendix_entries:
                key = (entry.title.strip().lower(), entry.page_number)
                if key not in known_keys:
                    merged_entries.append(entry)
                    known_keys.add(key)

        return TOCExtraction(
            has_toc=True,
            toc_pages=toc_page_numbers,
            entries=merged_entries,
        )

    def find_title_page_global(
        self,
        title: str,
        doc: Any,
        skip_pages: Optional[set[int]] = None,
        max_search_pages: int = 40,
    ) -> Optional[int]:
        skip_pages = skip_pages or set()
        normalized_title = self.normalize_text(title)
        if not normalized_title:
            return None

        for page in doc.pages[:max_search_pages]:
            if page.number in skip_pages:
                continue
            if normalized_title in self.normalize_text(page.text):
                return page.number
        return None

    def estimate_page_offset(
        self,
        entries: list[TOCEntry],
        doc: Any,
        toc_pages: list[int],
    ) -> int:
        skip_pages = set(toc_pages)
        offsets = []
        top_entries = [entry for entry in entries if entry.level == 1][:6]
        for entry in top_entries:
            physical_page = self.find_title_page_global(entry.title, doc, skip_pages=skip_pages)
            if physical_page is not None:
                offsets.append(physical_page - entry.page_number)
        if not offsets:
            return 0
        return Counter(offsets).most_common(1)[0][0]

    def find_section_start_page(
        self,
        title: str,
        doc: Any,
        expected_page: int,
        window: int = 4,
        skip_pages: Optional[set[int]] = None,
    ) -> Optional[int]:
        skip_pages = skip_pages or set()
        normalized_title = self.normalize_text(title)
        if not normalized_title:
            return None

        start = max(1, expected_page - window)
        end = min(len(doc.pages), expected_page + window)
        for page_number in range(start, end + 1):
            if page_number in skip_pages:
                continue
            page_text = self.normalize_text(doc.pages[page_number - 1].text)
            if normalized_title in page_text:
                return page_number

        local_pages = self.pages_to_text(doc, start, end, with_tags=True, skip_pages=skip_pages)
        prompt = f"""
Тебе дано название раздела и несколько страниц документа.
Найди страницу, на которой начинается этот раздел.

Правила:
- Используй нестрогое сопоставление.
- Игнорируй разрывы пробелов и OCR-шум.
- Если по этим страницам нельзя уверенно понять, где начинается раздел, верни matched=false.

Название раздела: {title}

Страницы:
{local_pages}
"""
        result = self.llm.parse(prompt, PageMatch, temperature=0)
        return result.page_number if result.matched else None

    def resolve_section_spans(
        self,
        entries: list[TOCEntry],
        doc: Any,
        toc_pages: list[int],
    ) -> list[SectionSpan]:
        resolved = []
        skip_pages = set(toc_pages)
        page_offset = self.estimate_page_offset(entries, doc, toc_pages)

        for entry in entries:
            adjusted_page = max(1, min(len(doc.pages), entry.page_number + page_offset))
            matched_page = self.find_section_start_page(
                entry.title,
                doc,
                adjusted_page,
                skip_pages=skip_pages,
            )
            start_page = matched_page or adjusted_page
            resolved.append(
                SectionSpan(
                    level=entry.level,
                    title=entry.title,
                    start_page=start_page,
                    end_page=start_page,
                )
            )

        for index, item in enumerate(resolved):
            next_start = None
            for candidate in resolved[index + 1 :]:
                if candidate.level <= item.level:
                    next_start = candidate.start_page
                    break
            item.end_page = (next_start - 1) if next_start else len(doc.pages)
            item.end_page = max(item.start_page, item.end_page)

        return resolved

    def find_anchor_position(self, text: str, anchor: str, start_pos: int = 0) -> Optional[int]:
        anchor = anchor.strip()
        if not anchor:
            return None
        exact_pos = text.find(anchor, start_pos)
        if exact_pos != -1:
            return exact_pos
        tokens = anchor.split()
        if not tokens:
            return None
        pattern = r"\s+".join(re.escape(token) for token in tokens)
        match = re.search(pattern, text[start_pos:], flags=re.IGNORECASE)
        if match:
            return start_pos + match.start()
        return None

    def split_text_by_pattern(self, text: str, pattern: re.Pattern[str]) -> list[str]:
        matches = list(pattern.finditer(text))
        if len(matches) < 2:
            return [text.strip()] if text.strip() else []
        positions = [match.start() for match in matches]
        segments = []
        for index, start in enumerate(positions):
            end = positions[index + 1] if index + 1 < len(positions) else len(text)
            segment = text[start:end].strip()
            if segment:
                segments.append(segment)
        return segments

    def split_by_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?;:])\s+", text)
        return [part.strip() for part in parts if part.strip()]

    def extract_units(self, text: str) -> list[str]:
        numbered_pattern = re.compile(r"(?=(?:^|(?<=\s))\d+(?:\.\d+)+\s+)")
        units = self.split_text_by_pattern(text, numbered_pattern)
        if len(units) > 1:
            return units
        blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
        if len(blocks) > 1:
            return blocks
        sentences = self.split_by_sentences(text)
        if sentences:
            return sentences
        return [text.strip()] if text.strip() else []

    def pack_units(self, units: list[str], max_chars: int) -> list[str]:
        chunks: list[str] = []
        buffer = ""
        for unit in units:
            unit = unit.strip()
            if not unit:
                continue
            if len(unit) > max_chars:
                if buffer:
                    chunks.append(buffer.strip())
                    buffer = ""
                sub_units = self.split_by_sentences(unit)
                if len(sub_units) <= 1:
                    sub_units = [unit[i : i + max_chars] for i in range(0, len(unit), max_chars)]
                chunks.extend(self.pack_units(sub_units, max_chars))
                continue
            candidate = unit if not buffer else f"{buffer} {unit}"
            if len(candidate) <= max_chars:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer.strip())
                buffer = unit
        if buffer:
            chunks.append(buffer.strip())
        if len(chunks) >= 2 and len(chunks[-1]) < self.min_text_chars:
            merged = f"{chunks[-2]} {chunks[-1]}".strip()
            if len(merged) <= max_chars:
                chunks[-2] = merged
                chunks.pop()
        return [chunk for chunk in chunks if chunk.strip()]

    def force_split_text(self, text: str, max_chars: int) -> list[str]:
        units = self.extract_units(text)
        chunks = self.pack_units(units, max_chars)
        if not chunks:
            if len(text) > max_chars:
                return [text[:max_chars].strip(), text[max_chars:].strip()]
            return [text.strip()]
        normalized_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        if len(text) > max_chars and (
            len(normalized_chunks) <= 1
            or (len(normalized_chunks) == 1 and normalized_chunks[0] == text.strip())
        ):
            return [
                text[i:i + max_chars].strip()
                for i in range(0, len(text), max_chars)
                if text[i:i + max_chars].strip()
            ]

        final_chunks: list[str] = []
        for chunk in normalized_chunks:
            if len(chunk) <= max_chars:
                final_chunks.append(chunk)
                continue
            final_chunks.extend(
                [
                    chunk[i:i + max_chars].strip()
                    for i in range(0, len(chunk), max_chars)
                    if chunk[i:i + max_chars].strip()
                ]
            )
        return final_chunks

    def plan_text_split(self, title: str, text: str) -> SplitDecision:
        prompt = f"""
Тебе дан фрагмент документа.
Нужно решить, стоит ли разбивать его на несколько последовательных частей для дерева документа.

Цель:
- получить крупные, связные и удобные для поиска фрагменты текста;
- не дробить текст слишком мелко;
- сохранить исходный текст без переписывания.

Правила:
- Если текст можно оставить цельным, верни can_split=false.
- Если делишь текст, разбивай его только на крупные смысловые блоки.
- Не создавай отдельную часть для каждого короткого нумерованного пункта.
- Несколько соседних пунктов нужно объединять, если они относятся к одной теме.
- Каждая часть должна быть непрерывным фрагментом исходного текста.
- Если в тексте есть хорошие названия частей, используй их.
- Если явных названий нет, придумай короткий осмысленный title самостоятельно.
- Не переписывай текст частей.
- Для каждой части верни anchor_start - точный фрагмент начала этой части, скопированный из исходного текста.
- Не делай части короче {self.min_text_chars} символов, если этого можно избежать.

Название текущего раздела: {title}

Исходный текст:
{text}
"""
        return self.llm.parse(prompt, SplitDecision, temperature=0)

    def sanitize_title(self, title: str) -> str:
        title = re.sub(r"\s+", " ", title)
        title = title.strip()
        title = title.strip(" -:;,.")
        return title

    def is_bad_title(self, title: str, text: str) -> bool:
        title = self.sanitize_title(title)
        if not title:
            return True
        if len(title) > self.max_title_chars:
            return True
        if len(title.split()) > self.max_title_words:
            return True
        if re.match(r"^(сп|гост)\b", title.lower()):
            return True
        if re.match(r"^\d+(?:\.\d+)+", title):
            return True
        if re.match(r"^\d{3,}\.\d+", title):
            return True
        normalized_title = self.normalize_text(title)
        normalized_start = self.normalize_text(text[: min(len(text), 300)])
        if normalized_title and normalized_title in normalized_start and len(title) > 20:
            return True
        return False

    def generate_chunk_title(self, parent_title: str, text: str, index: int) -> str:
        sample = text[: min(len(text), 3000)]
        prompt = f"""
Тебе дан фрагмент текста из раздела документа.
Дай для него короткий и понятный title на русском языке.

Правила:
- Верни только краткое название темы.
- Максимум 8 слов.
- Не копируй начало текста дословно.
- Не используй длинные номера пунктов и коды документа.
- Не пересказывай текст целиком.

Родительский раздел: {parent_title}

Текст:
{sample}
"""
        try:
            result = self.llm.parse(prompt, ChunkTitle, temperature=0)
            title = self.sanitize_title(result.title)
            if not self.is_bad_title(title, text):
                return title
        except Exception:
            pass
        return f"Часть {index}"

    def try_semantic_split(self, title: str, text: str) -> Optional[list[tuple[str, str]]]:
        if len(text) > self.max_llm_split_chars:
            return None
        decision = self.plan_text_split(title, text)
        if not decision.can_split or len(decision.parts) < 2:
            return None

        positions: list[tuple[int, str]] = []
        cursor = 0
        for part in decision.parts:
            pos = self.find_anchor_position(text, part.anchor_start, cursor)
            if pos is None:
                return None
            if positions and pos <= positions[-1][0]:
                return None
            positions.append((pos, self.sanitize_title(part.title)))
            cursor = pos + 1

        raw_segments: list[tuple[str, str]] = []
        for index, (start_pos, part_title) in enumerate(positions, start=1):
            end_pos = positions[index][0] if index < len(positions) else len(text)
            part_text = text[start_pos:end_pos].strip()
            if self.is_bad_title(part_title, part_text):
                part_title = self.generate_chunk_title(title, part_text, index)
            raw_segments.append((part_title, part_text))

        segments: list[tuple[str, str]] = []
        for part_title, part_text in raw_segments:
            if len(part_text) >= self.min_text_chars or not segments:
                segments.append((part_title, part_text))
                continue
            prev_title, prev_text = segments[-1]
            merged_text = f"{prev_text}\n\n{part_text}".strip()
            if len(merged_text) <= self.max_text_chars:
                segments[-1] = (prev_title, merged_text)
            else:
                segments.append((part_title, part_text))
        if len(segments) < 2:
            return None
        return segments

    def _make_text_leaf(
        self,
        text: str,
        pages: list[int],
        new_id: Any,
        nodes: dict[int, TreeNode],
    ) -> list[int]:
        leaf_id = new_id()
        nodes[leaf_id] = TreeNode(
            title="text",
            node_id=leaf_id,
            pages=pages,
            text=text,
            text_type="section_text",
        )
        return [leaf_id]

    def build_text_children(
        self,
        title: str,
        text: str,
        pages: list[int],
        new_id: Any,
        nodes: dict[int, TreeNode],
        depth: int = 0,
    ) -> list[int]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.max_text_chars:
            return self._make_text_leaf(text, pages, new_id, nodes)

        if depth >= self.max_split_depth:
            forced_chunks = self.force_split_text(text, self.max_text_chars)
            child_ids: list[int] = []
            for index, chunk_text in enumerate(forced_chunks, start=1):
                section_id = new_id()
                chunk_title = self.generate_chunk_title(title, chunk_text, index)
                nodes[section_id] = TreeNode(
                    title=chunk_title,
                    node_id=section_id,
                    pages=pages,
                    nodes=self.build_text_children(
                        chunk_title,
                        chunk_text,
                        pages,
                        new_id,
                        nodes,
                        depth + 1,
                    ),
                )
                child_ids.append(section_id)
            return child_ids

        if len(text) > self.max_llm_split_chars:
            coarse_chunks = self.force_split_text(text, self.max_llm_split_chars)
            child_ids: list[int] = []
            for index, chunk_text in enumerate(coarse_chunks, start=1):
                chunk_title = self.generate_chunk_title(title, chunk_text, index)
                section_id = new_id()
                nodes[section_id] = TreeNode(
                    title=chunk_title,
                    node_id=section_id,
                    pages=pages,
                    nodes=self.build_text_children(
                        chunk_title,
                        chunk_text,
                        pages,
                        new_id,
                        nodes,
                        depth + 1,
                    ),
                )
                child_ids.append(section_id)
            return child_ids

        semantic_parts = self.try_semantic_split(title, text)
        if semantic_parts:
            child_ids: list[int] = []
            for index, (part_title, part_text) in enumerate(semantic_parts, start=1):
                if self.is_bad_title(part_title, part_text):
                    part_title = self.generate_chunk_title(title, part_text, index)
                section_id = new_id()
                nodes[section_id] = TreeNode(
                    title=part_title,
                    node_id=section_id,
                    pages=pages,
                    nodes=self.build_text_children(
                        part_title,
                        part_text,
                        pages,
                        new_id,
                        nodes,
                        depth + 1,
                    ),
                )
                child_ids.append(section_id)
            return child_ids

        forced_chunks = self.force_split_text(text, self.max_text_chars)
        child_ids = []
        for index, chunk_text in enumerate(forced_chunks, start=1):
            chunk_title = self.generate_chunk_title(title, chunk_text, index)
            section_id = new_id()
            nodes[section_id] = TreeNode(
                title=chunk_title,
                node_id=section_id,
                pages=pages,
                nodes=self.build_text_children(
                    chunk_title,
                    chunk_text,
                    pages,
                    new_id,
                    nodes,
                    depth + 1,
                ),
            )
            child_ids.append(section_id)
        return child_ids

    def validate_tree_text_limits(self, tree: DocumentTree) -> None:
        oversized = [
            node
            for node in tree.nodes.values()
            if node.is_leaf and len(node.text or "") > self.max_text_chars
        ]
        if oversized:
            raise ValueError(
                f"Found oversized leaf nodes: {[node.node_id for node in oversized]}"
            )

    def generate_document_description(self, tree: DocumentTree, preview_depth: int = 3) -> str:
        root = tree.get_node(tree.root_id)
        preview = root.preview(depth=preview_depth)
        prompt = f"""
Тебе дано дерево структуры документа.
Нужно написать короткое описание источника, чтобы агент мог понять, стоит ли открывать этот файл для поиска информации.

Правила:
- Опиши, о чем документ и какие темы он покрывает.
- Пиши кратко, в 1-2 предложения.
- Не пересказывай структуру дерева целиком.
- Не используй маркированные списки.
- Не придумывай факты, которых нет в дереве.

Структура документа:
{preview}
"""
        result = self.llm.parse(prompt, DocumentDescription, temperature=0)
        return result.description.strip()

    def build_tree_from_spans(self, doc: Any, spans: list[SectionSpan]) -> DocumentTree:
        nodes: dict[int, TreeNode] = {}
        next_id = 1

        def new_id() -> int:
            nonlocal next_id
            node_id = next_id
            next_id += 1
            return node_id

        root_id = new_id()
        root = TreeNode(
            title=Path(doc.source_path).stem,
            node_id=root_id,
            pages=[1, len(doc.pages)],
            nodes=[],
        )
        nodes[root_id] = root

        stack: list[tuple[int, int]] = [(0, root_id)]
        for span in spans:
            while stack and stack[-1][0] >= span.level:
                stack.pop()
            parent_id = stack[-1][1]
            section_id = new_id()
            pages = list(range(span.start_page, span.end_page + 1))
            text = self.pages_to_text(doc, span.start_page, span.end_page, with_tags=False)
            nodes[section_id] = TreeNode(
                title=span.title,
                node_id=section_id,
                pages=pages,
                nodes=self.build_text_children(span.title, text, pages, new_id, nodes, depth=0),
            )
            parent = nodes[parent_id]
            if parent.nodes is None:
                parent.nodes = []
            parent.nodes.append(section_id)
            stack.append((span.level, section_id))

        tree = DocumentTree(
            root_id=root_id,
            nodes=nodes,
            source_name=Path(doc.source_path).name,
        )
        tree.collapse_single_leaf_children()
        self.validate_tree_text_limits(tree)
        tree.document_description = self.generate_document_description(tree)
        return tree

    def build_tree_without_toc(self, doc: Any) -> DocumentTree:
        nodes: dict[int, TreeNode] = {}
        next_id = 1

        def new_id() -> int:
            nonlocal next_id
            node_id = next_id
            next_id += 1
            return node_id

        root_id = new_id()
        root_title = Path(doc.source_path).stem
        root = TreeNode(
            title=root_title,
            node_id=root_id,
            pages=[1, len(doc.pages)],
            nodes=[],
        )
        nodes[root_id] = root

        full_text = self.pages_to_text(doc, 1, len(doc.pages), with_tags=False)
        root.nodes = self.build_text_children(
            root_title,
            full_text,
            list(range(1, len(doc.pages) + 1)),
            new_id,
            nodes,
            depth=0,
        )

        tree = DocumentTree(
            root_id=root_id,
            nodes=nodes,
            source_name=Path(doc.source_path).name,
        )
        tree.collapse_single_leaf_children()
        self.validate_tree_text_limits(tree)
        tree.document_description = self.generate_document_description(tree)
        return tree

    def preprocess_document(self, doc: Any, max_toc_pages: int = 20) -> DocumentTree:
        toc = self.extract_toc(doc, max_pages=max_toc_pages)
        if not toc.has_toc or not toc.entries:
            self.logger.warning(
                "TOC was not extracted for %s, fallback to full-document split",
                getattr(doc, "source_path", "<unknown>"),
            )
            return self.build_tree_without_toc(doc)
        spans = self.resolve_section_spans(toc.entries, doc, toc.toc_pages)
        return self.build_tree_from_spans(doc, spans)
