from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from .tree import DocumentTree, TreeNode


HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
BOLD_LINE_RE = re.compile(r"^\*\*(.+?)\*\*\s*$")
PAGE_RE_LIST = [
    re.compile(r"\[page\s+(\d+)\]", re.IGNORECASE),
    re.compile(r"\[p\.\s*(\d+)\]", re.IGNORECASE),
    re.compile(r"\[стр\.\s*(\d+)\]", re.IGNORECASE),
]
TABLE_SEPARATOR_RE = re.compile(r"^\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?$")
MATH_MARKERS = ("$$", r"\(", r"\)", r"\[", r"\]", r"\frac", r"\sum", r"\int")
MATH_SYMBOL_RE = re.compile(r"[=+\-*/^<>≤≥±∑∫√]")


@dataclass
class _Section:
    title: str
    level: int
    body_lines: List[str] = field(default_factory=list)
    children: List["_Section"] = field(default_factory=list)


def parse_md_file(path: str | Path) -> DocumentTree:
    md_path = Path(path)
    text = md_path.read_text(encoding="utf-8")
    return parse_md_text(text=text, document_title=md_path.stem)


def parse_md_text(text: str, document_title: str) -> DocumentTree:
    lines = text.splitlines()
    root_section = _build_section_tree(lines, document_title=document_title)
    builder = _TreeBuilder()
    root_id = builder.build_section(root_section)
    return DocumentTree(root_id=root_id, nodes=builder.nodes)


def _build_section_tree(lines: List[str], document_title: str) -> _Section:
    root = _Section(title=document_title, level=0)
    stack = [root]
    in_code_block = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            stack[-1].body_lines.append(line)
            continue

        if not in_code_block:
            match = HEADER_RE.match(stripped)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                while stack and stack[-1].level >= level:
                    stack.pop()

                section = _Section(title=title, level=level)
                stack[-1].children.append(section)
                stack.append(section)
                continue

        stack[-1].body_lines.append(line)

    return root


class _TreeBuilder:
    def __init__(self) -> None:
        self._next_id = 1
        self.nodes: dict[int, TreeNode] = {}

    def build_section(self, section: _Section) -> int:
        node_id = self._new_node_id()
        child_ids: List[int] = []

        content_parts = _split_bold_subsections(section.body_lines)
        for part in content_parts:
            if isinstance(part, _BoldSubsection):
                child_ids.append(self._build_bold_subsection(part))
            else:
                child_ids.extend(self._build_leaf_blocks(part))

        for child in section.children:
            child_ids.append(self.build_section(child))

        pages = _extract_pages(section.body_lines)
        if child_ids:
            self.nodes[node_id] = TreeNode(
                title=section.title,
                node_id=node_id,
                pages=pages,
                nodes=child_ids,
            )
        else:
            leaf_text = "\n".join(section.body_lines).strip() or section.title
            self.nodes[node_id] = TreeNode(
                title=section.title,
                node_id=node_id,
                pages=pages,
                text=leaf_text,
                text_type="text",
            )
        return node_id

    def _build_bold_subsection(self, subsection: "_BoldSubsection") -> int:
        node_id = self._new_node_id()
        child_ids = self._build_leaf_blocks(subsection.body_lines)
        pages = _extract_pages(subsection.body_lines)
        if child_ids:
            self.nodes[node_id] = TreeNode(
                title=subsection.title,
                node_id=node_id,
                pages=pages,
                nodes=child_ids,
            )
        else:
            self.nodes[node_id] = TreeNode(
                title=subsection.title,
                node_id=node_id,
                pages=pages,
                text=subsection.title,
                text_type="text",
            )
        return node_id

    def _build_leaf_blocks(self, lines: List[str]) -> List[int]:
        blocks = _split_content_blocks(lines)
        counters = {"text": 0, "table": 0, "formula": 0}
        node_ids: List[int] = []

        for block in blocks:
            counters[block.block_type] += 1
            node_id = self._new_node_id()
            self.nodes[node_id] = TreeNode(
                title=f"{block.block_type}_{counters[block.block_type]}",
                node_id=node_id,
                pages=_extract_pages(block.lines),
                text=block.text,
                text_type=block.block_type,
            )
            node_ids.append(node_id)

        return node_ids

    def _new_node_id(self) -> int:
        node_id = self._next_id
        self._next_id += 1
        return node_id


@dataclass
class _BoldSubsection:
    title: str
    body_lines: List[str]


def _split_bold_subsections(lines: List[str]) -> List[List[str] | _BoldSubsection]:
    result: List[List[str] | _BoldSubsection] = []
    regular_buffer: List[str] = []
    i = 0

    while i < len(lines):
        heading = _match_bold_subsection(lines, i)
        if heading is None:
            regular_buffer.append(lines[i])
            i += 1
            continue

        if regular_buffer:
            result.append(regular_buffer)
            regular_buffer = []

        title, end_index = heading
        body_lines = lines[i + 1 : end_index]
        result.append(_BoldSubsection(title=title, body_lines=body_lines))
        i = end_index

    if regular_buffer:
        result.append(regular_buffer)

    return result


def _match_bold_subsection(lines: List[str], start_index: int) -> Optional[tuple[str, int]]:
    line = lines[start_index].strip()
    match = BOLD_LINE_RE.match(line)
    if not match:
        return None

    prev_index = _previous_non_empty_index(lines, start_index)
    if prev_index is not None and BOLD_LINE_RE.match(lines[prev_index].strip()):
        return None

    next_index = _next_non_empty_index(lines, start_index + 1)
    if next_index is None or BOLD_LINE_RE.match(lines[next_index].strip()):
        return None

    next_heading_index = len(lines)
    scan_index = next_index
    found_content = False

    while scan_index < len(lines):
        candidate = lines[scan_index].strip()
        if BOLD_LINE_RE.match(candidate):
            next_heading_index = scan_index
            break
        if candidate:
            found_content = True
        scan_index += 1

    if not found_content:
        return None

    return match.group(1).strip(), next_heading_index


@dataclass
class _Block:
    block_type: str
    lines: List[str]

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


def _split_content_blocks(lines: List[str]) -> List[_Block]:
    blocks: List[_Block] = []
    current_type: Optional[str] = None
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_type, current_lines
        text = "\n".join(current_lines).strip()
        if current_type is not None and text:
            blocks.append(_Block(block_type=current_type, lines=current_lines[:]))
        current_type = None
        current_lines = []

    for line in lines:
        if not line.strip():
            if current_lines:
                current_lines.append(line)
            continue

        block_type = _detect_block_type(line)
        if current_type is None:
            current_type = block_type
            current_lines = [line]
            continue

        if block_type == current_type:
            current_lines.append(line)
            continue

        flush()
        current_type = block_type
        current_lines = [line]

    flush()
    return blocks


def _detect_block_type(line: str) -> str:
    if _is_table_line(line):
        return "table"
    if _is_formula_line(line):
        return "formula"
    return "text"


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    if TABLE_SEPARATOR_RE.match(stripped):
        return True
    if stripped.count("|") >= 2:
        return True
    return False


def _is_formula_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if any(marker in stripped for marker in MATH_MARKERS):
        return True
    if MATH_SYMBOL_RE.search(stripped) and _looks_formula_like(stripped):
        return True
    return False


def _looks_formula_like(text: str) -> bool:
    letters = sum(char.isalpha() for char in text)
    math_symbols = len(MATH_SYMBOL_RE.findall(text))
    return math_symbols >= 2 and letters <= max(40, len(text) // 2)


def _extract_pages(lines: Iterable[str]) -> Optional[List[int]]:
    pages = set()
    for line in lines:
        for pattern in PAGE_RE_LIST:
            for match in pattern.findall(line):
                pages.add(int(match))
    return sorted(pages) if pages else None


def _previous_non_empty_index(lines: List[str], start_index: int) -> Optional[int]:
    index = start_index - 1
    while index >= 0:
        if lines[index].strip():
            return index
        index -= 1
    return None


def _next_non_empty_index(lines: List[str], start_index: int) -> Optional[int]:
    index = start_index
    while index < len(lines):
        if lines[index].strip():
            return index
        index += 1
    return None
