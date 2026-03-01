from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Page:
    number: int
    text: str


@dataclass
class Document:
    source_path: str
    pages: List[Page]
    doc_id: Optional[str] = None


@dataclass
class Chunk:
    chunk_id: str
    doc_id: Optional[str]
    source_path: str
    text: str
    index: int
    page_number: Optional[int]
    extra: Optional[str]