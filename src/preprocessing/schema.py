from dataclasses import dataclass
from typing import List

@dataclass
class Page:
    number: int
    text: str


@dataclass
class Document:
    path: str
    pages: List[Page]