from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.tree_rag.tree import DocumentTree


@dataclass
class StoredDocument:
    idx: int
    tree: DocumentTree

    @property
    def title(self) -> str:
        return self.tree.source_name or self.tree.root.title

    @property
    def description(self) -> str:
        return self.tree.document_description or ""


class TreeStore:
    def __init__(self) -> None:
        self._documents: Dict[int, StoredDocument] = {}
        self._next_idx = 1
        self._source_to_idx: Dict[str, int] = {}

    def add_document(self, tree: DocumentTree) -> int:
        source_key = tree.source_name or tree.root.title
        if source_key in self._source_to_idx:
            return self._source_to_idx[source_key]

        idx = self._next_idx
        self._next_idx += 1
        self._documents[idx] = StoredDocument(idx=idx, tree=tree)
        self._source_to_idx[source_key] = idx
        return idx

    def remove_document(self, idx: int) -> DocumentTree:
        if idx not in self._documents:
            raise IndexError(f"Document idx out of range: {idx}")
        document = self._documents.pop(idx)
        source_key = document.tree.source_name or document.tree.root.title
        self._source_to_idx.pop(source_key, None)
        return document.tree

    def get_document(self, idx: int) -> DocumentTree:
        if idx not in self._documents:
            raise IndexError(f"Document idx out of range: {idx}")
        return self._documents[idx].tree

    def get_document_idx_by_source(self, source_name: str) -> Optional[int]:
        return self._source_to_idx.get(source_name)

    def max_document_idx(self) -> int:
        if not self._documents:
            return 0
        return max(self._documents)

    def llm_view(self, labels: Optional[Dict[int, List[str]]] = None) -> str:
        labels = labels or {}
        lines: List[str] = []
        for idx in sorted(self._documents):
            document = self._documents[idx]
            label_suffix = "".join(f"[{label}]" for label in labels.get(idx, []))
            description = document.description.strip()
            if description:
                lines.append(f"[{idx}]{label_suffix}: {document.title}: {description}")
            else:
                lines.append(f"[{idx}]{label_suffix}: {document.title}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "next_idx": self._next_idx,
            "documents": {
                idx: document.tree.to_flat_dict()
                for idx, document in self._documents.items()
            },
        }

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict) -> "TreeStore":
        store = cls()
        store._next_idx = data.get("next_idx", 1)
        documents = data.get("documents", {})
        for raw_idx, tree_data in documents.items():
            idx = int(raw_idx)
            tree = DocumentTree.from_flat_dict(tree_data)
            store._documents[idx] = StoredDocument(idx=idx, tree=tree)
            source_key = tree.source_name or tree.root.title
            store._source_to_idx[source_key] = idx
        if store._documents:
            store._next_idx = max(store._next_idx, max(store._documents) + 1)
        return store

    @classmethod
    def load(cls, path: str | Path) -> "TreeStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
