from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.indexing.vectorstore import FlatVectorStore, VetorStore
from src.preprocessing.chunker import BaseChunker
from src.preprocessing.embedder import HTTPEmbedder
from src.preprocessing.reader import Reader
from src.preprocessing.schema import Document, Page
from src.tree_rag.preprocessing import TreePreprocessor
from src.tree_rag.store import TreeStore
from src.tree_rag.tree import DocumentTree


class TreeIndex:
    def __init__(
        self,
        index_dir: str | Path,
        reader: Reader,
        preprocessor: TreePreprocessor,
        chunker: BaseChunker,
        embedder: HTTPEmbedder,
        logger: logging.Logger,
        tree_store: Optional[TreeStore] = None,
        vectorstore: Optional[VetorStore] = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.reader = reader
        self.preprocessor = preprocessor
        self.chunker = chunker
        self.embedder = embedder
        self.logger = logger

        self.sqlite_path = self.index_dir / "tree_index.sqlite"
        self.tree_store_path = self.index_dir / "tree_store.json"

        self.tree_store = tree_store or TreeStore()
        if vectorstore is None:
            dimensions = int(self.embedder.get_embeddings(["test"]).shape[-1])
            self.vectorstore = FlatVectorStore(
                str(self.index_dir / "vectors"),
                dimensions,
                logger,
            )
        else:
            self.vectorstore = vectorstore

        self.conn = sqlite3.connect(str(self.sqlite_path))
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        self.load()

    def _init_db(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS points (
                    document_idx INTEGER NOT NULL,
                    leaf_idx INTEGER NOT NULL,
                    vector_id INTEGER PRIMARY KEY
                )
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_points_document_idx ON points(document_idx)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_points_leaf_idx ON points(leaf_idx)"
            )

    def add_source(self, source_path: str) -> int:
        document = self.reader.read(source_path)
        if document is None:
            raise ValueError(f"Failed to read source: {source_path}")
        tree = self.preprocessor.preprocess_document(document)
        return self.add_tree(tree)

    def add_sources(self, source_paths: List[str]) -> List[int]:
        document_indices: List[int] = []
        for source_path in source_paths:
            document_indices.append(self.add_source(source_path))
        return document_indices

    def add_tree(self, tree: DocumentTree) -> int:
        document_idx = self.tree_store.add_document(tree)
        if self._document_has_points(document_idx):
            return document_idx

        leaf_nodes = tree.get_leaf_nodes()
        chunk_texts: List[str] = []
        chunk_points: List[tuple[int, int]] = []

        for leaf_idx, node in enumerate(leaf_nodes, start=1):
            if not node.text:
                continue
            leaf_path = tree.get_leaf_path_by_idx(leaf_idx)
            leaf_document = Document(
                source_path=tree.source_name or tree.root.title,
                pages=[Page(number=(node.pages or [1])[0], text=node.text)],
                doc_id=str(document_idx),
            )
            chunks = self.chunker.split_document_to_chunk(leaf_document)
            texts = [chunk.text for chunk in chunks if chunk.text.strip()]
            if not texts:
                texts = [node.text]
            for text in texts:
                chunk_texts.append(self._build_embedding_text(tree, leaf_path, node.title, text))
                chunk_points.append((document_idx, leaf_idx))

        if not chunk_texts:
            return document_idx

        vectors = self.embedder.get_embeddings(chunk_texts, text_type="document")
        vectors = np.asarray(vectors)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        start_idx = len(self.vectorstore.get_list_idx())
        if not self.vectorstore.add(vectors):
            raise ValueError(f"Failed to add vectors for {tree.source_name or tree.root.title}")

        with self.conn:
            for offset, (doc_idx, leaf_idx) in enumerate(chunk_points):
                vector_id = start_idx + offset
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO points (document_idx, leaf_idx, vector_id)
                    VALUES (?, ?, ?)
                    """,
                    (doc_idx, leaf_idx, vector_id),
                )
        return document_idx

    def remove_document(self, document_idx: int) -> None:
        points = self.conn.execute(
            """
            SELECT vector_id
            FROM points
            WHERE document_idx = ?
            ORDER BY vector_id
            """,
            (document_idx,),
        ).fetchall()
        vector_ids = [int(row["vector_id"]) for row in points]

        self.tree_store.remove_document(document_idx)

        with self.conn:
            self.conn.execute(
                "DELETE FROM points WHERE document_idx = ?",
                (document_idx,),
            )

        if not vector_ids:
            return

        old2new = self.vectorstore.delete(vector_ids)
        if not old2new:
            return

        with self.conn:
            for old in sorted(old2new.keys(), reverse=True):
                self.conn.execute(
                    "UPDATE points SET vector_id = ? WHERE vector_id = ?",
                    (-(old + 1), old),
                )
            for old, new in old2new.items():
                self.conn.execute(
                    "UPDATE points SET vector_id = ? WHERE vector_id = ?",
                    (new, -(old + 1)),
                )

    def search(
        self,
        query: str,
        top_k: int = 10,
        document_indices: Optional[List[int]] = None,
    ) -> Dict[int, List[int]]:
        query_vector = self.embedder.get_embeddings([query], text_type="query")
        query_vector = np.asarray(query_vector)[0]
        allowed_vector_ids = (
            self._get_vector_ids_by_document_indices(document_indices)
            if document_indices
            else None
        )
        vector_ids, scores = self.vectorstore.search(
            query_vector,
            top_k=top_k,
            allowed_idx=allowed_vector_ids,
        )

        grouped: Dict[int, Dict[int, float]] = {}
        for vector_id, score in zip(vector_ids, scores):
            row = self.conn.execute(
                "SELECT document_idx, leaf_idx FROM points WHERE vector_id = ?",
                (vector_id,),
            ).fetchone()
            if not row:
                continue
            document_idx = int(row["document_idx"])
            leaf_idx = int(row["leaf_idx"])
            grouped.setdefault(document_idx, {})
            prev_score = grouped[document_idx].get(leaf_idx)
            if prev_score is None or float(score) > prev_score:
                grouped[document_idx][leaf_idx] = float(score)

        ordered_documents = sorted(
            grouped.items(),
            key=lambda item: max(item[1].values()) if item[1] else float("-inf"),
            reverse=True,
        )
        results: Dict[int, List[int]] = {}
        for document_idx, leaf_scores in ordered_documents:
            ordered_leafs = sorted(
                leaf_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            results[document_idx] = [leaf_idx for leaf_idx, _ in ordered_leafs]
        return results

    def llm_view(self, labels: Optional[Dict[int, List[str]]] = None) -> str:
        return self.tree_store.llm_view(labels=labels)

    def llm_view_document(
        self,
        document_idx: int,
        labels: Optional[Dict[int, List[str]]] = None,
    ) -> str:
        return self.tree_store.get_document(document_idx).llm_view(labels=labels)

    def get_text(self, document_idx: int, leaf_idx: int) -> str:
        return self.tree_store.get_document(document_idx).get_text_by_idx(leaf_idx)

    def get_document(self, document_idx: int) -> DocumentTree:
        return self.tree_store.get_document(document_idx)

    def max_document_idx(self) -> int:
        return self.tree_store.max_document_idx()

    def save(self) -> None:
        self.tree_store.save(self.tree_store_path)
        self.vectorstore.save()

    def load(self) -> None:
        if self.tree_store_path.exists():
            self.tree_store = TreeStore.load(self.tree_store_path)
        self.vectorstore.load()

    def clear(self) -> None:
        self.tree_store = TreeStore()
        self.vectorstore.clear()
        with self.conn:
            self.conn.execute("DELETE FROM points")
        if self.tree_store_path.exists():
            os.remove(self.tree_store_path)

    def _document_has_points(self, document_idx: int) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM points WHERE document_idx = ? LIMIT 1",
            (document_idx,),
        ).fetchone()
        return row is not None

    def _get_vector_ids_by_document_indices(
        self,
        document_indices: List[int],
    ) -> List[int]:
        placeholders = ",".join(["?"] * len(document_indices))
        rows = self.conn.execute(
            f"""
            SELECT vector_id
            FROM points
            WHERE document_idx IN ({placeholders})
            ORDER BY vector_id
            """,
            document_indices,
        ).fetchall()
        return [int(row["vector_id"]) for row in rows]

    @staticmethod
    def _build_embedding_text(
        tree: DocumentTree,
        leaf_path: str,
        leaf_title: str,
        chunk_text: str,
    ) -> str:
        source_name = tree.source_name or tree.root.title
        description = tree.document_description or ""
        header_parts = [
            f"Document: {source_name}",
            f"Path: {leaf_path}",
            f"Title: {leaf_title}",
        ]
        if description:
            header_parts.append(f"Description: {description}")
        header = "\n".join(header_parts)
        return f"{header}\nContent:\n{chunk_text}"
