import logging
import os
import sqlite3
import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

import psutil
import torch

from tqdm import tqdm
import numpy as np

from ..preprocessing.chunker import BaseChunker
from ..preprocessing.schema import Document
from .bm25store import BaseBM25Store
from .chunkstore import ChunkStore
from .datastore import DataStore
from .vectorstore import VetorStore

if TYPE_CHECKING:
    from ..preprocessing.embedder import Embedder
    from ..preprocessing.reader import Reader


@dataclass
class SearchResult:
    doc_id: str
    chunk_id: str
    vector_id: int
    score: float
    chunk: Dict[str, Any]
    extra: Optional[str] = None


class Index:
    def __init__(
        self,
        datastore: DataStore,
        vectorstore: VetorStore,
        chunkstore: ChunkStore,
        chunker: BaseChunker,
        embedder: "Embedder",
        reader: "Reader",
        sqlite_path: str,
        logger: logging.Logger,
        bm25store: Optional[BaseBM25Store] = None,
    ) -> None:
        self.datastore = datastore
        self.vectorstore = vectorstore
        self.chunkstore = chunkstore
        self.chunker = chunker
        self.embedder = embedder
        self.reader = reader
        self.logger = logger
        self.sqlite_path = sqlite_path
        self.bm25store = bm25store
        self.conn = sqlite3.connect(self.sqlite_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        self._load_vectorstore()

    def _init_db(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS points (
                    doc_id TEXT NOT NULL,
                    chunk_id TEXT PRIMARY KEY,
                    vector_id INTEGER UNIQUE NOT NULL
                )
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_points_doc_id ON points(doc_id)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_points_vector_id ON points(vector_id)"
            )

    def add_sources(
        self,
        sources: List[str],
        save_vectorestore: bool = False,
    ) -> List[str]:
        indexed_source_paths = self._get_indexed_source_paths()
        new_sources = [source for source in sources if source not in indexed_source_paths]
        skipped_sources = len(sources) - len(new_sources)
        self.logger.info(
            f"add_sources: {len(new_sources)} new sources, {skipped_sources} skipped"
        )

        doc_ids: List[str] = []
        progress = tqdm(new_sources, desc="Indexing")
        for source in progress:
            rss_mb = self._get_rss_mb()
            accelerator_memory = self._get_accelerator_memory_label()
            progress.set_description(
                f"Indexing {os.path.basename(source)} | RSS {rss_mb:.0f} MB"
                f"{accelerator_memory}"
            )
            document = self.reader.read(source)
            if not document:
                self.logger.error(f"Failed to read source: {source}")
                self._clear_accelerator_cache()
                continue

            doc_id = self._index_document(document, add_to_datastore=True)
            if doc_id:
                doc_ids.append(doc_id)
        if save_vectorestore:
            self.save_vectorstore()
        return doc_ids

    def rebuild(
        self,
        save_vectorestore: bool = False,
    ) -> List[str]:
        doc_paths = self.datastore.get_list_doc_path()
        self.chunkstore.clear()
        self.vectorstore.clear()
        with self.conn:
            self.conn.execute("DELETE FROM points")

        doc_ids: List[str] = []
        progress = tqdm(doc_paths, desc="Rebuilding")
        for path in progress:
            document = self.datastore.read(path=path)
            if not document:
                self.logger.error(f"Failed to read document from datastore: {path}")
                continue
            rss_mb = self._get_rss_mb()
            accelerator_memory = self._get_accelerator_memory_label()
            progress.set_description(
                f"Rebuilding {os.path.basename(document.source_path)} | RSS {rss_mb:.0f} MB"
                f"{accelerator_memory}"
            )
            doc_id = self._index_document(document, add_to_datastore=False)
            if doc_id:
                doc_ids.append(doc_id)

        if save_vectorestore:
            self.save_vectorstore()
        return doc_ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[List[str]] = None,
        source_paths: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        return self.search_dense(
            query=query,
            top_k=top_k,
            doc_ids=doc_ids,
            source_paths=source_paths,
        )

    def search_dense(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[List[str]] = None,
        source_paths: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        allowed_doc_ids = self._resolve_doc_ids(doc_ids=doc_ids, source_paths=source_paths)
        allowed_vector_ids = self._get_vector_ids_by_doc_ids(allowed_doc_ids) if allowed_doc_ids else None

        query_vector = self.embedder.get_embeddings([query], text_type="query")
        query_vector = np.asarray(query_vector)[0]
        vector_ids, scores = self.vectorstore.search(
            query_vector,
            top_k=top_k,
            allowed_idx=allowed_vector_ids,
        )

        results: List[SearchResult] = []
        for vector_id, score in zip(vector_ids, scores):
            point = self.conn.execute(
                "SELECT doc_id, chunk_id FROM points WHERE vector_id = ?",
                (vector_id,),
            ).fetchone()
            if not point:
                continue
            chunk = self.chunkstore.read(chunk_id=point["chunk_id"])
            if not chunk:
                continue
            results.append(
                SearchResult(
                    doc_id=point["doc_id"],
                    chunk_id=point["chunk_id"],
                    vector_id=vector_id,
                    score=float(score),
                    chunk=chunk,
                )
            )
        return results

    def search_bm25(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[List[str]] = None,
        source_paths: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        if self.bm25store is None:
            return []

        allowed_doc_ids = self._resolve_doc_ids(doc_ids=doc_ids, source_paths=source_paths)
        raw_results = self.bm25store.search(
            query=query,
            top_k=top_k,
            doc_ids=allowed_doc_ids or None,
            source_paths=source_paths,
        )

        results: List[SearchResult] = []
        for item in raw_results:
            chunk = item.get("chunk") or self.chunkstore.read(chunk_id=item["chunk_id"])
            if not chunk:
                continue
            results.append(
                SearchResult(
                    doc_id=item["doc_id"],
                    chunk_id=item["chunk_id"],
                    vector_id=-1,
                    score=float(item["score"]),
                    chunk=chunk,
                    extra="bm25",
                )
            )
        return results

    def search_hybrid(
        self,
        query: str,
        top_k_dense: int = 5,
        top_k_bm25: int = 5,
        doc_ids: Optional[List[str]] = None,
        source_paths: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        
        if top_k_dense > 0:
            dense_results = self.search_dense(
                query=query,
                top_k=top_k_dense,
                doc_ids=doc_ids,
                source_paths=source_paths,
            )
        else:
            dense_results = []
            
        if top_k_bm25 > 0:
            bm25_results = self.search_bm25(
                query=query,
                top_k=top_k_bm25,
                doc_ids=doc_ids,
                source_paths=source_paths,
            )
        else:
            bm25_results = []
            
        results = dense_results + bm25_results
        if not results:
            return []

        merged: List[SearchResult] = []
        seen_chunk_ids: set[str] = set()
        for result in results:
            if result.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(result.chunk_id)
            merged.append(result)
        return merged

    def delete_documents(
        self,
        doc_ids: Optional[List[str]] = None,
        source_paths: Optional[List[str]] = None,
        save_vectorestore: bool = False,
    ) -> None:
        target_doc_ids = self._resolve_doc_ids(doc_ids=doc_ids, source_paths=source_paths)
        if not target_doc_ids:
            return

        points = self.conn.execute(
            f"""
            SELECT doc_id, chunk_id, vector_id
            FROM points
            WHERE doc_id IN ({",".join(["?"] * len(target_doc_ids))})
            ORDER BY vector_id
            """,
            target_doc_ids,
        ).fetchall()

        if not points:
            return

        vector_ids = [int(row["vector_id"]) for row in points]
        chunk_ids = [str(row["chunk_id"]) for row in points]

        for chunk_id in chunk_ids:
            self.chunkstore.delete(chunk_id=chunk_id)
        for doc_id in target_doc_ids:
            self.datastore.delete(doc_id=doc_id)
        if self.bm25store is not None:
            self.bm25store.delete_by_doc_ids(target_doc_ids)

        with self.conn:
            self.conn.executemany(
                "DELETE FROM points WHERE chunk_id = ?",
                [(chunk_id,) for chunk_id in chunk_ids],
            )

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
        if save_vectorestore:
            self.save_vectorstore()

    def delete_chunks(
        self,
        chunk_ids: List[str],
        save_vectorestore: bool = False,
    ) -> None:
        if not chunk_ids:
            return

        points = self.conn.execute(
            f"""
            SELECT chunk_id, vector_id
            FROM points
            WHERE chunk_id IN ({",".join(["?"] * len(chunk_ids))})
            ORDER BY vector_id
            """,
            chunk_ids,
        ).fetchall()
        if not points:
            return

        existing_chunk_ids = [str(row["chunk_id"]) for row in points]
        vector_ids = [int(row["vector_id"]) for row in points]

        for chunk_id in existing_chunk_ids:
            self.chunkstore.delete(chunk_id=chunk_id)
        if self.bm25store is not None:
            self.bm25store.delete_by_chunk_ids(existing_chunk_ids)

        with self.conn:
            self.conn.executemany(
                "DELETE FROM points WHERE chunk_id = ?",
                [(chunk_id,) for chunk_id in existing_chunk_ids],
            )

        old2new = self.vectorstore.delete(vector_ids)
        if old2new:
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
        if save_vectorestore:
            self.save_vectorstore()

    def clear(self) -> None:
        self.datastore.clear()
        self.chunkstore.clear()
        self.vectorstore.clear()
        if self.bm25store is not None:
            self.bm25store.clear()
        with self.conn:
            self.conn.execute("DELETE FROM points")
        self.logger.info("Index data removed")

    def sync_bm25_from_chunkstore(self) -> None:
        if self.bm25store is None:
            self.logger.warning("BM25 store is not configured")
            return

        doc_ids = self.list_documents()
        progress = tqdm(doc_ids, desc="Sync BM25")
        synced_chunks = 0
        for doc_id in progress:
            chunks = self.get_chunks(doc_id=doc_id)
            if not chunks:
                continue
            payloads = [self._chunk_dict_to_payload(chunk) for chunk in chunks]
            self.bm25store.add_chunks(payloads)
            synced_chunks += len(payloads)

        self.logger.info(
            "BM25 sync completed: %s documents, %s chunks",
            len(doc_ids),
            synced_chunks,
        )

    def save_vectorstore(self) -> None:
        self.vectorstore.save()
        self.logger.info("Index vectorstore saved")

    def _load_vectorstore(self) -> None:
        self.vectorstore.load()
        self.logger.info("Index vectorstore loaded")

    def get_document(
        self,
        doc_id: Optional[str] = None,
        source_path: Optional[str] = None,
        chunk_id: Optional[str] = None,
    ) -> Optional[Document]:
        self._validate_single_selector(
            doc_id=doc_id,
            source_path=source_path,
            chunk_id=chunk_id,
        )
        if doc_id:
            return self.datastore.read(doc_id=doc_id)
        if source_path:
            for path in self.datastore.get_list_doc_path():
                document = self.datastore.read(path=path)
                if document and document.source_path == source_path:
                    return document
            return None
        point = self.conn.execute(
            "SELECT doc_id FROM points WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if not point:
            return None
        return self.datastore.read(doc_id=point["doc_id"])

    def get_chunks(
        self,
        doc_id: Optional[str] = None,
        chunk_ids: Optional[List[str]] = None,
        source_path: Optional[str] = None,
        chunk_indices: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        selectors = [doc_id is not None, chunk_ids is not None, source_path is not None]
        if sum(selectors) != 1:
            raise ValueError("Exactly one of doc_id, chunk_ids, source_path must be set")
        if chunk_ids is not None and chunk_indices is not None:
            raise ValueError("chunk_indices cannot be used together with chunk_ids")

        resolved_chunk_ids: List[str]
        if doc_id is not None:
            resolved_chunk_ids = self._get_chunk_ids_for_doc_id(doc_id)
        elif chunk_ids is not None:
            resolved_chunk_ids = chunk_ids
        else:
            document = self.get_document(source_path=source_path)
            if not document or not document.doc_id:
                return []
            resolved_chunk_ids = self._get_chunk_ids_for_doc_id(document.doc_id)

        chunks: List[Dict[str, Any]] = []
        for chunk_id in resolved_chunk_ids:
            chunk = self.chunkstore.read(chunk_id=chunk_id)
            if chunk:
                chunks.append(chunk)

        if chunk_indices is None:
            return chunks

        chunk_by_index = {int(chunk["index"]): chunk for chunk in chunks}
        filtered_chunks: List[Dict[str, Any]] = []
        for chunk_index in chunk_indices:
            chunk = chunk_by_index.get(chunk_index)
            if chunk is not None:
                filtered_chunks.append(chunk)
        return filtered_chunks

    def list_documents(self) -> List[str]:
        return self.datastore.get_list_doc_id()

    def resolve_source_path(self, document_name: str) -> Optional[str]:
        for document in self._iter_documents_from_datastore():
            if document.source_path == document_name:
                return document.source_path

        matches = [
            document.source_path
            for document in self._iter_documents_from_datastore()
            if os.path.basename(document.source_path) == document_name
        ]
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(
                f"Multiple documents matched {document_name}: {matches}"
            )
        return matches[0]

    def info(self) -> Dict[str, Any]:
        doc_ids = self.datastore.get_list_doc_id()
        chunk_ids = self.chunkstore.get_list_chunk_id()
        vector_ids = self.vectorstore.get_list_idx()
        points_count = int(
            self.conn.execute("SELECT COUNT(*) AS n FROM points").fetchone()["n"]
        )

        point_chunk_rows = self.conn.execute("SELECT chunk_id FROM points").fetchall()
        point_chunk_ids = {str(row["chunk_id"]) for row in point_chunk_rows}
        chunk_ids_set = set(chunk_ids)

        stats = {
            "datastore": {"documents": len(doc_ids)},
            "chunkstore": {"chunks": len(chunk_ids)},
            "vectorstore": {
                "vectors": len(vector_ids),
                "dimensions": self.vectorstore.dimensions,
            },
            "index": {
                "points": points_count,
                "points_without_chunk": len(point_chunk_ids - chunk_ids_set),
                "chunks_without_point": len(chunk_ids_set - point_chunk_ids),
            },
        }
        if self.bm25store is not None:
            stats["bm25store"] = self.bm25store.info()
        self.logger.info(f"Index stats: {stats}")
        return stats

    def stats(self) -> Dict[str, Any]:
        return self.info()

    def list_chunks(
        self,
        doc_id: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> List[str]:
        if doc_id is not None and source_path is not None:
            raise ValueError("Use only one selector: doc_id or source_path")
        if doc_id is not None:
            rows = self.conn.execute(
                "SELECT chunk_id FROM points WHERE doc_id = ? ORDER BY vector_id",
                (doc_id,),
            ).fetchall()
            return [str(row["chunk_id"]) for row in rows]
        if source_path is not None:
            document = self.get_document(source_path=source_path)
            if not document or not document.doc_id:
                return []
            rows = self.conn.execute(
                "SELECT chunk_id FROM points WHERE doc_id = ? ORDER BY vector_id",
                (document.doc_id,),
            ).fetchall()
            return [str(row["chunk_id"]) for row in rows]
        return self.chunkstore.get_list_chunk_id()

    def close(self) -> None:
        self.save_vectorstore()
        self.conn.close()

    def _resolve_doc_ids(
        self,
        doc_ids: Optional[List[str]] = None,
        source_paths: Optional[List[str]] = None,
    ) -> List[str]:
        if doc_ids and source_paths:
            raise ValueError("Use doc_ids or source_paths, not both")
        if doc_ids:
            return doc_ids
        if source_paths:
            resolved: List[str] = []
            for source_path in source_paths:
                document = self.get_document(source_path=source_path)
                if document and document.doc_id:
                    resolved.append(document.doc_id)
            return resolved
        return []

    def _get_vector_ids_by_doc_ids(self, doc_ids: List[str]) -> List[int]:
        if not doc_ids:
            return []
        rows = self.conn.execute(
            f"""
            SELECT vector_id FROM points
            WHERE doc_id IN ({",".join(["?"] * len(doc_ids))})
            """,
            doc_ids,
        ).fetchall()
        return [int(row["vector_id"]) for row in rows]

    def _get_chunk_ids_for_doc_id(self, doc_id: str) -> List[str]:
        rows = self.conn.execute(
            "SELECT chunk_id FROM points WHERE doc_id = ? ORDER BY vector_id",
            (doc_id,),
        ).fetchall()
        return [str(row["chunk_id"]) for row in rows]

    def _get_indexed_source_paths(self) -> set[str]:
        return {
            document.source_path
            for document in self._iter_documents_from_datastore()
        }

    def _iter_documents_from_datastore(self) -> Iterator[Document]:
        for path in self.datastore.get_list_doc_path():
            document = self.datastore.read(path=path)
            if document:
                yield document

    @staticmethod
    def _chunk_to_payload(chunk: Any) -> Dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "source_path": chunk.source_path,
            "text": chunk.text,
            "index": chunk.index,
            "page_number": chunk.page_number,
            "extra": chunk.extra,
        }

    @staticmethod
    def _chunk_dict_to_payload(chunk: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "source_path": chunk["source_path"],
            "text": chunk["text"],
            "index": chunk["index"],
            "page_number": chunk["page_number"],
            "extra": chunk["extra"],
        }

    def _index_document(
        self,
        document: Document,
        add_to_datastore: bool,
    ) -> Optional[str]:
        chunks = None
        texts = None
        vectors = None
        try:
            if add_to_datastore:
                self.datastore.add(document)

            if not document.doc_id:
                self.logger.error(
                    f"Document doc_id is empty for source: {document.source_path}"
                )
                return None

            chunks = self.chunker.split_document_to_chunk(document)
            if not chunks:
                self.logger.info(
                    f"No chunks produced for source: {document.source_path}"
                )
                return document.doc_id

            for chunk in chunks:
                self.chunkstore.add(chunk)
            if self.bm25store is not None:
                self.bm25store.add_chunks(
                    [self._chunk_to_payload(chunk) for chunk in chunks]
                )

            texts = [chunk.text for chunk in chunks]
            vectors = self.embedder.get_embeddings(texts, text_type="document")
            vectors = np.asarray(vectors)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)

            start_idx = len(self.vectorstore.get_list_idx())
            if not self.vectorstore.add(vectors):
                self.logger.error(
                    f"Failed to upload vectors for source: {document.source_path}"
                )
                return None

            with self.conn:
                for offset, chunk in enumerate(chunks):
                    vector_id = start_idx + offset
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO points (doc_id, chunk_id, vector_id)
                        VALUES (?, ?, ?)
                        """,
                        (document.doc_id, chunk.chunk_id, vector_id),
                    )
            self.logger.debug(
                f"Indexed source={document.source_path}, doc_id={document.doc_id}, chunks={len(chunks)}"
            )
            return document.doc_id
        finally:
            if vectors is not None:
                del vectors
            if texts is not None:
                del texts
            if chunks is not None:
                del chunks
            self._clear_accelerator_cache()

    @staticmethod
    def _validate_single_selector(**kwargs: Optional[str]) -> None:
        if sum(value is not None for value in kwargs.values()) != 1:
            names = ", ".join(kwargs.keys())
            raise ValueError(f"Exactly one selector must be set: {names}")

    @staticmethod
    def _get_rss_mb() -> float:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def _clear_accelerator_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    @staticmethod
    def _get_accelerator_memory_label() -> str:
        if torch.cuda.is_available():
            cuda_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            return f" | CUDA {cuda_mb:.0f} MB"
        if torch.backends.mps.is_available():
            try:
                mps_current_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                mps_driver_mb = torch.mps.driver_allocated_memory() / (1024 * 1024)
                return f" | MPS {mps_current_mb:.0f}/{mps_driver_mb:.0f} MB"
            except Exception:
                return ""
        return ""
