from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


DEFAULT_BM25_MAPPING = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }
    },
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "source_path": {"type": "keyword"},
            "page_number": {"type": "integer"},
            "index": {"type": "integer"},
            "text": {"type": "text"},
            "extra": {"type": "text"},
        }
    },
}


class BaseBM25Store(ABC):
    @abstractmethod
    def ensure_index(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_by_doc_ids(self, doc_ids: List[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_by_chunk_ids(self, chunk_ids: List[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int,
        doc_ids: Optional[List[str]] = None,
        source_paths: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def info(self) -> Dict[str, Any]:
        raise NotImplementedError


class OpenSearchBM25Store(BaseBM25Store):
    def __init__(
        self,
        client: Any,
        index_name: str,
        logger: logging.Logger,
        mapping: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.client = client
        self.index_name = index_name
        self.logger = logger
        self.mapping = mapping or DEFAULT_BM25_MAPPING
        self.ensure_index()

    def ensure_index(self) -> None:
        if self.client.indices.exists(index=self.index_name):
            return
        self.client.indices.create(index=self.index_name, body=self.mapping)
        self.logger.info("OpenSearch index created: %s", self.index_name)

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return
        self.ensure_index()
        actions: List[Dict[str, Any]] = []
        for chunk in chunks:
            actions.append({"index": {"_index": self.index_name, "_id": chunk["chunk_id"]}})
            actions.append(chunk)
        self.client.bulk(body=actions, refresh=True)

    def delete_by_doc_ids(self, doc_ids: List[str]) -> None:
        if not doc_ids:
            return
        self.client.delete_by_query(
            index=self.index_name,
            body={"query": {"terms": {"doc_id": doc_ids}}},
            refresh=True,
            conflicts="proceed",
        )

    def delete_by_chunk_ids(self, chunk_ids: List[str]) -> None:
        if not chunk_ids:
            return
        self.client.delete_by_query(
            index=self.index_name,
            body={"query": {"terms": {"chunk_id": chunk_ids}}},
            refresh=True,
            conflicts="proceed",
        )

    def clear(self) -> None:
        if not self.client.indices.exists(index=self.index_name):
            return
        self.client.indices.delete(index=self.index_name)
        self.logger.info("OpenSearch index deleted: %s", self.index_name)

    def search(
        self,
        query: str,
        top_k: int,
        doc_ids: Optional[List[str]] = None,
        source_paths: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        self.ensure_index()
        filters: List[Dict[str, Any]] = []
        if doc_ids:
            filters.append({"terms": {"doc_id": doc_ids}})
        if source_paths:
            filters.append({"terms": {"source_path": source_paths}})

        body: Dict[str, Any] = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [{"match": {"text": query}}],
                }
            },
        }
        if filters:
            body["query"]["bool"]["filter"] = filters

        response = self.client.search(index=self.index_name, body=body)
        hits = response.get("hits", {}).get("hits", [])
        results: List[Dict[str, Any]] = []
        for hit in hits:
            source = hit.get("_source", {})
            results.append(
                {
                    "chunk_id": source.get("chunk_id"),
                    "doc_id": source.get("doc_id"),
                    "score": float(hit.get("_score", 0.0)),
                    "chunk": source,
                }
            )
        return results

    def info(self) -> Dict[str, Any]:
        exists = self.client.indices.exists(index=self.index_name)
        if not exists:
            return {
                "backend": "opensearch",
                "index_name": self.index_name,
                "exists": False,
                "documents": 0,
            }

        count_response = self.client.count(index=self.index_name)
        return {
            "backend": "opensearch",
            "index_name": self.index_name,
            "exists": True,
            "documents": int(count_response.get("count", 0)),
        }
