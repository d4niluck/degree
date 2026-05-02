from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from opensearchpy import OpenSearch

from ...indexing import ChunkStore, DataStore, FlatVectorStore, Index, OpenSearchBM25Store
from ...preprocessing import FixedCharChunker, HTTPEmbedder, HTTPReader


class IndexManager:
    def __init__(
        self,
        storage_root: str | Path,
        logger: logging.Logger,
        embedder_url: str = "http://127.0.0.1:8001",
        reader_url: str = "http://127.0.0.1:8002",
        opensearch_host: str = "127.0.0.1",
        opensearch_port: int = 9201,
        opensearch_index_prefix: str = "kb",
    ) -> None:
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.embedder = HTTPEmbedder(base_url=embedder_url)
        self.reader = HTTPReader(base_url=reader_url)
        self.opensearch_host = opensearch_host
        self.opensearch_port = int(opensearch_port)
        self.opensearch_index_prefix = opensearch_index_prefix
        self.dimensions: int | None = None
        self._cache: Dict[str, Index] = {}

    def get_index(self, kb_id: str) -> Index:
        if kb_id in self._cache:
            return self._cache[kb_id]
        if self.dimensions is None:
            self.dimensions = int(self.embedder.get_embeddings(["test"]).shape[-1])

        root = self.storage_root / kb_id
        index = self.create_index(root)
        self._cache[kb_id] = index
        return index
    
    def create_index(self, root):
        bm25store = None
        try:
            os_client = OpenSearch(hosts=[{"host": self.opensearch_host, "port": self.opensearch_port}])
            bm25store = OpenSearchBM25Store(
                client=os_client,
                index_name=f"{self.opensearch_index_prefix}_{root.name}",
                logger=self.logger,
            )
        except Exception as exc:
            self.logger.warning("OpenSearch BM25 disabled for %s: %s", root.name, exc)

        index = Index(
            datastore=DataStore(str(root / "documents"), self.logger),
            vectorstore=FlatVectorStore(str(root / "vectors"), self.dimensions, self.logger),
            chunkstore=ChunkStore(str(root / "chunks"), self.logger),
            chunker=FixedCharChunker(logger=self.logger, chunk_size=500, overlap=50),
            embedder=self.embedder,
            reader=self.reader,
            sqlite_path=str(root / "index.db"),
            logger=self.logger,
            bm25store=bm25store,
        )
        if bm25store is not None:
            index.sync_bm25_from_chunkstore()
        return index

    def close_all(self) -> None:
        for index in self._cache.values():
            index.close()
        self._cache.clear()

    def close_index(self, kb_id: str) -> None:
        index = self._cache.pop(kb_id, None)
        if index:
            index.close()
