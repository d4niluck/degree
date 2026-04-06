from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from ...indexing import ChunkStore, DataStore, FlatVectorStore, Index
from ...preprocessing import FixedCharChunker, HTTPEmbedder, HTTPReader


class IndexManager:
    def __init__(
        self,
        storage_root: str | Path,
        logger: logging.Logger,
        embedder_url: str = "http://127.0.0.1:8001",
        reader_url: str = "http://127.0.0.1:8002",
    ) -> None:
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.embedder = HTTPEmbedder(base_url=embedder_url)
        self.reader = HTTPReader(base_url=reader_url)
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
        index = Index(
            datastore=DataStore(str(root / "documents"), self.logger),
            vectorstore=FlatVectorStore(str(root / "vectors"), self.dimensions, self.logger),
            chunkstore=ChunkStore(str(root / "chunks"), self.logger),
            chunker=FixedCharChunker(logger=self.logger, chunk_size=500, overlap=50),
            embedder=self.embedder,
            reader=self.reader,
            sqlite_path=str(root / "index.db"),
            logger=self.logger,
        )
        return index

    def close_all(self) -> None:
        for index in self._cache.values():
            index.close()
        self._cache.clear()
