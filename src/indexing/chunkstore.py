import logging
import os
import re
import shutil
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional
import json

from ..preprocessing.schema import Chunk
from ..preprocessing.chunker import BaseChunker


class ChunkStore:
    def __init__(self, store_dir: str, logger: logging.Logger):
        self.logger = logger
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)
        self.logger.debug(f"Chunk store directory is ready: {self.store_dir}")
        self._ws = re.compile(r"\s+")
        self._keep = re.compile(r"[^0-9a-zа-яё.,:;!?()\[\]\"'/%+\- ]+", re.IGNORECASE)

    def add(self, chunk: Chunk) -> None:
        chunk_id = self._get_chunk_id(chunk)
        chunk.chunk_id = chunk_id
        save_path = self._create_chunk_path(chunk_id)
        if not save_path:
            self.logger.info(f"File {chunk_id}.json already exists")
        else:
            self._save_chunk(chunk, save_path)

    def read(self, path: Optional[str] = None, chunk_id: Optional[str] = None) -> Optional[Chunk]:
        if path:
            chunk = self._read_json(path)
        elif chunk_id:
            path = self.chunk_id2path(chunk_id)
            chunk = self._read_json(path)
        else:
            self.logger.error("Need file path or chunk id")
            chunk = None
        return chunk

    def delete(self, path: Optional[str] = None, chunk_id: Optional[str] = None) -> None:
        if path:
            self._delete_chunk(path)
        elif chunk_id:
            path = self.chunk_id2path(chunk_id)
            self._delete_chunk(path)
        else:
            self.logger.error("Need file path or chunk id")

    def clear(self) -> None:
        for name in os.listdir(self.store_dir):
            path = os.path.join(self.store_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        self.logger.info(f"ChunkStore data removed from {self.store_dir}")

    def chunk_id2path(self, chunk_id: str) -> Optional[str]:
        path = os.path.join(self.store_dir, f"{chunk_id}.json")
        return path if os.path.exists(path) else None

    def path2chunk_id(self, path: str) -> Optional[str]:
        if path and os.path.exists(path):
            chunk_id = path.split("/")[-1].split(".")[0]
        else:
            chunk_id = None
        return chunk_id

    def get_list_chunk_id(self) -> List[str]:
        chunk_ids = []
        for path in self.get_list_chunk_path():
            chunk = self.read(path)
            if not chunk["chunk_id"]:
                self.logger.error(f"File {path} without chunk_id")
            else:
                chunk_ids.append(chunk["chunk_id"])
        self.logger.debug(f"{len(chunk_ids)} files were read")
        return chunk_ids

    def get_list_chunk_path(self) -> List[str]:
        files_names = os.listdir(self.store_dir)
        paths = [os.path.join(self.store_dir, file_name) for file_name in files_names]
        self.logger.debug(f"{len(paths)} files were found")
        return paths

    def info(self) -> None:
        paths = self.get_list_chunk_path()
        total_bytes = sum(os.path.getsize(path) for path in paths if os.path.exists(path))
        total_mb = total_bytes / (1024 * 1024)
        print(f"{len(paths)} files, {total_mb:.2f} MB total")

    def _save_chunk(self, chunk: Chunk, path: str) -> None:
        try:
            payload = asdict(chunk)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            self.logger.debug(f"Successfully save {path}")
        except Exception as e:
            self.logger.error(f"Save {path} error:\n{e}")

    def _read_json(self, path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.logger.debug(f"Successfully read {path}")
            return payload
        except Exception as e:
            self.logger.error("Read {path} error:\n{e}")

    def _delete_chunk(self, path: str) -> None:
        if path and os.path.exists(path):
            os.remove(path)
            self.logger.info(f"File {path} deleted")
        else:
            self.logger.error(f"File {path} not exists")

    def _create_chunk_path(self, chunk_id: str) -> Optional[str]:
        path = os.path.join(self.store_dir, f"{chunk_id}.json")
        return path if not os.path.exists(path) else None

    def _get_chunk_id(self, chunk: Chunk) -> str:
        if chunk.chunk_id:
            return chunk.chunk_id
        if chunk.text:
            return BaseChunker._make_chunk_id_from_text(chunk.text)
        return uuid.uuid4().hex

    def _canonicalize_text(self, s: str) -> str:
        s = s.replace("\u00ad", "").lower()
        s = self._ws.sub(" ", s).strip()
        s = self._keep.sub("", s)
        return s
