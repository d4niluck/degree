import logging
import numpy as np
import requests
from typing import Callable, Dict, Iterable, List, Optional
from sentence_transformers import SentenceTransformer
import torch
import gc
import psutil


class Embedder:
    def __init__(
        self,
        model_name: str = "sergeyzh/LaBSE-ru-turbo",
        device: str = "cpu",
        type_handlers: Optional[Dict[str, Callable[[str], str]]] = None,
    ):
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        self.type_handlers = type_handlers or {}

    def get_embeddings(
        self,
        texts: Iterable[str],
        batch_size: int = 256,
        normalize: bool = True,
        text_type: Optional[str] = None,
    ):
        texts = list(texts)
        texts = self._prepare_texts(texts=texts, text_type=text_type)
        with torch.inference_mode():
            embs = self.model.encode(
                texts, batch_size=batch_size, convert_to_numpy=True,
                normalize_embeddings=normalize, show_progress_bar=False
            )
        return embs

    def _prepare_texts(self, texts: List[str], text_type: Optional[str] = None) -> List[str]:
        if not text_type:
            return texts
        handler = self.type_handlers.get(text_type)
        if handler is None:
            return texts
        return [handler(text) for text in texts]

    @staticmethod
    def clear_torch_cache() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass


class HTTPEmbedder:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001",
        timeout: float = 120.0,
        max_texts_per_request: int = 64,
        logger: Optional[logging.Logger] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_texts_per_request = max_texts_per_request
        self.session = requests.Session()
        self.logger = logger
        
    def get_embeddings(
        self,
        texts,
        batch_size: int = 100,
        normalize: bool = True,
        text_type: str | None = None,
    ):
        texts = list(texts)
        batches = self._split_texts(texts)

        all_embeddings = []
        for batch_index, batch in enumerate(batches, start=1):
            self._log_memory(
                f"HTTPEmbedder: batch {batch_index}/{len(batches)} before request"
            )
            response = self.session.post(
                f"{self.base_url}/embeddings",
                json={
                    "texts": batch,
                    "batch_size": batch_size,
                    "normalize": normalize,
                    "text_type": text_type,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            all_embeddings.extend(payload["embeddings"])
            self._log_memory(
                f"HTTPEmbedder: batch {batch_index}/{len(batches)} after request"
            )
        return np.asarray(all_embeddings, dtype=np.float32)

    def get_memory_usage(self):
        response = self.session.get(
            f"{self.base_url}/embeddings/memory",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _split_texts(self, texts):
        if len(texts) > self.max_texts_per_request:
            step = self.max_texts_per_request
            batches = [texts[i:i+step] for i in range(0, len(texts), step)]
            return batches
        else:
            batches = [texts]
        return batches

    def _log_memory(self, prefix: str) -> None:
        if self.logger is None:
            return
        try:
            memory = self.get_memory_usage()
            self.logger.info("%s | memory=%s", prefix, memory)
        except Exception as exc:
            self.logger.warning("%s | failed to fetch memory usage: %s", prefix, exc)


def get_process_memory_usage() -> Dict[str, Optional[float] | str]:
    process = psutil.Process()
    memory: Dict[str, Optional[float] | str] = {
        "cpu_rss_mb": process.memory_info().rss / (1024 * 1024),
    }

    if torch.cuda.is_available():
        memory["device"] = "cuda"
        memory["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        memory["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
    elif torch.backends.mps.is_available():
        memory["device"] = "mps"
        try:
            memory["mps_current_allocated_mb"] = (
                torch.mps.current_allocated_memory() / (1024 * 1024)
            )
            memory["mps_driver_allocated_mb"] = (
                torch.mps.driver_allocated_memory() / (1024 * 1024)
            )
        except Exception:
            memory["mps_current_allocated_mb"] = None
            memory["mps_driver_allocated_mb"] = None
    else:
        memory["device"] = "cpu"

    return memory
