import numpy as np
import requests
from typing import List
from sentence_transformers import SentenceTransformer
import torch
import gc


class Embedder:
    def __init__(self, model_name: str = "sergeyzh/LaBSE-ru-turbo", device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)

    def get_embeddings(self, texts, batch_size: int = 100, normalize: bool = True):
        with torch.inference_mode():
            embs = self.model.encode(
                texts, batch_size=batch_size, convert_to_numpy=True,
                normalize_embeddings=normalize, show_progress_bar=False
            )
        return embs

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
