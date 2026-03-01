import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import json


class VetorStore:
    def __init__(
        self, 
        store_dir: str, 
        dimensions: int, 
        logger: logging.Logger
    ):
        self.store_dir = store_dir
        self.dimensions = dimensions
        self.logger = logger
        os.makedirs(self.store_dir, exist_ok=True)
        self.logger.debug(f"Vector store directory is ready: {self.store_dir}")
        
    def add(self, vectors: np.ndarray) -> bool:
        ...
        
    def get(self, idx: int | List[int]) -> Optional[np.ndarray]:
        ...
                
    def update(self, idx: int | List[int], vectors: np.ndarray) -> bool:
        ...

    def delete(self, idx: int | List[int]) -> Optional[Dict[int, int]]: # возвращаем словарь изменений индексов
        ...
        
    def search(
        self,
        vector: np.ndarray,
        top_k: int = 3,
        allowed_idx: Optional[List[int]] = None,
    ) -> Tuple[List[int], List[float]]:
        ...
    
    def get_list_idx(self) -> List[int]:
        ...
        
    def save(self) -> None:
        ...
        
    def load(self) -> None:
        ...
        
    def clear(self) -> None:
        ...
        
    def _check_vectors_dims_to_store_upload(self, vectors: np.ndarray) -> bool:
        if vectors.shape[-1] != self.dimensions or len(vectors.shape) > 2:
            self.logger.error(f"Vector dimensions {vectors.shape[-1]} doesn't match store dimensions {self.dimensions}")
            self.logger.error(f'len(vectors.shape) must be <= 2, now {vectors.shape}')
            return False
        return True
        
    @staticmethod
    def norm_vectors(vectors: np.ndarray) -> np.ndarray:
        normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return normalized_vectors
    

class FlatVectorStore(VetorStore):
    def __init__(
        self, 
        store_dir: str, 
        dimensions: int, 
        logger: logging.Logger
    ):
        super().__init__(store_dir, dimensions, logger)
        self.vectors = np.array([]).reshape(0, dimensions)
        self.buffer: List[np.ndarray] = []
        
    def add(self, vectors: np.ndarray) -> bool:        
        if not isinstance(vectors, np.ndarray):
            self.logger.error('vectors must be np.ndarray')
            return False
        
        if not self._check_vectors_dims_to_store_upload(vectors):
            return False
        
        normalized_vectors = VetorStore.norm_vectors(vectors)
        self.buffer.append(normalized_vectors)
        self.logger.debug(f'Add {vectors.shape[0]} vectors in store')
        return True
    
    def get(self, idx: int | List[int]) -> Optional[np.ndarray]:
        self._flush_pending()
        max_idx = self.vectors.shape[0]
        if isinstance(idx, int) and 0 <= idx <= max_idx:
            return self.vectors[idx]
        elif isinstance(idx, list) and min(idx) >= 0 and max(idx) <= max_idx:
            return self.vectors[idx]
        else:
            self.logger.info(f'No vectors for {idx} idx')
            return None
    
    def update(self, idx: int | List[int], vectors: np.ndarray) -> bool:
        self._flush_pending()
        if isinstance(idx, int):
            idx = [idx]
        
        if not isinstance(idx, list):
            self.logger.error(f'idx must be int or List[int], not {idx}')
            return False
        
        if min(idx) < 0 or max(idx) > self.vectors.shape[0]:
            self.logger.error(f'Indexes are unavailable {idx}')
            return False
        
        if len(idx) != vectors.shape[0]:
            self.logger.error(f'The length of the indexes ({len(idx)}) does not match the number of vectors ({vectors.shape[0]})')
            return False
        
        if not self._check_vectors_dims_to_store_upload(vectors):
            return False
        
        normalized_vectors = VetorStore.norm_vectors(vectors)
        self.vectors[idx] = normalized_vectors
        return True
    
    def delete(self, idx: int | List[int]) -> Optional[Dict[int, int]]:
        self._flush_pending()
        if isinstance(idx, int):
            idx = [idx]
        
        if min(idx) < 0 or max(idx) > self.vectors.shape[0]:
            self.logger.error(f'Indexes are unavailable {idx}')
            return 
        
        idx = sorted(list(set(idx)))
        n_old = self.vectors.shape[0]
        to_delete = set(idx)
        self.vectors = np.delete(self.vectors, idx, axis=0)
        
        old2new_map = {}
        deleted_count = 0
        for i in range(n_old):
            if i in to_delete:
                deleted_count += 1
                continue
            new_i = i - deleted_count
            if new_i != i:
                old2new_map[i] = new_i
        self.logger.info(f'Deleted {len(idx)} vectors')
        return old2new_map

    def search(
        self,
        vector: np.ndarray,
        top_k: int = 3,
        allowed_idx: Optional[List[int]] = None,
    ) -> Tuple[List[int], List[float]]:
        self._flush_pending()
        if not isinstance(vector, np.ndarray) or vector.shape[-1] != self.dimensions:
            self.logger.error(f'Vector must be np.ndarray and vector.shape[-1] must be {self.dimensions}')
            return [], []

        if self.vectors.shape[0] == 0:
            return [], []

        if allowed_idx is not None:
            valid_idx = [i for i in allowed_idx if 0 <= i < self.vectors.shape[0]]
            if not valid_idx:
                return [], []
            candidate_idx = np.array(sorted(set(valid_idx)), dtype=int)
            candidate_vectors = self.vectors[candidate_idx]
            sims = (candidate_vectors @ vector.T).ravel()
            top_k = min(max(1, top_k), sims.shape[0])
            local_idx = np.argpartition(sims, -top_k)[-top_k:]
            local_idx = local_idx[np.argsort(sims[local_idx])[::-1]]
            idx = candidate_idx[local_idx]
            return idx.tolist(), sims[local_idx].tolist()

        top_k = min(max(1, top_k), self.vectors.shape[0])
        sims = (self.vectors @ vector.T).ravel()
        idx = np.argpartition(sims, -top_k)[-top_k:]
        idx = idx[np.argsort(sims[idx])[::-1]]
        return idx.tolist(), sims[idx].tolist()
    
    def get_list_idx(self) -> List[int]:
        total_vectors = self.vectors.shape[0] + sum(block.shape[0] for block in self.buffer)
        return list(range(total_vectors))

    def clear(self) -> None:
        for name in os.listdir(self.store_dir):
            path = os.path.join(self.store_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        self.vectors = np.array([]).reshape(0, self.dimensions)
        self.buffer.clear()
        self.logger.info(f"VectorStore data removed from {self.store_dir}")

    def save(self) -> None:
        self._flush_pending()
        os.makedirs(self.store_dir, exist_ok=True)
        path = self._vectors_path()
        try:
            np.save(path, self.vectors)
            self.logger.debug(f"Saved {self.vectors.shape[0]} vectors to {path}")
        except Exception as e:
            self.logger.exception(f"Failed to save vectors to {path}: {e}")
        
    def load(self) -> None:
        path = self._vectors_path()
        if not os.path.exists(path):
            self.logger.debug(f"No vectors file found at {path}. Store stays empty.")
            return

        vectors = np.load(path, allow_pickle=False)
        if vectors.shape[-1] != self.dimensions:
            self.logger.error(f"Loaded vectors shape doesn't match store dimensions ({self.dimensions})")
        else:
            self.vectors = vectors
            self.buffer.clear()
            self.logger.debug(f"Loaded {self.vectors.shape[0]} vectors from {path}")
        
    def _vectors_path(self) -> str:
        return os.path.join(self.store_dir, "vectors.npy")

    def _flush_pending(self) -> None:
        if not self.buffer:
            return
        self.vectors = np.concatenate([self.vectors, *self.buffer], axis=0)
        self.buffer.clear()


class GraphVectorStore(VetorStore):
    ...
