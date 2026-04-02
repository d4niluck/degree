import math
import re
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


class AutoEDAIndex:
    def __init__(
        self,
        index: Any,
        similarity_batch_size: int = 1024,
    ) -> None:
        self.index = index
        self.datastore = getattr(index, "datastore", None)
        self.chunkstore = getattr(index, "chunkstore", None)
        self.vectorstore = getattr(index, "vectorstore", None)
        self.similarity_batch_size = similarity_batch_size
        self._sentence_pattern = re.compile(r"[.!?]+")
        self._paragraph_pattern = re.compile(r"\n\s*\n")

    def run(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        document_rows = self._collect_document_stats()
        chunk_rows = self._collect_chunk_stats()
        embedding_rows = self._collect_embedding_stats()

        rows.extend(document_rows)
        rows.extend(chunk_rows)
        rows.extend(embedding_rows)

        dataframe = pd.DataFrame(rows, columns=["section", "metric", "value"])
        self._print_report(dataframe)
        return dataframe

    def _collect_document_stats(self) -> List[Dict[str, Any]]:
        if self.datastore is None:
            return []

        lengths: List[int] = []
        for path in self.datastore.get_list_doc_path():
            document = self.datastore.read(path=path)
            if not document:
                continue
            text = self._document_to_text(document)
            lengths.append(len(text))

        if not lengths:
            return []

        return self._make_length_rows("documents", lengths)

    def _collect_chunk_stats(self) -> List[Dict[str, Any]]:
        if self.chunkstore is None:
            return []

        lengths: List[int] = []
        words_per_chunk: List[int] = []
        sentences_per_chunk: List[int] = []
        paragraphs_per_chunk: List[int] = []

        for path in self.chunkstore.get_list_chunk_path():
            chunk = self.chunkstore.read(path=path)
            if not chunk:
                continue
            text = chunk.get("text", "") or ""
            lengths.append(len(text))
            words_per_chunk.append(self._count_words(text))
            sentences_per_chunk.append(self._count_sentences(text))
            paragraphs_per_chunk.append(self._count_paragraphs(text))

        if not lengths:
            return []

        rows = self._make_length_rows("chunks", lengths)
        rows.append(self._row("chunks", "text_length_std", float(np.std(lengths))))
        rows.append(
            self._row("chunks", "words_mean", float(np.mean(words_per_chunk)))
        )
        rows.append(
            self._row(
                "chunks",
                "sentences_mean",
                float(np.mean(sentences_per_chunk)),
            )
        )
        rows.append(
            self._row(
                "chunks",
                "paragraphs_mean",
                float(np.mean(paragraphs_per_chunk)),
            )
        )
        return rows

    def _collect_embedding_stats(self) -> List[Dict[str, Any]]:
        if self.vectorstore is None:
            return []

        vectors = self._get_vectors()
        if vectors is None or vectors.size == 0:
            return []

        rows = [
            self._row("embeddings", "matrix_rows", int(vectors.shape[0])),
            self._row("embeddings", "matrix_cols", int(vectors.shape[1])),
            self._row("embeddings", "dtype", str(vectors.dtype)),
        ]

        dimension_std = np.std(vectors, axis=0)
        rows.extend(
            [
                self._row(
                    "embeddings",
                    "dimension_std_mean",
                    float(np.mean(dimension_std)),
                ),
                self._row(
                    "embeddings",
                    "dimension_std_min",
                    float(np.min(dimension_std)),
                ),
                self._row(
                    "embeddings",
                    "dimension_std_max",
                    float(np.max(dimension_std)),
                ),
            ]
        )

        similarity_stats = self._pairwise_similarity_stats(vectors)
        for metric, value in similarity_stats.items():
            rows.append(self._row("embeddings", metric, value))
        return rows

    def _pairwise_similarity_stats(self, vectors: np.ndarray) -> Dict[str, Any]:
        n_vectors = vectors.shape[0]
        if n_vectors < 2:
            return {
                "pairwise_similarity_pairs": 0,
                "pairwise_similarity_min": math.nan,
                "pairwise_similarity_max": math.nan,
                "pairwise_similarity_mean": math.nan,
                "pairwise_similarity_std": math.nan,
            }

        total_count = 0
        total_sum = 0.0
        total_sum_sq = 0.0
        min_similarity = math.inf
        max_similarity = -math.inf
        batch_size = max(1, self.similarity_batch_size)

        for i_start in range(0, n_vectors, batch_size):
            i_end = min(i_start + batch_size, n_vectors)
            left = vectors[i_start:i_end]
            for j_start in range(i_start, n_vectors, batch_size):
                j_end = min(j_start + batch_size, n_vectors)
                right = vectors[j_start:j_end]
                sim_block = left @ right.T

                if i_start == j_start:
                    tri_idx = np.triu_indices(sim_block.shape[0], k=1)
                    selected = sim_block[tri_idx]
                else:
                    selected = sim_block.reshape(-1)

                if selected.size == 0:
                    continue

                selected = selected.astype(np.float64, copy=False)
                total_count += int(selected.size)
                total_sum += float(np.sum(selected))
                total_sum_sq += float(np.sum(selected * selected))
                min_similarity = min(min_similarity, float(np.min(selected)))
                max_similarity = max(max_similarity, float(np.max(selected)))

        mean_similarity = total_sum / total_count
        variance = max(total_sum_sq / total_count - mean_similarity**2, 0.0)
        return {
            "pairwise_similarity_pairs": total_count,
            "pairwise_similarity_min": min_similarity,
            "pairwise_similarity_max": max_similarity,
            "pairwise_similarity_mean": mean_similarity,
            "pairwise_similarity_std": math.sqrt(variance),
        }

    def _get_vectors(self) -> Optional[np.ndarray]:
        if self.vectorstore is None:
            return None
        flush_pending = getattr(self.vectorstore, "_flush_pending", None)
        if callable(flush_pending):
            flush_pending()
        vectors = getattr(self.vectorstore, "vectors", None)
        if vectors is None:
            return None
        return np.asarray(vectors)

    @staticmethod
    def _document_to_text(document: Any) -> str:
        pages = getattr(document, "pages", []) or []
        return "\n".join((page.text or "") for page in pages)

    @staticmethod
    def _count_words(text: str) -> int:
        return len(text.split())

    def _count_sentences(self, text: str) -> int:
        sentences = [part.strip() for part in self._sentence_pattern.split(text) if part.strip()]
        return len(sentences) if sentences else int(bool(text.strip()))

    def _count_paragraphs(self, text: str) -> int:
        stripped = text.strip()
        if not stripped:
            return 0
        paragraphs = [part.strip() for part in self._paragraph_pattern.split(stripped) if part.strip()]
        return len(paragraphs) if paragraphs else 1

    @staticmethod
    def _make_length_rows(section: str, lengths: Iterable[int]) -> List[Dict[str, Any]]:
        arr = np.asarray(list(lengths), dtype=np.int64)
        return [
            AutoEDAIndex._row(section, "count", int(arr.shape[0])),
            AutoEDAIndex._row(section, "text_length_mean", float(np.mean(arr))),
            AutoEDAIndex._row(section, "text_length_median", float(np.median(arr))),
            AutoEDAIndex._row(section, "text_length_min", int(np.min(arr))),
            AutoEDAIndex._row(section, "text_length_max", int(np.max(arr))),
        ]

    @staticmethod
    def _row(section: str, metric: str, value: Any) -> Dict[str, Any]:
        return {"section": section, "metric": metric, "value": value}

    def _print_report(self, dataframe: pd.DataFrame) -> None:
        if dataframe.empty:
            print("AutoEDAIndex: no data available")
            return

        for section in dataframe["section"].drop_duplicates().tolist():
            print(section.upper())
            section_df = dataframe[dataframe["section"] == section]
            for _, row in section_df.iterrows():
                print(f"- {row['metric']}: {row['value']}")
            print()
