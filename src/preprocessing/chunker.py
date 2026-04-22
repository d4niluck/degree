import logging
import hashlib
import re
from typing import List, Optional

from .schema import Chunk, Document, Page


class BaseChunker:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def split_document_to_chunk(self, document: Document) -> List[Chunk]:
        raise NotImplementedError()
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("\u00a0", " ")  # замена неразрывного пробела на обычный 
        text = re.sub(r"[ \t]+", " ", text) # сжатие пробелов и табов в один пробел
        text = re.sub(r"\r\n?", "\n", text) # нормализация переводов
        return text.strip()
    
    @staticmethod
    def _make_chunk_id_from_text(text: str, max_len: int = 4000) -> str:
        norm = BaseChunker._normalize_text(text)
        payload = norm[:max_len].encode("utf-8", errors="ignore")
        return hashlib.sha256(payload).hexdigest()
    
    def _make_chunk(self, chunk_id, doc_id, source_path,
                    text, index, page_number, extra_context):
        return Chunk(
            chunk_id,
            doc_id, 
            source_path,
            text,
            index,
            page_number,
            extra_context
        )


class FixedCharChunker(BaseChunker):
    def __init__(
        self, 
        logger: logging.Logger, 
        chunk_size: int = 1000,
        overlap: int = 200,
        min_chunk_chars: int = 50
    ):
        super().__init__(logger)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_chars = min_chunk_chars

    def split_document_to_chunk(self, document: Document):
        chunks: List[Chunk] = []
        chunk_index = 0
        
        for page in document.pages:
            page_text = self._normalize_text(page.text)
            page_number = page.number
            if not page_text:
                continue
            
            step = self.chunk_size - self.overlap
            for i in range(0, len(page_text), step):
                chunk_text = page_text[i:min(i+self.chunk_size, len(page_text))]
                chunk_id = self._make_chunk_id_from_text(chunk_text)
                if len(chunk_text) > self.min_chunk_chars:
                    chunk = self._make_chunk(
                        chunk_id=chunk_id,
                        doc_id=document.doc_id,
                        source_path=document.source_path,
                        text=chunk_text,
                        index=chunk_index,
                        page_number=page_number,
                        extra_context=None,
                    )
                    chunks.append(chunk)
                chunk_index += 1
            
        self.logger.debug(f"FixedCharChunker: produced {len(chunks)} chunks from {len(document.pages)} pages")
        return chunks


class ParagraphChunker(BaseChunker):
    def __init__(
        self,
        logger: logging.Logger,
        max_length: int = 490,
        overlap_sentences: int = 1,
        min_chunk_chars: int = 50,
    ):
        super().__init__(logger)
        self.max_length = max_length
        self.overlap_sentences = overlap_sentences
        self.min_chunk_chars = min_chunk_chars

    def split_document_to_chunk(self, document: Document) -> List[Chunk]:
        chunks: List[Chunk] = []
        chunk_index = 0

        for page in document.pages:
            page_text = self._normalize_text(page.text)
            if not page_text:
                continue

            paragraphs = self._split_page_to_paragraphs(page_text)
            for paragraph in paragraphs:
                paragraph_chunks = self._split_paragraph(paragraph)
                for chunk_text in paragraph_chunks:
                    if len(chunk_text) < self.min_chunk_chars:
                        continue
                    chunk = self._make_chunk(
                        chunk_id=self._make_chunk_id_from_text(chunk_text),
                        doc_id=document.doc_id,
                        source_path=document.source_path,
                        text=chunk_text,
                        index=chunk_index,
                        page_number=page.number,
                        extra_context=None,
                    )
                    chunks.append(chunk)
                    chunk_index += 1

        self.logger.debug(
            "ParagraphChunker: produced %s chunks from %s pages",
            len(chunks),
            len(document.pages),
        )
        return chunks

    def _split_page_to_paragraphs(self, text: str) -> List[str]:
        raw_paragraphs = re.split(r"\n\s*\n+", text)
        paragraphs = []
        for paragraph in raw_paragraphs:
            cleaned = re.sub(r"\s*\n\s*", " ", paragraph).strip()
            if cleaned:
                paragraphs.append(cleaned)
        return paragraphs

    def _split_paragraph(self, paragraph: str) -> List[str]:
        if len(paragraph) <= self.max_length:
            return [paragraph]

        sentences = self._split_text_to_sentences(paragraph)
        if not sentences:
            return [paragraph]
        units: List[str] = []
        for sentence in sentences:
            units.extend(self._split_long_sentence(sentence))
        return self._build_sentence_chunks(units)

    def _build_sentence_chunks(self, units: List[str]) -> List[str]:
        chunks: List[str] = []
        start = 0

        while start < len(units):
            current_units: List[str] = []
            index = start

            while index < len(units):
                candidate_units = current_units + [units[index]]
                candidate_text = " ".join(candidate_units).strip()
                if current_units and len(candidate_text) > self.max_length:
                    break
                current_units = candidate_units
                index += 1

            if not current_units:
                current_units = [units[start]]
                index = start + 1

            chunks.append(" ".join(current_units).strip())
            if index >= len(units):
                break

            overlap_count = self._get_overlap_count(
                current_units=current_units,
                next_unit=units[index],
            )
            start = index - overlap_count

        return chunks

    def _get_overlap_count(self, current_units: List[str], next_unit: str) -> int:
        if self.overlap_sentences <= 0:
            return 0

        max_count = min(self.overlap_sentences, len(current_units))
        for count in range(max_count, 0, -1):
            overlap_units = current_units[-count:]
            candidate_text = " ".join(overlap_units + [next_unit]).strip()
            if len(candidate_text) <= self.max_length:
                return count
        return 0

    def _split_text_to_sentences(self, text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        sentences = [part.strip() for part in parts if part.strip()]
        return sentences

    def _split_long_sentence(self, sentence: str) -> List[str]:
        if len(sentence) <= self.max_length:
            return [sentence]

        separators = r"(?<=[,;:])\s+|(?<=\))\s+|(?<=])\s+|(?<=-)\s+"
        parts = [part.strip() for part in re.split(separators, sentence) if part.strip()]
        if len(parts) <= 1:
            return self._split_by_words(sentence)
        return self._merge_parts(parts)

    def _merge_parts(self, parts: List[str]) -> List[str]:
        merged: List[str] = []
        current = ""

        for part in parts:
            candidate = f"{current} {part}".strip()
            if current and len(candidate) > self.max_length:
                merged.append(current)
                current = part
            else:
                current = candidate

        if current:
            merged.append(current)

        result: List[str] = []
        for item in merged:
            if len(item) <= self.max_length:
                result.append(item)
            else:
                result.extend(self._split_by_words(item))
        return result

    def _split_by_words(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []

        chunks: List[str] = []
        current_words: List[str] = []
        for word in words:
            candidate_words = current_words + [word]
            candidate_text = " ".join(candidate_words)
            if current_words and len(candidate_text) > self.max_length:
                chunks.append(" ".join(current_words))
                current_words = [word]
            else:
                current_words = candidate_words

        if current_words:
            chunks.append(" ".join(current_words))
        return chunks


class SentenceChunker(BaseChunker):
    ...


class SlidingWindowChunker(BaseChunker):
    ...
