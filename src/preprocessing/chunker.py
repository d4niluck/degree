import logging
from .schema import Document, Page, Chunk
from typing import Optional, List
import hashlib
import re


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
    ...


class SentenceChunker(BaseChunker):
    ...


class SlidingWindowChunker(BaseChunker):
    ...
