from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import logging

from src.preprocessing import Reader

debug = False
logger = logging.getLogger()
logging.basicConfig(
    level=logging.DEBUG if debug else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


class ReaderRequest(BaseModel):
    file_path: str
    forced_ocr: bool = False


class PagePayload(BaseModel):
    number: int
    text: str


class DocumentPayload(BaseModel):
    source_path: str
    doc_id: str | None = None
    pages: list[PagePayload]


reader = Reader(logger)

app = FastAPI()

@app.post("/reader/read")
def read_document(payload: ReaderRequest):
    document = reader.read(payload.file_path, forced_ocr=payload.forced_ocr)
    if document is None:
        raise HTTPException(status_code=422, detail="Failed to read document")
    return {
        "source_path": document.source_path,
        "doc_id": document.doc_id,
        "pages": [{"number": page.number, "text": page.text} for page in document.pages]
        }
    
# uvicorn src.api.reader.main:app --host 127.0.0.1 --port 8002 --reload