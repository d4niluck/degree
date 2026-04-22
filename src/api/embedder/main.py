import logging
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()

from src.preprocessing import Embedder, get_process_memory_usage


class EmbedderRequest(BaseModel):
    texts: List[str]
    batch_size: int = 256
    normalize: bool = True
    text_type: Optional[str] = None


embedder = Embedder(
    device=os.getenv('EMBEDDER_DEVICE') or 'cpu',
    model_name='ai-forever/FRIDA',
    type_handlers={
        'query': lambda text: f'search_query: {text}',
        'document': lambda text: f'search_document: {text}',
    }
)

app = FastAPI()
logger = logging.getLogger(__name__)

@app.post("/embeddings")
def get_embeddings(payload: EmbedderRequest):
    logger.info("Embedder request start | memory=%s", get_process_memory_usage())
    embedd = None
    try:
        embedd = embedder.get_embeddings(
            texts=payload.texts,
            batch_size=payload.batch_size,
            normalize=payload.normalize,
            text_type=payload.text_type,
        )
        return {"embeddings": embedd.tolist()}
    finally:
        if embedd is not None:
            del embedd
        embedder.clear_torch_cache()
        logger.info("Embedder request end | memory=%s", get_process_memory_usage())


@app.get("/embeddings/memory")
def get_memory_usage():
    memory = get_process_memory_usage()
    memory["device"] = str(getattr(embedder.model, "device", memory.get("device", "unknown")))
    return memory

# uvicorn src.api.embedder.main:app --host 127.0.0.1 --port 8001 --reload
