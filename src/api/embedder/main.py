from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
import psutil
import torch
load_dotenv()

from src.preprocessing import Embedder


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

@app.post("/embeddings")
def get_embeddings(payload: EmbedderRequest):
    embedd = embedder.get_embeddings(
        texts=payload.texts,
        batch_size=payload.batch_size,
        normalize=payload.normalize,
        text_type=payload.text_type,
    )
    return {"embeddings": embedd.tolist()}


@app.get("/embeddings/memory")
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory = {
        "device": str(getattr(embedder.model, "device", "unknown")),
        "cpu_rss_mb": process.memory_info().rss / (1024 * 1024),
    }

    if torch.cuda.is_available():
        memory["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        memory["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
    elif torch.backends.mps.is_available():
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

    return memory

# uvicorn src.api.embedder.main:app --host 127.0.0.1 --port 8001 --reload