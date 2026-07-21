
from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    query: str
    ticker: str
    top_k: Optional[int] = 5
    alpha: Optional[float] = 0.5


class Source(BaseModel):
    section: str
    page: float
    chunk_type: str
    child_score: float


class QueryResponse(BaseModel):
    query: str
    ticker: str
    answer: str
    route: dict
    sources: list[Source]
    usage: dict
    latency_ms: int
    prompt_version: int


class HealthResponse(BaseModel):
    status: str
    docstore_parents: int
    index: str
    embedding_model: str
    llm: str