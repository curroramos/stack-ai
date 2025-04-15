from typing import Literal
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    library_id: str
    query_text: str
    k: int = Field(default=5, ge=1)
    distance_metric: Literal["euclidean", "cosine"] = "euclidean"

class QueryResult(BaseModel):
    chunk_id: str
    score: float
    chunk_content: str  # or some subset of chunk fields