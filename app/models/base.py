from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field, ConfigDict
from fastapi import Query
from uuid import uuid4
from app.utils.indexing import LinearIndex

class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    embedding: List[float]
    metadata: Optional[Dict[str, Any]] = None

class ChunkInput(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None

class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    chunks: List[Chunk] = Field(default_factory=list)
    metadata: Optional[Dict[str, str]] = None

class Library(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    documents: List[Any] = Field(default_factory=list)
    metadata: Optional[Dict[str, str]] = None
    index: LinearIndex = Field(default_factory=LinearIndex, exclude=True)
    chunk_map: Dict[str, Chunk] = Field(default_factory=dict)

    def add_chunk(self, chunk: Chunk):
        self.chunk_map[chunk.id] = chunk
        self.index.add_vector(chunk.embedding, chunk.id)

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        return self.chunk_map.get(chunk_id)

    model_config = ConfigDict(arbitrary_types_allowed=True)

class LibraryCreate(BaseModel):
    name: str
    metadata: Optional[Dict[str, str]] = None

class LibraryResponse(BaseModel):
    id: str
    name: str
    documents: List[Any]
    metadata: Optional[Dict[str, str]] = None


class QueryRequest(BaseModel):
    library_id: str
    query_text: str
    k: int = Query(default=5, ge=1)
    distance_metric: Literal["euclidean", "cosine"] = "euclidean"

class QueryResult(BaseModel):
    chunk_id: str
    score: float
    chunk_content: str  # or some subset of chunk fields
