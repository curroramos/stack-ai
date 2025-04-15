from typing import List, Dict, Optional, Any
from uuid import uuid4
from pydantic import BaseModel, Field, ConfigDict
from app.utils.indexing import LinearIndex
from .chunk_models import Chunk
from .metadata_models import LibraryMetadata

class Library(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    documents: List[Any] = Field(default_factory=list)
    metadata: Optional[LibraryMetadata] = None
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
    metadata: Optional[LibraryMetadata] = None

class LibraryResponse(BaseModel):
    id: str
    name: str
    documents: List[Any]
    metadata: Optional[LibraryMetadata] = None

