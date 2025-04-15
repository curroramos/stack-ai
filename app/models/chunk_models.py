from uuid import uuid4
from typing import List, Optional
from pydantic import BaseModel, Field
from .metadata_models import ChunkMetadata

class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    embedding: List[float]
    metadata: Optional[ChunkMetadata] = None

class ChunkInput(BaseModel):
    text: str
    metadata: Optional[ChunkMetadata] = None
