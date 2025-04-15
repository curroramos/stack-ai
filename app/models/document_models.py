from uuid import uuid4
from typing import List, Optional
from pydantic import BaseModel, Field
from .chunk_models import Chunk
from .metadata_models import DocumentMetadata

class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    chunks: List[Chunk] = Field(default_factory=list)
    metadata: Optional[DocumentMetadata] = None