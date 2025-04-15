from uuid import uuid4
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from .chunk_models import Chunk
from .metadata_models import DocumentMetadata

class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    chunk_ids: List[str] = Field(default_factory=list)
    metadata: Optional[DocumentMetadata] = None
