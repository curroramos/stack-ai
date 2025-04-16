from uuid import uuid4
from typing import List, Optional
from pydantic import BaseModel, Field
from .metadata_models import DocumentMetadata

class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    library_id: str
    chunk_ids: List[str] = Field(default_factory=list)
    metadata: Optional[DocumentMetadata] = None

class DocumentInput(BaseModel):
    title: str
    metadata: Optional[DocumentMetadata] = None