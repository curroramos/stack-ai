from typing import List, Optional, Literal
from pydantic import BaseModel

class ChunkMetadata(BaseModel):
    source: Optional[str] = None
    created_at: Optional[str] = None
    author: Optional[str] = None
    language: Optional[str] = None

class DocumentMetadata(BaseModel):
    category: Optional[str] = None
    created_at: Optional[str] = None
    source_type: Optional[str] = None
    tags: Optional[List[str]] = None

class LibraryMetadata(BaseModel):
    created_by: Optional[str] = None
    created_at: Optional[str] = None
    use_case: Optional[str] = None
    access_level: Optional[Literal["private", "public", "restricted"]] = "private"
