from typing import List, Literal
from pydantic import BaseModel

class ChunkMetadata(BaseModel):
    source: str
    created_at: str
    author: str
    language: str

class DocumentMetadata(BaseModel):
    category: str
    created_at: str
    source_type: str
    tags: List[str]

class LibraryMetadata(BaseModel):
    created_by: str
    created_at: str
    use_case: str
    access_level: Literal["private", "public", "restricted"] = "private"
