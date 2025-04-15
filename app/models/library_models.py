from typing import List, Dict, Optional, Any
from uuid import uuid4
from pydantic import BaseModel, Field, ConfigDict
from app.utils.indexing.linear_index import LinearIndex
from .chunk_models import Chunk
from .metadata_models import LibraryMetadata
from .document_models import Document

class Library(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    documents: Dict[str, Document] = Field(default_factory=dict)
    metadata: Optional[LibraryMetadata] = None
    index: LinearIndex = Field(default_factory=LinearIndex, exclude=True)
    chunk_map: Dict[str, Chunk] = Field(default_factory=dict)

    def add_document(self, document: Document, chunks: List[Chunk]):
        self.documents[document.id] = document
        for chunk in chunks:
            self.chunk_map[chunk.id] = chunk
            document.chunk_ids.append(chunk.id)
        self.index.rebuild(self.chunk_map)

    def remove_document(self, document_id: str):
        document = self.documents.pop(document_id, None)
        if document:
            for chunk_id in document.chunk_ids:
                self.chunk_map.pop(chunk_id, None)
            self.index.rebuild(self.chunk_map)

    def update_document(self, document_id: str, new_document: Document, new_chunks: List[Chunk]):
        self.remove_document(document_id)
        self.add_document(new_document, new_chunks)

    def add_chunk_to_document(self, document_id: str, chunk: Chunk):
        document = self.documents.get(document_id)
        if document:
            self.chunk_map[chunk.id] = chunk
            document.chunk_ids.append(chunk.id)
            self.index.add_vector(chunk.embedding, chunk.id)

    def remove_chunk_from_document(self, document_id: str, chunk_id: str):
        document = self.documents.get(document_id)
        if document and chunk_id in document.chunk_ids:
            document.chunk_ids.remove(chunk_id)
            self.chunk_map.pop(chunk_id, None)
            self.index.rebuild(self.chunk_map)

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        return self.chunk_map.get(chunk_id)

    model_config = ConfigDict(arbitrary_types_allowed=True)

class LibraryCreate(BaseModel):
    name: str
    metadata: Optional[LibraryMetadata] = None

class LibraryResponse(BaseModel):
    id: str
    name: str
    documents: Dict[str, Any]
    metadata: Optional[LibraryMetadata] = None

