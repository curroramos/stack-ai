from fastapi import APIRouter, HTTPException
from uuid import uuid4
from datetime import datetime, timezone
from app.models.chunk_models import Chunk, ChunkInput
from app.models.metadata_models import ChunkMetadata
from app.core.db import db  # InMemoryDB instance
from app.utils.embeddings import get_embedding

router = APIRouter(
    prefix="/libraries/{library_id}/documents/{document_id}/chunks",
    tags=["chunks"]
)

def now_iso():
    return datetime.now(timezone.utc).isoformat()

@router.post("/")
def add_chunk(library_id: str, document_id: str, chunk_input: ChunkInput):
    library = db.get_library(library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")

    document = library.documents.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    embedding = get_embedding(chunk_input.text)
    metadata = ChunkMetadata(**chunk_input.metadata.model_dump()) if chunk_input.metadata else ChunkMetadata()
    metadata.created_at = now_iso()

    new_chunk = Chunk(
        id=str(uuid4()),
        text=chunk_input.text,
        document_id=document_id,
        embedding=embedding,
        metadata=metadata
    )

    library.chunk_map[new_chunk.id] = new_chunk
    document.chunk_ids.append(new_chunk.id)  

    indexing_service = db.get_indexing_service(library_id)
    if not indexing_service:
        raise HTTPException(status_code=500, detail="Indexing service not initialized for this library")
    indexing_service.add_chunk(new_chunk)

    db.update_library(library)

    return new_chunk

@router.get("/")
def list_chunks(library_id: str, document_id: str):
    library = db.get_library(library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")

    document = library.documents.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return [library.chunk_map[chunk_id] for chunk_id in document.chunk_ids]

@router.get("/{chunk_id}")
def get_chunk(library_id: str, document_id: str, chunk_id: str):
    library = db.get_library(library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")

    document = library.documents.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if chunk_id not in document.chunk_ids:
        raise HTTPException(status_code=404, detail="Chunk not found")

    chunk = library.chunk_map.get(chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")

    return chunk

@router.put("/{chunk_id}")
def update_chunk(library_id: str, document_id: str, chunk_id: str, chunk_input: ChunkInput):
    library = db.get_library(library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")

    document = library.documents.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Validate chunk belongs to the document
    if chunk_id not in document.chunk_ids:
        raise HTTPException(status_code=404, detail="Chunk not found in document")

    # Update metadata
    embedding = get_embedding(chunk_input.text)
    metadata = ChunkMetadata(**chunk_input.metadata.model_dump()) if chunk_input.metadata else ChunkMetadata()
    metadata.created_at = now_iso()

    updated_chunk = Chunk(
        id=chunk_id,
        text=chunk_input.text,
        document_id=document_id,
        embedding=embedding,
        metadata=metadata
    )

    # Replace chunk in chunk_map
    library.chunk_map[chunk_id] = updated_chunk
    
    # Rebuild with updated chunk
    indexing_service = db.get_indexing_service(library_id)
    if indexing_service:
        indexing_service.rebuild_index(library.chunk_map)

    db.update_library(library)
    return updated_chunk

@router.delete("/{chunk_id}")
def delete_chunk(library_id: str, document_id: str, chunk_id: str):
    library = db.get_library(library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")

    document = library.documents.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Check if the chunk is actually in the document
    if chunk_id not in document.chunk_ids:
        raise HTTPException(status_code=404, detail="Chunk not found in document")

    # Remove chunk ID reference from document
    document.chunk_ids.remove(chunk_id)

    # Remove from chunk_map
    library.chunk_map.pop(chunk_id, None)

    # Rebuild without deleted chunk
    indexing_service = db.get_indexing_service(library_id)
    if indexing_service:
        indexing_service.rebuild_index(library.chunk_map)

    # Update the library
    db.update_library(library)

    return {"detail": "Chunk deleted"}