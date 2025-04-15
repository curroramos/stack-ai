from fastapi import APIRouter, HTTPException
from uuid import uuid4
from datetime import datetime, timezone
from app.models.chunk_models import Chunk, ChunkInput
from app.models.metadata_models import ChunkMetadata
from app.core.db import db
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

    document = next((doc for doc in library.documents if str(doc.id) == document_id), None)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    embedding = get_embedding(chunk_input.text)

    metadata = ChunkMetadata(**chunk_input.metadata.model_dump()) if chunk_input.metadata else ChunkMetadata()
    metadata.created_at = now_iso()

    new_chunk = Chunk(
        id=str(uuid4()),
        text=chunk_input.text,
        embedding=embedding,
        metadata=metadata
    )

    document.chunks.append(new_chunk)
    library.add_chunk(new_chunk)
    
    db.update_library(library)

    return new_chunk

@router.get("/")
def list_chunks(library_id: str, document_id: str):
    library = db.get_library(library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")

    document = next((doc for doc in library.documents if str(doc.id) == document_id), None)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return document.chunks

@router.get("/{chunk_id}")
def get_chunk(library_id: str, document_id: str, chunk_id: str):
    library = db.get_library(library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")

    document = next((doc for doc in library.documents if str(doc.id) == document_id), None)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    chunk = next((c for c in document.chunks if str(c.id) == chunk_id), None)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")

    return chunk

@router.put("/{chunk_id}")
def update_chunk(library_id: str, document_id: str, chunk_id: str, chunk_input: ChunkInput):
    library = db.get_library(library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")

    document = next((doc for doc in library.documents if str(doc.id) == document_id), None)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    chunk_index = next((i for i, c in enumerate(document.chunks) if str(c.id) == chunk_id), -1)
    if chunk_index == -1:
        raise HTTPException(status_code=404, detail="Chunk not found")

    embedding = get_embedding(chunk_input.text)

    metadata = ChunkMetadata(**chunk_input.metadata.model_dump()) if chunk_input.metadata else ChunkMetadata()
    metadata.created_at = now_iso()

    updated_chunk = Chunk(
        id=chunk_id,
        text=chunk_input.text,
        embedding=embedding,
        metadata=metadata
    )

    document.chunks[chunk_index] = updated_chunk
    library.chunk_map[chunk_id] = updated_chunk
    library.index.update_vector(chunk_id, embedding) if hasattr(library.index, "update_vector") else None

    db.update_library(library)
    return updated_chunk

@router.delete("/{chunk_id}")
def delete_chunk(library_id: str, document_id: str, chunk_id: str):
    library = db.get_library(library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")

    document = next((doc for doc in library.documents if str(doc.id) == document_id), None)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    original_length = len(document.chunks)
    document.chunks = [c for c in document.chunks if str(c.id) != chunk_id]
    library.chunk_map.pop(chunk_id, None)

    if len(document.chunks) == original_length:
        raise HTTPException(status_code=404, detail="Chunk not found")

    # remove or reindex
    if hasattr(library.index, "remove_vector"):
        library.index.remove_vector(chunk_id)
    else:
        library.index.rebuild(library.chunk_map)

    db.update_library(library)
    return {"detail": "Chunk deleted"}
