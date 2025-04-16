from fastapi import APIRouter, HTTPException
from uuid import uuid4
from typing import List
from datetime import datetime, timezone
from app.core.db import db
from app.models import Document, DocumentMetadata, DocumentInput

router = APIRouter(prefix="/libraries/{library_id}/documents", tags=["documents"])

def now_iso():
    return datetime.now(timezone.utc).isoformat()

@router.post("/", response_model=Document)
def create_document(library_id: str, document_input: DocumentInput):
    library = db.get_library(library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")

    meta_dict = (
        document_input.metadata.model_dump()
        if document_input.metadata
        else {}
    )
    meta_dict.setdefault("created_at", now_iso())

    document = Document(
        id=str(uuid4()),
        title=document_input.title,
        library_id=library_id,
        metadata=DocumentMetadata(**meta_dict),
    )

    library.documents[document.id] = document
    db.update_library(library)

    return document

@router.get("/")
def list_documents(library_id: str) -> List[Document]:
    library = db.get_library(library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    return list(library.documents.values())

@router.get("/{document_id}", response_model=Document)
def get_document(library_id: str, document_id: str):
    library = db.get_library(library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")

    document = library.documents.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return document


@router.put("/{document_id}", response_model=Document)
def update_document(library_id: str, document_id: str, updated_doc_input: DocumentInput):
    library = db.get_library(library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")

    document = library.documents.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Normalize and merge metadata
    meta_dict = (
        updated_doc_input.metadata.model_dump()
        if updated_doc_input.metadata
        else {}
    )
    meta_dict.setdefault("created_at", document.metadata.created_at if document.metadata else now_iso())

    updated_doc = Document(
        id=document_id,
        title=updated_doc_input.title,
        library_id=document.library_id,
        chunk_ids=document.chunk_ids,
        metadata=DocumentMetadata(**meta_dict),
    )

    library.documents[document_id] = updated_doc
    db.update_library(library)

    return updated_doc

@router.delete("/{document_id}")
def delete_document(library_id: str, document_id: str):
    library = db.get_library(library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")

    document = library.documents.pop(document_id, None)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove associated chunks
    for chunk_id in document.chunk_ids:
        library.chunk_map.pop(chunk_id, None)

    # Rebuild index
    indexing_service = db.get_indexing_service(library_id)
    if indexing_service:
        indexing_service.rebuild_index(library.chunk_map)

    db.update_library(library)
    return {"detail": f"Document {document_id} and its chunks deleted"}