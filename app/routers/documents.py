from fastapi import APIRouter, HTTPException
from uuid import uuid4
from typing import List
from datetime import datetime, timezone
from app.core.db import db
from app.models import Document, DocumentMetadata

router = APIRouter(prefix="/libraries/{library_id}/documents", tags=["documents"])

def now_iso():
    return datetime.now(timezone.utc).isoformat()

@router.post("/", response_model=Document)
def create_document(library_id: str, document: Document):
    library = db.get_library(library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")

    # Normalize metadata
    if isinstance(document.metadata, dict):
        meta_dict = document.metadata
    elif document.metadata:
        meta_dict = document.metadata.model_dump()
    else:
        meta_dict = {}

    meta_dict.setdefault("created_at", now_iso())
    document.metadata = DocumentMetadata(**meta_dict)

    document.id = str(uuid4())
    library.documents.append(document)
    db.update_library(library)

    return document

@router.get("/")
def list_documents(library_id: str) -> List[Document]:
    library = db.get_library(library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    return library.documents

@router.get("/{document_id}")
def get_document(library_id: str, document_id: str):
    library = db.get_library(library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    for doc in library.documents:
        if doc.id == document_id:
            return doc
    raise HTTPException(status_code=404, detail="Document not found")
