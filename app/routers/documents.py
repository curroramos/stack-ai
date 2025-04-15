from fastapi import APIRouter, HTTPException
from app.models.base import Document
from app.core.db import db
from uuid import uuid4
from typing import List

router = APIRouter(prefix="/libraries/{library_id}/documents", tags=["documents"])

@router.post("/")
def create_document(library_id: str, document: Document):
    library = db.get_library(library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
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
