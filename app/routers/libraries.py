from fastapi import APIRouter, HTTPException
from app.models.base import Library, LibraryCreate, LibraryResponse
from app.core.db import db

router = APIRouter(prefix="/libraries", tags=["libraries"])

@router.post("/", response_model=LibraryResponse)
def create_library(library_data: LibraryCreate):
    library = Library(name=library_data.name, metadata=library_data.metadata)
    db.add_library(library)
    return LibraryResponse(
        id=library.id,
        name=library.name,
        documents=library.documents,
        metadata=library.metadata
    )

@router.get("/", response_model=list[LibraryResponse])
def list_libraries():
    libraries = db.list_libraries()
    return [
        LibraryResponse(
            id=lib.id,
            name=lib.name,
            documents=lib.documents,
            metadata=lib.metadata
        ) for lib in libraries
    ]

@router.get("/{library_id}", response_model=LibraryResponse)
def get_library(library_id: str):
    library = db.get_library(library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")

    return LibraryResponse(
        id=library.id,
        name=library.name,
        documents=library.documents,
        metadata=library.metadata
    )

@router.put("/{library_id}", response_model=LibraryResponse)
def update_library(library_id: str, updated_data: LibraryCreate):
    library = db.get_library(library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")

    library.name = updated_data.name
    library.metadata = updated_data.metadata or {}

    db.update_library(library)

    return LibraryResponse(
        id=library.id,
        name=library.name,
        documents=library.documents,
        metadata=library.metadata
    )

@router.delete("/{library_id}")
def delete_library(library_id: str):
    if not db.get_library(library_id):
        raise HTTPException(status_code=404, detail="Library not found")

    db.delete_library(library_id)
    return {"detail": "Deleted"}
