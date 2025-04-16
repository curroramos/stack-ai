from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timezone
from app.core.db import db
from app.models.library_models import Library, LibraryCreate, LibraryResponse
from app.models.metadata_models import LibraryMetadata
from app.utils.indexing.index_type import IndexType


router = APIRouter(prefix="/libraries", tags=["libraries"])

def now_iso():
    return datetime.now(timezone.utc).isoformat()

@router.post("/", response_model=LibraryResponse)
def create_library(library_data: LibraryCreate, index_type: IndexType = Query(default=IndexType.LINEAR)):
    meta_dict = library_data.metadata.model_dump() if library_data.metadata else {}
    meta_dict["created_at"] = now_iso()
    metadata = LibraryMetadata(**meta_dict)

    library = Library(name=library_data.name, metadata=metadata)
    db.add_library(library, index_type=index_type)

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

    if updated_data.metadata:
        new_meta = updated_data.metadata.model_dump()
        new_meta["created_at"] = library.metadata.created_at if library.metadata and library.metadata.created_at else now_iso()
        library.metadata = LibraryMetadata(**new_meta)

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
