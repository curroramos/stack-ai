import os
import json
from pathlib import Path
from app.models.library_models import Library

PERSIST_PATH = Path("data/db.json")

class InMemoryDB:
    def __init__(self):
        self._libraries: Dict[str, Library] = {}
        self._indexing_services: Dict[str, IndexingService] = {}
        self._lock = RLock()
        self._load_from_disk()

    def _save_to_disk(self):
        with self._lock:
            PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(PERSIST_PATH, "w") as f:
                json.dump({lid: lib.model_dump() for lid, lib in self._libraries.items()}, f, indent=2)

    def _load_from_disk(self):
        if not PERSIST_PATH.exists():
            return
        with open(PERSIST_PATH, "r") as f:
            raw = json.load(f)
        for lid, lib_data in raw.items():
            library = Library(**lib_data)
            self._libraries[lid] = library
            strategy = create_index_by_type(IndexType.LINEAR)  # or store index_type in metadata
            indexing_service = IndexingService(strategy)
            indexing_service.rebuild_index(library.chunk_map)
            self._indexing_services[lid] = indexing_service

    def _persist_and_rebuild(self, library: Library):
        self._libraries[str(library.id)] = library
        indexing_service = self._indexing_services.get(str(library.id))
        if indexing_service:
            indexing_service.rebuild_index(library.chunk_map)
        self._save_to_disk()

    # Use _persist_and_rebuild instead of duplicating logic in `update_library`
    def add_library(self, library: Library, index_type: IndexType = IndexType.LINEAR):
        with self._lock:
            self._libraries[str(library.id)] = library
            strategy = create_index_by_type(index_type)
            indexing_service = IndexingService(strategy)
            indexing_service.rebuild_index(library.chunk_map)
            self._indexing_services[str(library.id)] = indexing_service
            self._save_to_disk()

    def update_library(self, library: Library):
        with self._lock:
            self._persist_and_rebuild(library)

    def delete_library(self, library_id: str):
        with self._lock:
            self._libraries.pop(library_id, None)
            self._indexing_services.pop(library_id, None)
            self._save_to_disk()
