from threading import RLock
from typing import Dict, Optional
from app.models.library_models import Library
from app.utils.indexing.indexing_service import IndexingService
from app.utils.indexing.linear_index import LinearIndex

class InMemoryDB:
    def __init__(self):
        self._libraries: Dict[str, Library] = {}
        self._indexing_services: Dict[str, IndexingService] = {}
        self._lock = RLock()

    def add_library(self, library: Library, indexing_strategy: Optional[IndexingService] = None):
        with self._lock:
            self._libraries[str(library.id)] = library
            # Initialize an indexing service per library
            if indexing_strategy is None:
                indexing_strategy = IndexingService(strategy=LinearIndex())
            indexing_strategy.rebuild_index(library.chunk_map)
            self._indexing_services[str(library.id)] = indexing_strategy

    def get_library(self, library_id: str) -> Optional[Library]:
        with self._lock:
            return self._libraries.get(library_id)

    def update_library(self, library: Library):
        with self._lock:
            self._libraries[str(library.id)] = library
            indexing_service = self._indexing_services.get(str(library.id))
            if indexing_service:
                indexing_service.rebuild_index(library.chunk_map)

    def delete_library(self, library_id: str):
        with self._lock:
            self._libraries.pop(library_id, None)
            self._indexing_services.pop(library_id, None)

    def list_libraries(self):
        with self._lock:
            return list(self._libraries.values())

    def get_indexing_service(self, library_id: str) -> Optional[IndexingService]:
        with self._lock:
            return self._indexing_services.get(library_id)

db = InMemoryDB()
