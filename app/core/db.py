from threading import RLock
from typing import Dict
from app.models.base import Library

class InMemoryDB:
    def __init__(self):
        self._libraries: Dict[str, Library] = {}
        self._lock = RLock()

    def add_library(self, library: Library):
        with self._lock:
            self._libraries[str(library.id)] = library

    def get_library(self, library_id: str):
        with self._lock:
            return self._libraries.get(library_id)

    def update_library(self, library: Library):
        with self._lock:
            self._libraries[str(library.id)] = library

    def delete_library(self, library_id: str):
        with self._lock:
            self._libraries.pop(library_id, None)

    def list_libraries(self):
        with self._lock:
            return list(self._libraries.values())

    def get_index(self, library_id: str):
        with self._lock:
            library = self._libraries.get(library_id)
            if not library:
                return None
            return library.index

db = InMemoryDB()
