from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from app.models.chunk_models import Chunk

class Indexer(ABC):
    @abstractmethod
    def add_vector(self, vector: List[float], chunk_id: str):
        pass

    @abstractmethod
    def remove_vector(self, chunk_id: str):
        pass

    @abstractmethod
    def rebuild(self, chunk_map: Dict[str, Chunk]):
        pass

    @abstractmethod
    def search(self, query: List[float], k: int) -> List[Tuple[str, float]]:
        pass
