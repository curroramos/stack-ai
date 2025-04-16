from typing import List, Tuple
from app.models.chunk_models import Chunk
from .base import Indexer

class IndexingService:
    def __init__(self, strategy: Indexer):
        self.strategy = strategy

    def add_chunk(self, chunk: Chunk):
        self.strategy.add_vector(chunk.embedding, chunk.id)

    # def remove_chunk(self, chunk_id: str):
    #     self.strategy.remove_vector(chunk_id)

    def rebuild_index(self, chunk_map: dict):
        self.strategy.rebuild(chunk_map)

    def search_chunks(self, query_embedding: List[float], k: int) -> List[Tuple[str, float]]:
        return self.strategy.search(query_embedding, k)
