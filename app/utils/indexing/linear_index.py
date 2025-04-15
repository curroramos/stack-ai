from typing import List, Tuple, Optional, Callable, Dict
from app.models import Chunk
from app.utils.similarity import euclidean_distance, cosine_similarity

class LinearIndex:
    """
    Linear indexing method with:
    - Time complexity: O(n) per query where n is the number of chunks
    - Space complexity: O(n*d) where n is the number of chunks and d is the dimensionality of embeddings
    """
    def __init__(self, distance_fn: Callable[[List[float], List[float]], float] = euclidean_distance):
        self.vectors: Dict[str, List[float]] = {}  # chunk_id -> vector
        self.distance_fn = distance_fn

    def add_vector(self, vector: List[float], chunk_id: str):
        self.vectors[chunk_id] = vector  # Overwrite if chunk_id exists

    def remove_vector(self, chunk_id: str):
        self.vectors.pop(chunk_id, None)

    def rebuild(self, chunk_map: Dict[str, Chunk]):
        self.vectors.clear()
        for chunk_id, chunk in chunk_map.items():
            self.add_vector(chunk.embedding, chunk_id)

    def search(self, query: List[float], k: int) -> List[Tuple[str, float]]:
        distances = []
        for cid, vec in self.vectors.items():
            dist = self.distance_fn(query, vec)
            distances.append((cid, dist))
        distances.sort(key=lambda x: x[1])
        return distances[:k]