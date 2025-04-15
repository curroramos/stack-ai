import math
from typing import List, Tuple, Optional, Callable, Dict

def euclidean_distance(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x**2 for x in a))
    norm_b = math.sqrt(sum(y**2 for y in b))
    return dot / (norm_a * norm_b)

class LinearIndex:
    def __init__(self, distance_fn: Callable[[List[float], List[float]], float] = euclidean_distance):
        self.vectors: Dict[str, List[float]] = {}  # chunk_id -> vector
        self.distance_fn = distance_fn

    def add_vector(self, vector: List[float], chunk_id: str):
        self.vectors[chunk_id] = vector  # Overwrite if chunk_id exists

    def remove_vector(self, chunk_id: str):
        self.vectors.pop(chunk_id, None)

    def search(self, query: List[float], k: int) -> List[Tuple[str, float]]:
        distances = []
        for cid, vec in self.vectors.items():
            dist = self.distance_fn(query, vec)
            distances.append((cid, dist))
        distances.sort(key=lambda x: x[1])
        return distances[:k]

class KDNode:
    def __init__(self, point: List[float], chunk_id: str, depth: int = 0,
                 left: Optional['KDNode'] = None, right: Optional['KDNode'] = None):
        self.point = point
        self.chunk_id = chunk_id
        self.left = left
        self.right = right
        self.depth = depth

class KDTreeIndex:
    def __init__(self, distance_fn: Callable[[List[float], List[float]], float] = euclidean_distance):
        self.root = None
        self.k = None  # dimensionality
        self.distance_fn = distance_fn
        self.chunk_ids = set()  # Track existing IDs to prevent duplicates

    def add_vector(self, vector: List[float], chunk_id: str):
        if chunk_id in self.chunk_ids:
            return  # Skip duplicate insert for simplicity
        if self.k is None:
            self.k = len(vector)
        self.root = self._insert(self.root, vector, chunk_id, depth=0)
        self.chunk_ids.add(chunk_id)

    def _insert(self, node: Optional[KDNode], point: List[float], chunk_id: str, depth: int) -> KDNode:
        if node is None:
            return KDNode(point, chunk_id, depth)

        axis = depth % self.k
        if point[axis] < node.point[axis]:
            node.left = self._insert(node.left, point, chunk_id, depth + 1)
        else:
            node.right = self._insert(node.right, point, chunk_id, depth + 1)
        return node

    def search(self, query: List[float], k: int) -> List[Tuple[str, float]]:
        best = []  # list of (distance, chunk_id)

        def _search(node: Optional[KDNode], depth: int):
            if node is None:
                return

            dist = self.distance_fn(query, node.point)
            if len(best) < k:
                best.append((dist, node.chunk_id))
                best.sort()
            elif dist < best[-1][0]:
                best[-1] = (dist, node.chunk_id)
                best.sort()

            axis = depth % self.k
            next_branch = None
            opposite_branch = None

            if query[axis] < node.point[axis]:
                next_branch = node.left
                opposite_branch = node.right
            else:
                next_branch = node.right
                opposite_branch = node.left

            _search(next_branch, depth + 1)

            if len(best) < k or abs(query[axis] - node.point[axis]) < best[-1][0]:
                _search(opposite_branch, depth + 1)

        _search(self.root, 0)
        return [(cid, dist) for dist, cid in best]