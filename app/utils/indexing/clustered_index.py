import math
from typing import List, Tuple, Callable, Dict

from app.models import Chunk
from app.utils.similarity import euclidean_distance

class ClusteredIndex:
    def __init__(self, num_clusters: int = 8, distance_fn: Callable[[List[float], List[float]], float] = euclidean_distance):
        self.num_clusters = num_clusters
        self.distance_fn = distance_fn
        self.centroids: List[List[float]] = []
        self.clusters: List[Dict[str, List[float]]] = []

    def _closest_centroid_idx(self, vector: List[float]) -> int:
        if not self.centroids:
            return 0  # fallback
        dists = [self.distance_fn(vector, c) for c in self.centroids]
        return dists.index(min(dists))

    def _init_cluster(self, vector: List[float]):
        self.centroids.append(vector)
        self.clusters.append({})

    def add_vector(self, vector: List[float], chunk_id: str):
        if len(self.centroids) < self.num_clusters:
            self._init_cluster(vector)
            self.clusters[-1][chunk_id] = vector
            return

        idx = self._closest_centroid_idx(vector)
        self.clusters[idx][chunk_id] = vector

    def remove_vector(self, chunk_id: str):
        for cluster in self.clusters:
            if chunk_id in cluster:
                del cluster[chunk_id]
                return

    def rebuild(self, chunk_map: Dict[str, Chunk]):
        self.centroids.clear()
        self.clusters.clear()
        for chunk_id, chunk in chunk_map.items():
            self.add_vector(chunk.embedding, chunk_id)

    def search(self, query: List[float], k: int, probe_clusters: int = 2) -> List[Tuple[str, float]]:
        cluster_dists = [(i, self.distance_fn(query, c)) for i, c in enumerate(self.centroids)]
        cluster_dists.sort(key=lambda x: x[1])
        top_clusters = [i for i, _ in cluster_dists[:probe_clusters]]

        candidates = []
        for idx in top_clusters:
            for cid, vec in self.clusters[idx].items():
                dist = self.distance_fn(query, vec)
                candidates.append((cid, dist))

        if len(candidates) < k:
            # fallback: brute force across all clusters
            for idx in range(len(self.clusters)):
                if idx in top_clusters:
                    continue
                for cid, vec in self.clusters[idx].items():
                    dist = self.distance_fn(query, vec)
                    candidates.append((cid, dist))

        candidates.sort(key=lambda x: x[1])
        return candidates[:k]
