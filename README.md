##  Project Overview

This project provides a **REST API** for managing a custom in-memory **vector database** with semantic search capabilities.

Users can:
- **Create, read, update, and delete** libraries
- **Add and manage documents** within each library
- **Add, update, and remove chunks** from documents

Chunks (text + metadata) are automatically vectorized and stored. Each library maintains its own index (e.g., **Linear** or **Clustered**) over its chunks, enabling **k-nearest neighbor (k-NN)** search for semantically similar content.


## Technical Architecture & Choices

### Data Model

![Database Architecture](assets/db.png)

#### 1. **Chunk**
- **Definition**: A `Chunk` is the atomic unit of text + vector embedding + metadata.
- **Model**:
  ```python
  class Chunk(BaseModel):
      id: str = Field(default_factory=lambda: str(uuid4()))
      text: str
      document_id: str
      embedding: List[float]
      metadata: Optional[ChunkMetadata] = None

  class ChunkInput(BaseModel):
      text: str
      metadata: Optional[ChunkMetadata] = None
  ```

#### 2. **Document**
- **Definition**: A `Document` is a logical grouping of chunks.
- **Model**:
  ```python
  class Document(BaseModel):
      id: str = Field(default_factory=lambda: str(uuid4()))
      title: str
      library_id: str
      chunk_ids: List[str] = Field(default_factory=list)
      metadata: Optional[DocumentMetadata] = None

  class DocumentInput(BaseModel):
      title: str
      metadata: Optional[DocumentMetadata] = None
  ```
-  By storing only `chunk_ids`, the document remains lightweight and allows modular access to underlying chunks stored in the library.

#### 3. **Library**
- **Definition**: A `Library` is a top-level collection of documents + centralized chunk storage + index.
- **Model**:
  ```python
  class Library(BaseModel):
      id: str
      name: str
      documents: Dict[str, Document]
      chunk_map: Dict[str, Chunk]
      index: IndexType
      metadata: Optional[LibraryMetadata]

  class IndexType(str, Enum):
    LINEAR = "linear"
    CLUSTERED = "clustered"
  ```

- Provides fast lookup (`chunk_map`)
- Maintains consistency through encapsulated `add/update/remove` operations
- Supports indexing via `LinearIndex` and enables k-NN search



###  Fixed Schemas

I followed the task's suggestion and used fixed schemas for metadata to keep things simple and robust.

Here are the fixed metadata schemas:

```python
class ChunkMetadata(BaseModel):
    source: str
    created_at: str
    author: str
    language: str

class DocumentMetadata(BaseModel):
    category: str
    created_at: str
    source_type: str
    tags: List[str]

class LibraryMetadata(BaseModel):
    created_by: str
    created_at: str
    use_case: str
    access_level: Literal["private", "public", "restricted"] = "private"

```



### Vector Indexing

#### 1. **Linear Index (Brute-Force Search)**

- **Time Complexity:**
  - **Insert:** `O(1)` per chunk  
  - **Search:**  
    - Distance computations: `O(n·d)`  
    - Full sort: `O(n log n)`  
    - **Total:** `O(n·d + n log n)`
- **Space Complexity:** `O(n·d)` (n = number of vectors, d = embedding dimension)
- **Use Case:** Small-to-medium datasets; deterministic and reliable.
- **Tradeoffs:**
  - Simple, no preprocessing
  - Doesn’t scale — slow search on large datasets due to brute-force scan and sort


```python
class LinearIndex:
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
```

#### 2. **Clustered Index (Flat Clustering, e.g., k-means-lite)**

- **Time Complexity**:
  - Insert: `O(1)` (after initial clustering)
  - Search: `O(p·m)` per query (p = probed clusters, m = avg vectors per cluster)
  - Rebuild: `O(n·k)` (n = total vectors, k = number of clusters)
- **Space Complexity**: `O(n·d + k·d)`
- **Use Case**: Faster search on medium-to-large datasets
- **Tradeoffs**:
  - Faster than brute-force for large n
  - Greedy cluster assignment (no centroid updates)
  - Requires manual rebuild after updates/deletes

```python
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

```


#####  Notes

- **LinearIndex** is the baseline — robust, no assumptions.
- **ClusteredIndex** improves query speed at the cost of accuracy and added complexity.
- Both indexes will be rebuildt when the entire dataset is refreshed or updated

### Concurrency & Data Consistency

- `RLock` ensures thread-safe access to in-memory data and indexing operations.

- All CRUD actions are wrapped to avoid data races and ensure consistent state.

- Index updates and persistence are atomic to prevent partial writes.

- Each library manages its own index, avoiding shared mutable state.

### Persistence Layer

- The system persists data to a local JSON file (data/db.json) using model_dump() from Pydantic.

- Data is saved after every CRUD operation to ensure consistency across restarts.

- On startup, the database is restored and indexes are rebuilt.


##  API Overview
You can explore and test the API using this [Postman Collection](https://www.postman.com/curroramos/stack-ai/collection/up69kv0/stack-ai-vector-db?action=share&creator=37688986)

###  CRUD Libraries
####  `/libraries`
- `POST /libraries/` – Create a new library. Optional query parameter `index_type` sets the indexing strategy (e.g., LINEAR).
- `GET /libraries/` – List all libraries.
- `GET /libraries/{library_id}` – Retrieve a specific library by ID.
- `PUT /libraries/{library_id}` – Update an existing library's name and metadata.
- `DELETE /libraries/{library_id}` – Delete a library by ID.

### CRUD Documents
#### `/libraries/{library_id}/documents` 
- `POST /libraries/{library_id}/documents/` – Create a new document within the specified library.
- `GET /libraries/{library_id}/documents/` – List all documents in a library.
- `GET /libraries/{library_id}/documents/{document_id}` – Retrieve a specific document by ID.
- `PUT /libraries/{library_id}/documents/{document_id}` – Update an existing document’s content and metadata.
- `DELETE /libraries/{library_id}/documents/{document_id}` – Delete a document and its associated chunks.

### CRUD Chunks
#### `/libraries/{library_id}/documents/{document_id}/chunks` 
- `POST /libraries/{library_id}/documents/{document_id}/chunks/` – Add a new chunk to the specified document.
- `GET /libraries/{library_id}/documents/{document_id}/chunks/` – List all chunks in a document.
- `GET /libraries/{library_id}/documents/{document_id}/chunks/{chunk_id}` – Retrieve a specific chunk by ID.
- `PUT /libraries/{library_id}/documents/{document_id}/chunks/{chunk_id}` – Update an existing chunk.
- `DELETE /libraries/{library_id}/documents/{document_id}/chunks/{chunk_id}` – Delete a chunk from a document.


### kNN Search
#### `/query` 
- `POST /query` – Perform a k-nearest neighbor search in a specified library.
  - Requires: `library_id`, `query_text`, `k` (number of neighbors), and optional `distance_metric` (e.g., cosine or euclidean).


## Testing

- **Test Files**:  
  - `tests/test_main.py` – covers CRUD operations for libraries, documents, and chunks.  
  - `tests/test_query.py` – tests vector indexing and query behavior (add, delete, search).  
- **Framework**: [pytest](https://docs.pytest.org/)  
- **Run tests**:  
  ```bash
  pytest -v
  ```

Optionally, set `log_cli = true` to see test logs
```ini
[pytest]
log_cli = false
log_cli_level = INFO
log_cli_format = %(asctime)s [%(levelname)s] %(name)s: %(message)s
log_cli_date_format = %Y-%m-%d %H:%M:%S
```

##  Optional Enhancements

- Data persistence


## 🚀 Getting Started

Instructions:
```bash
# Clone repo
git clone https://github.com/curroramos/stack-ai
cd stack-ai

# Install dependencies
pip install -r requirements.txt

# Run app
uvicorn app.main:app --reload

# Run tests
pytest -v
```

Create .env file in root folder and add
```python
COHERE_API_KEY=your_key
```

## 🐳 Docker & Kubernetes

This project is containerized and can be deployed using [Helm](https://helm.sh) on a local Kubernetes cluster powered by [Minikube](https://minikube.sigs.k8s.io/).

### 🛠 Prerequisites

Make sure the following tools are installed on your system:

- [Docker](https://www.docker.com/)
- [Minikube](https://minikube.sigs.k8s.io/)
- [Helm](https://helm.sh/)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)


### 📦 1. Start Minikube

```bash
minikube start
```

(Optional) Launch the Kubernetes dashboard:

```bash
minikube dashboard
```

### 🐋 2. Build and Push Docker Image

Build and push the image to Docker Hub (already done):

```bash
docker build -t stack-ai-vector-db:latest .
docker tag stack-ai-vector-db:latest franciscoramos3010/stack-ai-vector-db:latest
docker push franciscoramos3010/stack-ai-vector-db:latest
```

>  Replace `franciscoramos3010` with your Docker Hub username if needed.


### 📦 3. Helm Chart Structure

Ensure the Helm chart directory is structured as follows:

```
charts/
└── stack-ai-vector-db/
    ├── Chart.yaml
    ├── values.yaml
    └── templates/
        ├── deployment.yaml
        └── service.yaml
```


### 4. Update `values.yaml`

In `charts/stack-ai-vector-db/values.yaml`, set:

```yaml
image:
  repository: franciscoramos3010/stack-ai-vector-db
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: NodePort  # or ClusterIP if you're using Ingress instead
  port: 8000
```


### 📦 5. Deploy with Helm

From the root of your project, install the chart:

```bash
helm install stack-ai-vector-db charts/stack-ai-vector-db
```

To apply updates:

```bash
helm upgrade stack-ai-vector-db charts/stack-ai-vector-db
```

### 6. Access the API

To access the API:

#### Option A: Use `minikube service` (for local clusters)

```bash
minikube service stack-ai-vector-db
```

This opens your browser to:

```
http://<minikube-ip>:<node-port>/docs
```

#### Option B: Get NodePort manually

```bash
kubectl get svc stack-ai-vector-db
```

Then open:
```
http://<minikube-ip>:<node-port>/docs
```


### 7. Verify Deployment

Check if everything's running:

```bash
kubectl get pods
kubectl logs <your-pod-name>
```


## Future Improvements

- Better indexing algorithms.
- Real database integration.
- Full-text chunking + embedding pipeline.
- Real user auth and role management for library access
