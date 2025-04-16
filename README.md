

do i need to have and api CRUD for documents? or just add when a chunk document is new


linear index, vs KD index. complexity


decision: when to index - index incrementally each time a chunk is added/updated
                        - reindex all each time """



Strategy	Task-Valid	Easy	Scalable	Consistent	Comment
index.add_vector() inline	✅	✅	⚠️	✅	Good start
Rebuild index after changes	✅	✅	✅	✅✅	Even safer
Library model owns its index	✅	✅	✅	✅	Cleaner design
Async / queue-based indexing	❌ (overkill)	⚠️	✅✅	✅✅	Out of scope


Library model pydantic index field -> problem. solved using librarycreate and libraryresponse models

in the search results. i want to get the chunk content, not the chunk id

it’s better to keep just the chunk_id in the index, and maintain a separate map of chunk_id → Chunk data. This keeps the index lightweight and provides O(1) lookups for chunk content. A recommended approach is:

    Store the chunk’s embedding in your vector index with the chunk_id as a key.

    Maintain a separate dictionary (e.g. chunk_map) on the Library, keyed by chunk_id, containing the full Chunk.

    When you retrieve top matches from the index, you can quickly get the chunk content by looking up the chunk_id in the chunk_map.


TODO
fix: chunk should come with the document_id. 
fix dictionary document_id: list[Chunks] anywhere in memory


fix 

```py
@router.post("/")
def add_chunk(library_id: str, document_id: str, chunk_input: ChunkInput):
    library = db.get_library(library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")

    document = next((doc for doc in library.documents if str(doc.id) == document_id), None)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    embedding = get_embedding(chunk_input.text)

    new_chunk = Chunk(
        id=str(uuid4()),
        text=chunk_input.text,
        embedding=embedding,
        metadata=chunk_input.metadata or {}
    )

    document.chunks.append(new_chunk) # where is this going?
    library.add_chunk(new_chunk)
    
    db.update_library(library) # update library in memory after applying changes


    return new_chunk
```



Implement the necessary data structures/algorithms to ensure that there are no **data races** between reads and writes to the database. Explain your design choices.





Create the logic to do the CRUD operations on libraries and documents/chunks. Ensure **data consistency and integrity** during these operations.

Implement an API layer on top of that logic to let users interact with the vector database.

Create a **docker image** for the project and a **helmchart** to install it in a kubernetes cluster like minikube. 





Rebuilding the KD-Tree after a deletion (or batch of deletions) is the most reasonable trade-off, it avoids the complexity of maintaining balance and keeps the code easy to reason about.

created_at field in metadata is automated 



No Data Races:

Using RLock in InMemoryDB methods ensures safe concurrent access.

Indexing service updates are also wrapped.

Logical access patterns avoid shared mutable state.





Nice work wrapping up the project! For your **documentation and technical choices section in the README**, you’ll want to clearly communicate what you built, why you made certain decisions, and how someone can understand, run, and evaluate your work. Here's a structured outline you can use:

---

## 📄 Project Overview

Briefly summarize:
- What the project does (index/query documents with a vector DB via a REST API).
- Core components: Libraries → Documents → Chunks with embeddings.
- Technologies used: FastAPI, Pydantic, Docker, Helm, Kubernetes, etc.




---

## Technical Architecture & Choices

### Data Model

![Database Architecture](assets/db.png)

#### 1. **Chunk**
- **Definition**: A `Chunk` is the atomic unit of text + vector embedding + metadata.
- **Model**:
  ```python
  class Chunk(BaseModel):
      id: str
      text: str
      embedding: List[float]
      metadata: Optional[ChunkMetadata]
  ```
- **Why**: This separation ensures that each text segment can be independently indexed and retrieved with rich metadata (e.g. author, language, timestamp).

#### 2. **Document**
- **Definition**: A `Document` is a logical grouping of chunks.
- **Model**:
  ```python
  class Document(BaseModel):
      id: str
      title: str
      chunk_ids: List[str]
      metadata: Optional[DocumentMetadata]
  ```
- **Why**: By storing only `chunk_ids`, the document remains lightweight and allows modular access to underlying chunks stored in the library.

#### 3. **Library**
- **Definition**: A `Library` is a top-level collection of documents + centralized chunk storage + index.
- **Model**:
  ```python
  class Library(BaseModel):
      id: str
      name: str
      documents: Dict[str, Document]
      chunk_map: Dict[str, Chunk]
      index: LinearIndex
      metadata: Optional[LibraryMetadata]
  ```
- **Why**:
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

- **Time Complexity**:
  - Insert: **O(1)** per chunk
  - Search: **O(n)** per query (where *n* = number of chunks)
- **Space Complexity**: **O(n·d)** (where *d* = embedding dimension)
- **Use Case**: Default fallback, simplest and most reliable for small-to-medium datasets.
- **Tradeoffs**:
  - Simple, deterministic, no preprocessing
  - Doesn’t scale well for large datasets (slow search)

```python
class LinearIndex:
    def search(self, query: List[float], k: int) -> List[Tuple[str, float]]:
        distances = []
        for cid, vec in self.vectors.items():
            dist = self.distance_fn(query, vec)
            distances.append((cid, dist))
        distances.sort(key=lambda x: x[1])
        return distances[:k]
```

---

#### 2. **KD-Tree Index**

- **Time Complexity**:
  - Insert: **O(log n)** (average), **O(n)** (worst case if unbalanced)
  - Search: **O(log n)** (best), **O(n)** (worst)
- **Space Complexity**: **O(n)** for the tree structure
- **Use Case**: Spatial partitioning for faster k-NN in lower dimensions.
- **Tradeoffs**:
  - Better than linear search for balanced, low-dimensional data (ideal < 20D)
  - Degrades in high dimensions (curse of dimensionality), no balancing logic
  - Rebuilding needed after bulk updates

```python
class KDTreeIndex:
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
```

---

### Tradeoffs

- **Linear Index**: Easy to implement, test, and debug. Guaranteed to work in all scenarios, regardless of data distribution or dimensionality.
- **KD-Tree Index**: More efficient for low-dimension embeddings with static or moderately changing data. Offers insight into spatial indexing tradeoffs.


In this project, both indexes are fully rebuilt after chunk deletions or updates for consistency and simplicity. While KD-Tree needs rebuilding to stay balanced, the Linear Index technically does not (its structure supports in-place updates efficiently). The unified rebuild strategy was chosen to reduce complexity and keep the indexing logic consistent across implementations.

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



### ADD POSTMAN LIBRARY

---

## 🐳 Docker & Kubernetes

Explain:
- How to build and run the Docker container.
- Helm chart overview: what configs are in `values.yaml`.
- How to deploy in Minikube (step-by-step if needed).

---

## 🧪 Testing

Mention:
- What tests were written (`tests/test_*.py`).
- Framework used (pytest).
- How to run tests: `pytest`

---

##  Optional Enhancements

- data persistence

---

## 🚀 Getting Started

Instructions:
```bash
# Clone repo
git clone <your-repo>
cd <your-repo>

# Install dependencies
pip install -r requirements.txt

# Run app
uvicorn app.main:app --reload

# Run tests
pytest
```

Add any Helm or Docker run instructions here too.


## 📚 Future Improvements

You can list things like:
- Better indexing algorithms.
- Real database integration.
- Full-text chunking + embedding pipeline.
- Real user auth.




## TODO
add error handling