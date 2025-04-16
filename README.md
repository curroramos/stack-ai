

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

Define the Chunk, Document and Library classes. To simplify schema definition, we suggest you use a fixed schema for each of the classes. This means not letting the user define which fields should be present within the **metadata for each class**. Following this path you will have fewer problems validating insertions/updates, but feel to let the users define their own schemas for each library if you are up for the challenge.

Implement two or three indexing algorithms, do not use external libraries, we want to see you code them up. What is the space and time complexity for each of the indexes? Why did you choose this index?

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