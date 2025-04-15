from fastapi import FastAPI
from app.routers import libraries, documents, chunks, query

app = FastAPI(
    title="Vector DB API",
    version="1.0.0",
    description="API for managing libraries, documents, chunks, and performing vector similarity search."
)

app.include_router(libraries.router, tags=["Libraries"])
app.include_router(documents.router, tags=["Documents"])
app.include_router(chunks.router, tags=["Chunks"])
app.include_router(query.router, tags=["Query"])
