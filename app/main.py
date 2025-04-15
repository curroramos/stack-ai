from fastapi import FastAPI
from app.routers import libraries, documents, chunks, query

app = FastAPI()
app.include_router(libraries.router)
app.include_router(documents.router)
app.include_router(chunks.router)
app.include_router(query.router)
