from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal
from app.core.db import db
from app.utils.embeddings import get_embedding
from app.utils.similarity import cosine_similarity, euclidean_distance
from app.models import QueryRequest, QueryResult

router = APIRouter()

@router.post("/query", response_model=List[QueryResult])
def search_library(req: QueryRequest):
    library = db.get_library(req.library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")

    indexing_service = db.get_indexing_service(req.library_id)
    if not indexing_service:
        raise HTTPException(status_code=404, detail="Index not initialized for this library")

    if hasattr(indexing_service.strategy, "distance_fn"):
        if req.distance_metric == "cosine":
            indexing_service.strategy.distance_fn = cosine_similarity
        else:
            indexing_service.strategy.distance_fn = euclidean_distance

    try:
        query_vector = get_embedding(req.query_text)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    results = indexing_service.search_chunks(query_vector, k=req.k)

    return [
        QueryResult(
            chunk_id=cid,
            score=score,
            text=chunk.text,
            metadata=chunk.metadata
        )
        for cid, score in results
        if (chunk := library.chunk_map.get(cid)) is not None
    ]