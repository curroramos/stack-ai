from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal
from app.core.db import db
from app.utils.embeddings import get_embedding
from app.utils.indexing import cosine_similarity, euclidean_distance
from app.models import QueryRequest, QueryResult

router = APIRouter()

@router.post("/query", response_model=List[QueryResult])
def search_library(req: QueryRequest):
    library = db.get_library(req.library_id)
    if not library:
        raise HTTPException(status_code=404, detail="Library not found")

    index = library.index
    if not index:
        raise HTTPException(status_code=404, detail="Index not built for this library")

    if req.distance_metric == "cosine":
        index.distance_fn = cosine_similarity
    else:
        index.distance_fn = euclidean_distance

    try:
        query_vector = get_embedding(req.query_text)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    results = index.search(query_vector, k=req.k)

    # Retrieve chunk content for each result
    return [
        {
            "chunk_id": cid,
            "score": score,
            "chunk_content": library.get_chunk_by_id(cid).text # could be complete chunk
        }
        for cid, score in results
    ]