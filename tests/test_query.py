import pytest
import logging
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from app.main import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
client = TestClient(app)


@pytest.fixture
def fruit_chunks():
    # Proper chunk metadata matching the ChunkMetadata schema
    created_at = datetime.now(timezone.utc).isoformat()
    base_metadata = {
        "created_at": created_at,
        "author": "index-test",
        "language": "en",
        "source": "query-fixture"
    }
    texts = [
        "banana fruit salad",
        "apple",
        "mathematics",
        "car repair manual",
        "quantum physics introduction", 
        "orange", 
        "watermelon"
    ]
    return [{"text": text, "metadata": base_metadata} for text in texts]


@pytest.fixture
def test_library():
    return {
        "name": "Live Index Library",
        "metadata": {
            "created_by": "index-tester",
            "created_at": "2023-04-01T12:00:00Z",
            "use_case": "vector-search",
            "access_level": "private"
        }
    }

@pytest.fixture
def test_document():
    return {
        "title": "Live Index Doc",
        "metadata": {
            "category": "indexing",
            "created_at": "2023-04-01T12:30:00Z",
            "source_type": "manual",
            "tags": ["ml", "search"]
        }
    }

def test_chunk_index_add_delete_and_query(test_library, test_document, fruit_chunks):
    logger.info("Creating library")
    lib_resp = client.post("/libraries/", json=test_library)
    assert lib_resp.status_code == 200, f"Failed to create library: {lib_resp.text}"
    library_id = lib_resp.json()["id"]

    logger.info("Creating document")
    doc_resp = client.post(f"/libraries/{library_id}/documents/", json=test_document)
    assert doc_resp.status_code == 200, f"Failed to create document: {doc_resp.text}"
    document_id = doc_resp.json()["id"]

    chunk_ids = []
    for chunk in fruit_chunks:
        logger.info(f"Adding chunk: {chunk['text']}")
        resp = client.post(f"/libraries/{library_id}/documents/{document_id}/chunks/", json=chunk)
        assert resp.status_code == 200, f"Failed to add chunk '{chunk['text']}': {resp.text}"
        chunk_ids.append(resp.json()["id"])

    # Perform a search
    query_payload = {
        "library_id": library_id,
        "query_text": "fruit",
        "k": 3,
        "distance_metric": "euclidean"
    }
    query_resp = client.post("/query", json=query_payload)
    assert query_resp.status_code == 200, f"Query failed: {query_resp.text}"
    results_before_delete = query_resp.json()
    returned_ids_before = [res["chunk_id"] for res in results_before_delete]

    # Delete top hit
    chunk_to_delete = returned_ids_before[0]
    logger.info(f"Deleting chunk: {chunk_to_delete}")
    del_resp = client.delete(f"/libraries/{library_id}/documents/{document_id}/chunks/{chunk_to_delete}")
    assert del_resp.status_code == 200, f"Delete failed: {del_resp.text}"

    # Perform the query again
    query_resp_after = client.post("/query", json=query_payload)
    assert query_resp_after.status_code == 200, f"Query after delete failed: {query_resp_after.text}"
    results_after_delete = query_resp_after.json()
    returned_ids_after = [res["chunk_id"] for res in results_after_delete]

    assert chunk_to_delete not in returned_ids_after, "Deleted chunk still appears in query results"
