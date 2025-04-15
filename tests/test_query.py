import pytest
import logging
from fastapi.testclient import TestClient
from app.main import app

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.fixture
def test_library():
    return {
        "name": "Live Index Library",
        "metadata": {"env": "test"}
    }

@pytest.fixture
def test_document():
    return {
        "title": "Live Index Doc",
        "metadata": {"cat": "test"}
    }

def test_chunk_index_add_delete_and_query(test_library, test_document):
    logger.info("Creating library")
    lib_resp = client.post("/libraries/", json=test_library)
    assert lib_resp.status_code == 200, f"Failed to create library: {lib_resp.text}"
    library_id = lib_resp.json()["id"]
    logger.info(f"Library created successfully with id: {library_id}")

    logger.info("Creating document")
    doc_resp = client.post(f"/libraries/{library_id}/documents/", json=test_document)
    assert doc_resp.status_code == 200, f"Failed to create document: {doc_resp.text}"
    document_id = doc_resp.json()["id"]
    logger.info(f"Document created successfully with id: {document_id}")

    # Add multiple chunks
    chunk_texts = [
        "banana fruit salad",
        "apple pie recipe",
        "formula mathematics",
        "car repair manual",
        "quantum physics introduction"
    ]
    chunk_ids = []
    for text in chunk_texts:
        logger.info(f"Adding chunk: {text}")
        chunk_input = {
            "text": text,
            "metadata": {"topic": "mixed"}
        }
        resp = client.post(f"/libraries/{library_id}/documents/{document_id}/chunks/", json=chunk_input)
        assert resp.status_code == 200, f"Failed to add chunk '{text}': {resp.text}"
        chunk_id = resp.json()["id"]
        chunk_ids.append(chunk_id)
        logger.info(f"Chunk '{text}' added successfully with id: {chunk_id}")

    # Perform a search for something fruit-related
    query_payload = {
        "library_id": library_id,
        "query_text": "fruit smoothie",
        "k": 3,
        "distance_metric": "cosine"
    }
    logger.info(f"Performing query: {query_payload}")
    query_resp = client.post("/query", json=query_payload)
    assert query_resp.status_code == 200, f"Query failed: {query_resp.text}"
    results_before_delete = query_resp.json()
    logger.info(f"Query results before deletion: {results_before_delete}")

    returned_ids_before = [res["chunk_id"] for res in results_before_delete]

    # Delete one of the top chunks (likely fruit-related)
    chunk_to_delete = returned_ids_before[0]
    logger.info(f"Deleting chunk with id: {chunk_to_delete}")
    delete_resp = client.delete(
        f"/libraries/{library_id}/documents/{document_id}/chunks/{chunk_to_delete}"
    )
    assert delete_resp.status_code == 200, f"Failed to delete chunk: {delete_resp.text}"
    logger.info(f"Chunk with id {chunk_to_delete} deleted successfully")

    # Search again — deleted chunk should no longer be present
    logger.info("Performing query after deletion")
    query_resp_after = client.post("/query", json=query_payload)
    assert query_resp_after.status_code == 200, f"Query after delete failed: {query_resp_after.text}"
    results_after_delete = query_resp_after.json()
    logger.info(f"Query results after deletion: {results_after_delete}")

    returned_ids_after = [res["chunk_id"] for res in results_after_delete]
    assert chunk_to_delete not in returned_ids_after, "Deleted chunk still appears in search results"
