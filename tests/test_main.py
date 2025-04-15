import pytest
from fastapi.testclient import TestClient
from app.main import app
import uuid

client = TestClient(app)

# Fixtures
@pytest.fixture
def test_library():
    return {
        "name": "Test Library",
        "metadata": {"owner": "tester"}
    }

@pytest.fixture
def test_document():
    return {
        "title": "Test Doc",
        "metadata": {"doc_type": "article"}
    }

@pytest.fixture
def test_chunk_input():
    return {
        "text": "Hello world test chunk",
        "metadata": {"type": "text"}
    }

@pytest.fixture
def updated_library():
    return {
        "name": "Updated Library",
        "metadata": {"owner": "updated"}
    }

@pytest.fixture
def updated_chunk_input():
    return {
        "text": "Updated chunk text",
        "metadata": {"type": "text", "updated": True}
    }

# ------------------------------
# Library Tests
# ------------------------------
def test_create_library(test_library):
    response = client.post("/libraries/", json=test_library)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == test_library["name"]
    assert "id" in data

def test_list_libraries():
    response = client.get("/libraries/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_update_delete_library(test_library, updated_library):
    # Create library
    lib_resp = client.post("/libraries/", json=test_library)
    library_id = lib_resp.json()["id"]

    # Get
    get_resp = client.get(f"/libraries/{library_id}")
    assert get_resp.status_code == 200

    # Update
    put_resp = client.put(f"/libraries/{library_id}", json=updated_library)
    assert put_resp.status_code == 200
    assert put_resp.json()["name"] == updated_library["name"]

    # Delete
    del_resp = client.delete(f"/libraries/{library_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["detail"] == "Deleted"

# ------------------------------
# Chunk Tests
# ------------------------------
def test_crud_chunk(test_library, test_document, test_chunk_input, updated_chunk_input):
    # Create library
    lib_resp = client.post("/libraries/", json=test_library)
    library_id = lib_resp.json()["id"]

    # Add document
    doc_resp = client.post(f"/libraries/{library_id}/documents/", json=test_document)
    document_id = doc_resp.json()["id"]

    # Create chunk
    chunk_resp = client.post(
        f"/libraries/{library_id}/documents/{document_id}/chunks/",
        json=test_chunk_input
    )
    assert chunk_resp.status_code == 200
    chunk_id = chunk_resp.json()["id"]

    # Read all
    list_resp = client.get(f"/libraries/{library_id}/documents/{document_id}/chunks/")
    assert list_resp.status_code == 200
    assert any(c["id"] == chunk_id for c in list_resp.json())

    # Read one
    get_resp = client.get(f"/libraries/{library_id}/documents/{document_id}/chunks/{chunk_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == chunk_id

    # Update
    update_resp = client.put(
        f"/libraries/{library_id}/documents/{document_id}/chunks/{chunk_id}",
        json=updated_chunk_input
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["text"] == updated_chunk_input["text"]

    # Delete
    delete_resp = client.delete(
        f"/libraries/{library_id}/documents/{document_id}/chunks/{chunk_id}"
    )
    assert delete_resp.status_code == 200
    assert delete_resp.json()["detail"] == "Chunk deleted"

