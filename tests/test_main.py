import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Fixtures
@pytest.fixture
def test_library():
    return {
        "name": "Test Library",
        "metadata": {
            "created_by": "tester",
            "created_at": "2023-04-01T12:00:00Z",
            "use_case": "unit-testing",
            "access_level": "private",
            "index_type": "clustered"
        }
    }

@pytest.fixture
def test_document():
    return {
        "title": "Test Doc",
        "metadata": {
            "category": "research",
            "created_at": "2023-04-01T12:30:00Z",
            "source_type": "manual",
            "tags": ["nlp", "test"]
        }
    }

@pytest.fixture
def test_chunk_input():
    return {
        "text": "Hello world test chunk",
        "metadata": {
            "source": "unit-test",
            "created_at": "2023-04-01T12:45:00Z",
            "author": "tester",
            "language": "en"
        }
    }

@pytest.fixture
def updated_library():
    return {
        "name": "Updated Library",
        "metadata": {
            "created_by": "updated_user",
            "created_at": "2023-04-02T12:00:00Z",
            "use_case": "updated-case",
            "access_level": "restricted"
        }
    }

@pytest.fixture
def updated_chunk_input():
    return {
        "text": "Updated chunk text",
        "metadata": {
            "source": "unit-test-updated",
            "created_at": "2023-04-02T12:15:00Z",
            "author": "updated_tester",
            "language": "en"
        }
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
# Document Tests
# ------------------------------
def test_crud_document(test_library, test_document):
    # Create library
    lib_resp = client.post("/libraries/", json=test_library)
    library_id = lib_resp.json()["id"]

    # Create document
    doc_resp = client.post(f"/libraries/{library_id}/documents/", json=test_document)
    assert doc_resp.status_code == 200
    doc_data = doc_resp.json()
    document_id = doc_data["id"]
    assert doc_data["title"] == test_document["title"]

    # List documents
    list_resp = client.get(f"/libraries/{library_id}/documents/")
    assert list_resp.status_code == 200
    docs = list_resp.json()
    assert any(d["id"] == document_id for d in docs)

    # Get specific document
    get_resp = client.get(f"/libraries/{library_id}/documents/{document_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == document_id

    # Update document
    updated_doc = {
        "id": document_id,
        "title": "Updated Doc Title",
        "metadata": {
            "category": "updated-research",
            "created_at": "2023-04-05T00:00:00Z",
            "source_type": "automated",
            "tags": ["ai", "update"]
        },
        "chunk_ids": []
    }
    put_resp = client.put(f"/libraries/{library_id}/documents/{document_id}", json=updated_doc)
    assert put_resp.status_code == 200
    assert put_resp.json()["title"] == updated_doc["title"]

    # Delete document
    del_resp = client.delete(f"/libraries/{library_id}/documents/{document_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["detail"] == f"Document {document_id} and its chunks deleted"


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

