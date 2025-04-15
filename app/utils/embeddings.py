import os
import requests
from typing import List
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_EMBEDDING_URL = "https://api.cohere.ai/v1/embed"

HEADERS = {
    "Authorization": f"Bearer {COHERE_API_KEY}",
    "Content-Type": "application/json"
}

def get_embedding(text: str) -> List[float]:
    if not COHERE_API_KEY:
        raise ValueError("Cohere API key is not set. Please set COHERE_API_KEY in your environment.")

    payload = {
        "model": "embed-english-v3.0",
        "texts": [text],
        "input_type": "search_document"
    }

    try:
        response = requests.post(COHERE_EMBEDDING_URL, headers=HEADERS, json=payload)
        response.raise_for_status()

        response_json = response.json()
        embedding = response_json.get("embeddings", [])[0]
        return embedding

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to get embedding: {e}")
    except (IndexError, KeyError) as e:
        raise RuntimeError(f"Unexpected response structure: {e}")
