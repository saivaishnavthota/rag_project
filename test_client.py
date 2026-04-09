"""
Test client for the RAG API.
Run this after starting the server to test the endpoints.
"""

import requests

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())
    return response.status_code == 200


def test_query(question: str):
    """Test query endpoint."""
    response = requests.post(
        f"{BASE_URL}/query",
        json={"question": question}
    )
    result = response.json()
    print(f"\nQuestion: {question}")
    print(f"Answer: {result.get('answer', 'Error')}")
    print(f"Sources: {result.get('num_docs_retrieved', 0)} documents retrieved")
    return result


def test_add_text(content: str, metadata: dict = None):
    """Test adding text document."""
    response = requests.post(
        f"{BASE_URL}/add-text",
        json={
            "content": content,
            "metadata": metadata or {}
        }
    )
    print("\nAdd Text Result:", response.json())
    return response.json()


def test_upload_file(filepath: str):
    """Test file upload."""
    with open(filepath, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/upload",
            files={"file": f}
        )
    print("\nUpload Result:", response.json())
    return response.json()


def test_stats():
    """Test stats endpoint."""
    response = requests.get(f"{BASE_URL}/stats")
    print("\nStats:", response.json())
    return response.json()


def test_reindex():
    """Test reindex endpoint."""
    response = requests.post(f"{BASE_URL}/reindex")
    print("\nReindex Result:", response.json())
    return response.json()


if __name__ == "__main__":
    print("=" * 50)
    print("RAG API Test Client")
    print("=" * 50)

    # Test health
    print("\n1. Testing health endpoint...")
    test_health()

    # Test stats
    print("\n2. Testing stats endpoint...")
    test_stats()

    # Test query
    print("\n3. Testing query endpoint...")
    test_query("What is this RAG system about?")

    # Test adding text
    print("\n4. Testing add-text endpoint...")
    test_add_text(
        content="This is a test document added via API. It contains information about testing.",
        metadata={"source": "api_test", "type": "test"}
    )

    # Test query again
    print("\n5. Testing query after adding document...")
    test_query("What was added via API?")

    print("\n" + "=" * 50)
    print("Tests completed!")
    print("=" * 50)
