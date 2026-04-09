"""
Local embedding model setup using sentence-transformers.
Runs entirely on GPU - no external API calls.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embeddings():
    """
    Returns a HuggingFace embedding model that runs locally on GPU.
    Uses all-MiniLM-L6-v2 which is fast and effective for RAG.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )
