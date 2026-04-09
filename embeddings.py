"""
Local embedding model setup using sentence-transformers.
Runs locally with automatic device detection.
"""

import torch
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_device():
    """Detect the best available device."""
    if torch.cuda.is_available():
        try:
            # Test if CUDA actually works
            torch.tensor([1.0]).cuda()
            return "cuda"
        except Exception:
            pass
    return "cpu"


def get_embeddings():
    """
    Returns a HuggingFace embedding model that runs locally.
    Uses all-MiniLM-L6-v2 which is fast and effective for RAG.
    """
    device = get_device()
    print(f"Embeddings using device: {device}")

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
