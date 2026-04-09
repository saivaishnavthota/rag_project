"""
FastAPI server for the RAG system.
Provides REST API endpoints for querying and document management.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from rag_pipeline import RAGPipeline


# Global RAG pipeline instance
rag: Optional[RAGPipeline] = None


# Request/Response Models
class QueryRequest(BaseModel):
    question: str
    num_docs: int = 4
    max_new_tokens: int = 512


class QueryResponse(BaseModel):
    answer: str
    sources: list
    num_docs_retrieved: int


class DocumentRequest(BaseModel):
    content: str
    metadata: dict = {}


class StatusResponse(BaseModel):
    status: str
    message: str = ""


class StatsResponse(BaseModel):
    total_chunks: int
    embedding_model: str
    llm_model: str


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global rag
    print("Starting RAG server...")
    print("Loading models and vector store (this may take a few minutes)...")
    rag = RAGPipeline()
    print("Server ready!")
    yield
    # Shutdown
    print("Shutting down RAG server...")


# Create FastAPI app
app = FastAPI(
    title="Local RAG API",
    description="RAG system using Qwen 2.5 7B running locally on GPU",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "running", "message": "RAG API is ready"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system with a question.

    - **question**: The question to ask
    - **num_docs**: Number of documents to retrieve (default: 4)
    - **max_new_tokens**: Maximum tokens in response (default: 512)
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        result = rag.query(
            question=request.question,
            num_docs=request.num_docs,
            max_new_tokens=request.max_new_tokens
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    metadata: str = Form(default="{}")
):
    """
    Upload a document file to add to the knowledge base.

    Supported formats: .txt, .pdf, .docx
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    # Check file extension
    allowed_extensions = {".txt", ".pdf", ".docx", ".md"}
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )

    try:
        content = await file.read()

        # For text-based files, decode directly
        if file_ext in {".txt", ".md"}:
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                text = content.decode("latin-1")

            import json
            meta = json.loads(metadata)
            meta["filename"] = file.filename
            meta["file_type"] = file_ext

            result = rag.add_document(content=text, metadata=meta)

            return {
                "status": "success",
                "filename": file.filename,
                "chunks_added": result["chunks_added"]
            }
        else:
            # For PDF/DOCX, save to documents folder for processing
            docs_dir = os.path.join(os.path.dirname(__file__), "documents")
            file_path = os.path.join(docs_dir, file.filename)

            with open(file_path, "wb") as f:
                f.write(content)

            return {
                "status": "success",
                "filename": file.filename,
                "message": "File saved. Run /reindex to process."
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-text")
async def add_text_document(request: DocumentRequest):
    """
    Add a text document directly via API.

    - **content**: The text content to add
    - **metadata**: Optional metadata dictionary
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        result = rag.add_document(
            content=request.content,
            metadata=request.metadata
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reindex", response_model=StatusResponse)
async def reindex():
    """
    Reindex all documents from the documents directory.
    Use this after adding new files to the documents folder.
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        result = rag.reindex()
        return StatusResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get statistics about the RAG system."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        stats = rag.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
