"""
FastAPI server for the RAG system.
Provides REST API endpoints for querying and document management.
"""

import os
import json
import secrets
from contextlib import asynccontextmanager
from typing import Optional, Any, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import uvicorn


# API Key configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_api_keys() -> set:
    """Load API keys from environment variable or file."""
    # First check environment variable (comma-separated keys)
    env_keys = os.getenv("RAG_API_KEYS", "")
    if env_keys:
        return set(key.strip() for key in env_keys.split(",") if key.strip())

    # Fall back to file-based keys
    keys_file = os.path.join(os.path.dirname(__file__), ".api_keys")
    if os.path.exists(keys_file):
        with open(keys_file, "r") as f:
            return set(line.strip() for line in f if line.strip() and not line.startswith("#"))

    return set()


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify the API key from request header."""
    if api_key is None:
        raise HTTPException(status_code=401, detail="API key required")

    valid_keys = get_api_keys()
    if not valid_keys:
        # If no keys configured, allow access (development mode)
        return api_key

    if api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return api_key


from rag_pipeline import RAGPipeline


# Global RAG pipeline instance
rag: Optional[RAGPipeline] = None


# Request/Response Models
class QueryRequest(BaseModel):
    question: str
    num_docs: int = 6
    max_new_tokens: int = 2048  # Increased for formatted responses
    use_history: bool = True  # Whether to use conversation history


class QueryResponse(BaseModel):
    answer: str
    sources: list
    num_docs_retrieved: int
    history_used: int = 0  # Number of previous exchanges used
    detected_followup: bool = False  # Whether this was detected as a follow-up question


class GetDataRequest(BaseModel):
    prompt: str = Field(min_length=1)
    schema: Dict[str, Any]
    system_prompt: Optional[str] = None
    num_docs: int = 6
    max_new_tokens: int = 2048
    use_history: bool = True


class GetDataResponse(BaseModel):
    data: Dict[str, Any]


class HistoryResponse(BaseModel):
    history: list
    count: int


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


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("LLM returned empty response")

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    if raw.startswith("```"):
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start = raw.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(raw)):
            ch = raw[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw[start:i + 1]
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed

    raise ValueError("LLM did not return valid JSON object")


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
async def query(request: QueryRequest, api_key: str = Depends(verify_api_key)):
    """
    Query the RAG system with a question.

    - **question**: The question to ask
    - **num_docs**: Number of documents to retrieve (default: 6)
    - **max_new_tokens**: Maximum tokens in response (default: 1024)
    - **use_history**: Whether to use conversation history (default: True)

    The system remembers previous conversations and learns from interactions.
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        result = rag.query(
            question=request.question,
            num_docs=request.num_docs,
            max_new_tokens=request.max_new_tokens,
            use_history=request.use_history
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get-data", response_model=GetDataResponse)
async def get_data(request: GetDataRequest, api_key: str = Depends(verify_api_key)):
    """
    Get strict structured JSON data from the RAG/LLM system.

    - **prompt**: The user's request
    - **schema**: Target JSON schema shape to generate
    - **system_prompt**: Optional system instruction
    - **num_docs**: Number of documents to retrieve
    - **max_new_tokens**: Maximum tokens in response
    - **use_history**: Whether to use conversation history
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    system_prompt = request.system_prompt or (
        "You are a structured JSON generator. "
        "Return strict JSON only. "
        "Do not return markdown, bullets, or explanations."
    )

    final_prompt = (
        f"{system_prompt}\n\n"
        f"User request:\n{request.prompt}\n\n"
        f"Return JSON matching this schema shape exactly:\n"
        f"{json.dumps(request.schema, indent=2)}"
    )

    try:
        result = rag.query(
            question=final_prompt,
            num_docs=request.num_docs,
            max_new_tokens=request.max_new_tokens,
            use_history=request.use_history
        )
        answer = (result or {}).get("answer", "")
        data = _extract_json_object(answer)
        return GetDataResponse(data=data)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    metadata: str = Form(default="{}"),
    api_key: str = Depends(verify_api_key)
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
async def add_text_document(request: DocumentRequest, api_key: str = Depends(verify_api_key)):
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
async def reindex(api_key: str = Depends(verify_api_key)):
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
async def get_stats(api_key: str = Depends(verify_api_key)):
    """Get statistics about the RAG system."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        stats = rag.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history", response_model=HistoryResponse)
async def get_history(api_key: str = Depends(verify_api_key)):
    """
    Get the current conversation history.

    Returns the list of previous Q&A exchanges in the current session.
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        history = rag.get_history()
        return HistoryResponse(history=history, count=len(history))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/history/clear", response_model=StatusResponse)
async def clear_history(api_key: str = Depends(verify_api_key)):
    """
    Clear the conversation history.

    Use this to start a fresh conversation without previous context.
    """
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        result = rag.clear_history()
        return StatusResponse(**result)
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
