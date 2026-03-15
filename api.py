import warnings
warnings.filterwarnings("ignore")

import os
import uvicorn
import faiss
import numpy as np
import tempfile
from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama as ollama_lib

load_dotenv()

# ── Ollama Client ─────────────────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_client = ollama_lib.Client(host=OLLAMA_HOST)

# ── Rate Limiter Setup ────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── API Key Setup ─────────────────────────────────────────
API_KEY = os.getenv("API_KEY", "samrat-secret-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key. Include X-API-Key header."
        )
    return api_key

# Initialize FastAPI app
app = FastAPI(
    title="Samrat AI API",
    description="Privacy-first local AI assistant API built by Samrat",
    version="1.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class RAGRequest(BaseModel):
    message: str
    session_id: str

class RAGResponse(BaseModel):
    response: str
    session_id: str
    chunks_searched: int
    model: str

# ── Request/Response Models ──────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    model: str

# ── In-memory conversation store ─────────────────────────
conversations = {}

# ── RAG Storage ───────────────────────────────────────────
rag_sessions = {}  # stores {session_id: {chunks, index}}

# ── Routes ───────────────────────────────────────────────

# Public routes — no auth needed
# Health — generous limit
@app.get("/health")
@limiter.limit("60/minute")
def health(request: Request):
    return {"status": "healthy"}

# Chat — strict limit (CPU protection)
@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
def chat(request: Request, chat_request: ChatRequest, api_key: str = Depends(verify_api_key)):
    if chat_request.session_id not in conversations:
        conversations[chat_request.session_id] = []

    conversations[chat_request.session_id].append({
        "role": "user",
        "content": chat_request.message
    })

    messages = [
        {
            "role": "system",
            "content": "You are Samrat AI, a personal AI assistant built by Samrat. Be helpful, concise and intelligent."
        }
    ] + conversations[chat_request.session_id]

    response = ollama_client.chat(
        model="samrat-ai:latest",
        messages=messages
    )

    reply = response["message"]["content"]

    conversations[chat_request.session_id].append({
        "role": "assistant",
        "content": reply
    })

    return ChatResponse(
        response=reply,
        session_id=chat_request.session_id,
        model="samrat-ai:latest"
    )

# Conversations — moderate limit
@app.get("/conversations")
@limiter.limit("20/minute")
def list_conversations(request: Request, api_key: str = Depends(verify_api_key)):
    return {
        "active_sessions": list(conversations.keys()),
        "total": len(conversations)
    }

# Clear conversation — moderate limit
@app.delete("/chat/{session_id}")
@limiter.limit("20/minute")
def clear_conversation(request: Request, session_id: str, api_key: str = Depends(verify_api_key)):
    if session_id in conversations:
        del conversations[session_id]
    return {"message": f"Conversation {session_id} cleared"}

# ── RAG Helper Functions ──────────────────────────────────
def get_embedding(text):
    response = ollama_client.embeddings(
        model='nomic-embed-text',
        prompt=text
    )
    return np.array(response['embedding'], dtype=np.float32)

def create_faiss_index(chunks):
    embeddings = []
    for chunk in chunks:
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.stack(embeddings))
    return index

def find_relevant_chunks(question, chunks, index, k=5):
    question_embedding = get_embedding(question)
    question_embedding = np.array([question_embedding])
    distances, indices = index.search(question_embedding, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


# ── PDF Upload Endpoint ───────────────────────────────────
@app.post("/upload-pdf/{session_id}")
@limiter.limit("5/minute")
async def upload_pdf(
    request: Request,
    session_id: str,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Extract text from PDF
    reader = PdfReader(tmp_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from PDF"
        )

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)

    # Create FAISS index
    index = create_faiss_index(chunks)

    # Store in RAG sessions
    rag_sessions[session_id] = {
        "chunks": chunks,
        "index": index,
        "filename": file.filename
    }

    # Cleanup temp file
    import os
    os.unlink(tmp_path)

    return {
        "message": "PDF processed successfully",
        "session_id": session_id,
        "filename": file.filename,
        "chunks_created": len(chunks),
        "pages": len(reader.pages)
    }

# ── RAG Chat Endpoint ─────────────────────────────────────
@app.post("/rag-chat", response_model=RAGResponse)
@limiter.limit("10/minute")
def rag_chat(
    request: Request,
    rag_request: RAGRequest,
    api_key: str = Depends(verify_api_key)
):
    # Check if session exists
    if rag_request.session_id not in rag_sessions:
        raise HTTPException(
            status_code=404,
            detail=f"No PDF found for session {rag_request.session_id}. Upload a PDF first via /upload-pdf/{{session_id}}"
        )

    session = rag_sessions[rag_request.session_id]
    chunks = session["chunks"]
    index = session["index"]

    # Find relevant chunks
    relevant_chunks = find_relevant_chunks(
        rag_request.message,
        chunks,
        index,
        k=5
    )
    context = "\n\n".join(relevant_chunks)

    # Build messages
    messages = [
        {
            "role": "system",
            "content": (
                "You are Samrat AI.\n"
                "Answer the question using ONLY the context below.\n"
                "If answer is not in context say: I could not find that in the document.\n"
                "\n\nContext:\n" + context
            )
        },
        {
            "role": "user",
            "content": rag_request.message
        }
    ]

    # Get AI response
    response = ollama_client.chat(
        model="samrat-ai:latest",
        messages=messages
    )

    return RAGResponse(
        response=response["message"]["content"],
        session_id=rag_request.session_id,
        chunks_searched=len(relevant_chunks),
        model="samrat-ai:latest"
    )

# ── List RAG Sessions ─────────────────────────────────────
@app.get("/rag-sessions")
@limiter.limit("20/minute")
def list_rag_sessions(request: Request, api_key: str = Depends(verify_api_key)):
    return {
        "sessions": [
            {
                "session_id": sid,
                "filename": data["filename"],
                "chunks": len(data["chunks"])
            }
            for sid, data in rag_sessions.items()
        ],
        "total": len(rag_sessions)
    }

# ── Delete RAG Session ────────────────────────────────────
@app.delete("/rag-sessions/{session_id}")
@limiter.limit("20/minute")
def delete_rag_session(request: Request, session_id: str, api_key: str = Depends(verify_api_key)):
    if session_id in rag_sessions:
        del rag_sessions[session_id]
        return {"message": f"RAG session {session_id} deleted"}
    raise HTTPException(status_code=404, detail="Session not found")
# ── Run ──────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
