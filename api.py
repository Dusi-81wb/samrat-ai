import warnings
warnings.filterwarnings("ignore")

import os
import asyncio
import hashlib
import json
import pickle
import uvicorn
import faiss
import numpy as np
import tempfile
import redis

from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
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

# ── Redis Client ──────────────────────────────────────────
try:
    redis_client = redis.Redis(
        host='localhost',
        port=6379,
        db=0,
        decode_responses=False
    )
    redis_client.ping()
    print("Redis connected!")
except:
    redis_client = None
    print("Redis not available — caching disabled")

# ── API Key Setup ─────────────────────────────────────────
API_KEY = os.getenv("API_KEY", "samrat-secret-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key"
        )
    return api_key

# ── Rate Limiter ──────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── FastAPI App ───────────────────────────────────────────
app = FastAPI(
    title="Samrat AI API",
    description="Privacy-first local AI assistant API built by Samrat",
    version="2.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Storage ───────────────────────────────────────────────
conversations = {}
rag_sessions = {}
embedding_cache = {}

# ── Models ───────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    session_id: str
    model: str
    cached: bool = False

class RAGRequest(BaseModel):
    message: str
    session_id: str

class RAGResponse(BaseModel):
    response: str
    session_id: str
    chunks_searched: int
    model: str

# ── Cache Helpers ─────────────────────────────────────────
def get_cache_key(message: str) -> str:
    """Generate unique hash key for a message"""
    return hashlib.md5(message.lower().strip().encode()).hexdigest()

def get_cached_response(message: str):
    """Get cached response from Redis"""
    if not redis_client:
        return None
    try:
        key = f"response:{get_cache_key(message)}"
        cached = redis_client.get(key)
        if cached:
            return json.loads(cached.decode('utf-8'))
    except:
        pass
    return None

def cache_response(message: str, response: str, ttl: int = 3600):
    """Cache response in Redis with TTL"""
    if not redis_client:
        return
    try:
        key = f"response:{get_cache_key(message)}"
        redis_client.setex(key, ttl, json.dumps(response))
    except:
        pass

def get_cached_embeddings(file_hash: str):
    """Get cached FAISS embeddings from disk"""
    cache_path = f".embedding_cache/{file_hash}.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            print(f"Loading embeddings from cache!")
            return pickle.load(f)
    return None

def save_embeddings_cache(file_hash: str, data: dict):
    """Save FAISS embeddings to disk"""
    os.makedirs(".embedding_cache", exist_ok=True)
    cache_path = f".embedding_cache/{file_hash}.pkl"
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Embeddings cached to disk!")

# ── RAG Helpers ───────────────────────────────────────────
def get_embedding(text: str):
    """Get embedding — check memory cache first"""
    if text in embedding_cache:
        return embedding_cache[text]
    response = ollama_client.embeddings(
        model='nomic-embed-text',
        prompt=text
    )
    embedding = np.array(response['embedding'], dtype=np.float32)
    embedding_cache[text] = embedding
    return embedding

def create_faiss_index(chunks: list):
    """Create FAISS index from chunks"""
    embeddings = []
    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i+1}/{len(chunks)}...", end='\r')
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
    print(f"\nEmbeddings done!")
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.stack(embeddings))
    return index, embeddings

def find_relevant_chunks(question: str, chunks: list, index, k: int = 5):
    """Find most relevant chunks for a question"""
    question_embedding = get_embedding(question)
    question_embedding = np.array([question_embedding])
    distances, indices = index.search(question_embedding, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# ── Async AI Response ─────────────────────────────────────
async def get_ai_response_async(messages: list) -> str:
    """Run Ollama in thread pool to avoid blocking"""
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: ollama_client.chat(
            model="samrat-ai:latest",
            messages=messages
        )
    )
    return response["message"]["content"]

async def stream_ai_response(messages: list) -> AsyncGenerator[str, None]:
    """Stream AI response token by token"""
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()

    def run_stream():
        stream = ollama_client.chat(
            model="samrat-ai:latest",
            messages=messages,
            stream=True
        )
        for chunk in stream:
            token = chunk['message']['content']
            asyncio.run_coroutine_threadsafe(
                queue.put(token), loop
            )
        asyncio.run_coroutine_threadsafe(
            queue.put(None), loop
        )

    # Run stream in thread pool
    loop.run_in_executor(None, run_stream)

    while True:
        token = await queue.get()
        if token is None:
            break
        yield token

# ── Routes ────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "Samrat AI API",
        "version": "2.0.0",
        "status": "running",
        "built_by": "Samrat",
        "redis": "connected" if redis_client else "disabled",
        "features": [
            "async",
            "streaming",
            "response_cache",
            "embedding_cache",
            "rate_limiting",
            "api_key_auth"
        ]
    }

@app.get("/health")
@limiter.limit("60/minute")
async def health(request: Request):
    return {
        "status": "healthy",
        "redis": redis_client.ping() if redis_client else False,
        "ollama": True
    }

@app.post("/chat")
@limiter.limit("10/minute")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    # Check cache first for non-streaming requests
    if not chat_request.stream:
        cached = get_cached_response(chat_request.message)
        if cached:
            return ChatResponse(
                response=cached,
                session_id=chat_request.session_id,
                model="samrat-ai:latest",
                cached=True
            )

    # Build conversation history
    if chat_request.session_id not in conversations:
        conversations[chat_request.session_id] = []

    conversations[chat_request.session_id].append({
        "role": "user",
        "content": chat_request.message
    })

    messages = [
        {
            "role": "system",
            "content": "You are Samrat AI, built by Samrat. Be helpful and concise."
        }
    ] + conversations[chat_request.session_id]

    # Streaming response
    if chat_request.stream:
        async def generate():
            full_response = ""
            async for token in stream_ai_response(messages):
                full_response += token
                yield f"data: {json.dumps({'token': token})}\n\n"

            # Save to conversation history
            conversations[chat_request.session_id].append({
                "role": "assistant",
                "content": full_response
            })
            yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )

    # Normal async response
    reply = await get_ai_response_async(messages)

    conversations[chat_request.session_id].append({
        "role": "assistant",
        "content": reply
    })

    # Cache the response
    cache_response(chat_request.message, reply)

    return ChatResponse(
        response=reply,
        session_id=chat_request.session_id,
        model="samrat-ai:latest",
        cached=False
    )

@app.post("/upload-pdf/{session_id}")
@limiter.limit("5/minute")
async def upload_pdf(
    request: Request,
    session_id: str,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    # Read file content
    content = await file.read()

    # Generate hash for caching
    file_hash = hashlib.md5(content).hexdigest()

    # Check embedding cache first
    cached_data = get_cached_embeddings(file_hash)
    if cached_data:
        rag_sessions[session_id] = cached_data
        return {
            "message": "PDF loaded from cache!",
            "session_id": session_id,
            "filename": file.filename,
            "chunks_created": len(cached_data["chunks"]),
            "from_cache": True
        }

    # Save temporarily and process
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    # Extract text
    reader = PdfReader(tmp_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    pages = len(reader.pages)

    os.unlink(tmp_path)

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)

    # Create FAISS index
    index, embeddings = create_faiss_index(chunks)

    # Save to session
    session_data = {
        "chunks": chunks,
        "index": index,
        "filename": file.filename,
        "embeddings": embeddings
    }
    rag_sessions[session_id] = session_data

    # Cache embeddings to disk
    save_embeddings_cache(file_hash, session_data)

    return {
        "message": "PDF processed successfully",
        "session_id": session_id,
        "filename": file.filename,
        "chunks_created": len(chunks),
        "pages": pages,
        "from_cache": False
    }

@app.post("/rag-chat", response_model=RAGResponse)
@limiter.limit("10/minute")
async def rag_chat(
    request: Request,
    rag_request: RAGRequest,
    api_key: str = Depends(verify_api_key)
):
    if rag_request.session_id not in rag_sessions:
        raise HTTPException(
            status_code=404,
            detail="No PDF found for this session. Upload a PDF first."
        )

    session = rag_sessions[rag_request.session_id]
    chunks = session["chunks"]
    index = session["index"]

    # Find relevant chunks
    relevant_chunks = find_relevant_chunks(
        rag_request.message, chunks, index, k=5
    )
    context = "\n\n".join(relevant_chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are Samrat AI.\n"
                "Answer using ONLY the context below.\n"
                "If not found say: I could not find that in the document.\n"
                "\n\nContext:\n" + context
            )
        },
        {"role": "user", "content": rag_request.message}
    ]

    # Async response
    reply = await get_ai_response_async(messages)

    return RAGResponse(
        response=reply,
        session_id=rag_request.session_id,
        chunks_searched=len(relevant_chunks),
        model="samrat-ai:latest"
    )

@app.delete("/chat/{session_id}")
@limiter.limit("20/minute")
async def clear_conversation(
    request: Request,
    session_id: str,
    api_key: str = Depends(verify_api_key)
):
    if session_id in conversations:
        del conversations[session_id]
    return {"message": f"Conversation {session_id} cleared"}

@app.get("/conversations")
@limiter.limit("20/minute")
async def list_conversations(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    return {
        "active_sessions": list(conversations.keys()),
        "total": len(conversations)
    }

@app.get("/rag-sessions")
@limiter.limit("20/minute")
async def list_rag_sessions(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
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

@app.delete("/rag-sessions/{session_id}")
@limiter.limit("20/minute")
async def delete_rag_session(
    request: Request,
    session_id: str,
    api_key: str = Depends(verify_api_key)
):
    if session_id in rag_sessions:
        del rag_sessions[session_id]
        return {"message": f"RAG session {session_id} deleted"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/cache/stats")
@limiter.limit("20/minute")
async def cache_stats(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    if not redis_client:
        return {"redis": "disabled"}
    keys = redis_client.keys("response:*")
    return {
        "cached_responses": len(keys),
        "redis_status": "connected",
        "embedding_cache_size": len(embedding_cache),
        "disk_embedding_cache": len(os.listdir(".embedding_cache"))
        if os.path.exists(".embedding_cache") else 0
    }

@app.delete("/cache/clear")
@limiter.limit("5/minute")
async def clear_cache(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    if redis_client:
        keys = redis_client.keys("response:*")
        for key in keys:
            redis_client.delete(key)
    embedding_cache.clear()
    return {"message": "Cache cleared!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
