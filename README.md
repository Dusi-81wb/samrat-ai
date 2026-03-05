# Samrat AI — Personal Local AI Assistant

A fully private, local AI assistant with RAG pipeline built from scratch.
Runs 100% on-device with zero cloud dependency.

## Features
- 100% Private — nothing leaves your machine
- PDF Document Q&A using RAG pipeline
- Persistent memory across conversations
- Owner vs Stranger mode with password protection
- Beautiful Streamlit web interface
- Works completely offline
- Local network and public sharing via Ngrok or anyother of your choice

## Tech Stack
- Python 3.14
- Ollama — local model runner
- Qwen2.5 3B — language model
- FAISS — vector similarity search
- LangChain — RAG pipeline
- Streamlit — web interface
- nomic-embed-text — embeddings

## Architecture
User Query
    ->
PDF chunks stored in FAISS vector DB
    ->
Semantic search finds relevant chunks
    ->
Qwen2.5 3B generates answer locally
    ->
Persistent memory saves important facts

## Project Structure
- app.py — Streamlit web UI
- chat.py — Terminal chatbot
- rag.py — RAG pipeline
- memory.py — Persistent memory system
- Modelfile — Custom AI model config
- requirements.txt — Python dependencies

## Installation

### 1. Clone the repository
git clone https://github.com/Dusi-81wb/samrat-ai.git
cd samrat-ai

### 2. Install Ollama
Download from https://ollama.com and pull models:

ollama pull qwen2.5:3b-instruct
ollama pull nomic-embed-text

### 3. Create virtual environment
python -m venv venv
source venv/bin/activate.fish

### 4. Install dependencies
pip install -r requirements.txt

### 5. Create your own Modelfile
ollama create samrat-ai -f Modelfile

### 6. Run
streamlit run app.py

## Author
Built by Samrat — Data Science and AI Engineering Student
