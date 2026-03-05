import ollama
import sys
import faiss
import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(file_path):
    """Extract text from PDF"""
    print(f" Loading PDF: {file_path}")
    reader = PdfReader(file_path)
    text = ""
    for i, page in enumerate(reader.pages):
        text += page.extract_text()
        print(f"   Reading page {i+1}/{len(reader.pages)}...", end='\r')
    print(f"\n PDF loaded — {len(reader.pages)} pages extracted")
    return text

def get_embedding(text):
    """Get embedding from Ollama"""
    response = ollama.embeddings(
        model='samrat-ai:latest',
        prompt=text
    )
    return np.array(response['embedding'], dtype=np.float32)

def create_vectorstore(chunks):
    """Create FAISS index from chunks"""
    print("\n Creating embeddings...")
    
    embeddings = []
    for i, chunk in enumerate(chunks):
        print(f"   Embedding chunk {i+1}/{len(chunks)}...", end='\r')
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
    
    print(f"\n Created {len(embeddings)} embeddings")
    
    # Build FAISS index
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.stack(embeddings))
    
    print(" Vector store ready!")
    return index, embeddings

def find_relevant_chunks(question, chunks, index, k=3):
    """Find most relevant chunks for a question"""
    question_embedding = get_embedding(question)
    question_embedding = np.array([question_embedding])
    
    distances, indices = index.search(question_embedding, k)
    
    relevant = [chunks[i] for i in indices[0] if i < len(chunks)]
    return relevant

def ask_question(question, chunks, index):
    """Find relevant chunks and ask AI"""
    print("\n Searching relevant context...")
    
    relevant_chunks = find_relevant_chunks(question, chunks, index)
    context = "\n\n".join(relevant_chunks)
    
    messages = [
        {
            'role': 'system',
            'content': f"""You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context say 'I could not find that in the document.'

Context:
{context}"""
        },
        {
            'role': 'user',
            'content': question
        }
    ]
    
    response = ollama.chat(
        model='samrat-ai:latest',
        messages=messages
    )
    
    return response['message']['content']

def main(pdf_path):
    # Load PDF
    text = load_pdf(pdf_path)
    
    # Split into chunks
    print("\n Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    print(f" Created {len(chunks)} chunks")
    
    # Create vector store
    index, embeddings = create_vectorstore(chunks)
    
    print("\n RAG Chatbot Ready!")
    print("Ask anything about your document.")
    print("Type 'quit' to exit")
    print("-" * 30)
    
    while True:
        question = input("\nYou: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        if question.strip() == '':
            continue
        
        answer = ask_question(question, chunks, index)
        print(f"\nAI: {answer}")
        print("-" * 30)

# Entry point
if len(sys.argv) < 2:
    print(" Please provide a PDF path")
    print("Usage: python rag.py /path/to/your/file.pdf")
else:
    main(sys.argv[1])
