import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import ollama
import faiss
import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from memory import load_memory, save_memory, extract_memories, build_memory_prompt, add_conversation_summary




# ── Owner Password ───────────────────────────────────────
from dotenv import load_dotenv
import os
load_dotenv()
OWNER_PASSWORD = os.getenv("OWNER_PASSWORD", "changeme")

# Page Config
st.set_page_config(
    page_title="Samrat AI",
    page_icon="O_O",
    layout="centered"
)

# Header
st.title("Samrat AI")
st.caption("Personal AI by Samrat | 100% Private | Zero Cloud | Runs on your device")
st.divider()

# Helper Functions
def load_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_embedding(text):
    response = ollama.embeddings(
        model='nomic-embed-text',
        prompt=text
    )
    return np.array(response['embedding'], dtype=np.float32)

def create_vectorstore(chunks):
    embeddings = []
    progress = st.progress(0, text="Creating embeddings...")
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
        progress.progress(
            (i + 1) / len(chunks),
            text=f"Embedding chunk {i+1}/{len(chunks)}..."
        )
    progress.empty()
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.stack(embeddings))
    return index, embeddings

def find_relevant_chunks(question, chunks, index, k=5):
    question_embedding = get_embedding(question)
    question_embedding = np.array([question_embedding])
    distances, indices = index.search(question_embedding, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def ask_question(question, chunks, index):
    relevant_chunks = find_relevant_chunks(question, chunks, index)
    context = "\n\n".join(relevant_chunks)
    memory = load_memory()
    memory_context = build_memory_prompt(memory)

    messages = [
        {
            'role': 'system',
            'content': (
                "You are Samrat AI.\n"
                "Answer the question using ONLY the context below.\n"
                "If answer is not in context say: I could not find that in the document.\n"
                + memory_context +
                "\n\nContext:\n" + context

            )
        },
        {
            'role': 'user',
            'content': question
        }
    ]
    response = ollama.chat(model='samrat-ai', messages=messages)
    return response['message']['content']

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "index" not in st.session_state:
    st.session_state.index = None
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "is_owner" not in st.session_state:
    st.session_state.is_owner = False
if "show_login" not in st.session_state:
    st.session_state.show_login = False

# Sidebar
with st.sidebar:
    st.markdown("### Samrat AI")
    st.caption("Powered by Qwen2.5 | Built by Samrat")
    st.divider()

    # Owner Login
    if not st.session_state.is_owner:
        if st.button("Owner Login", use_container_width=True, key="btn_owner_login"):
            st.session_state.show_login = True

        if st.session_state.show_login:
            password = st.text_input(
                "Enter password",
                type="password",
                key="password_input"
            )
            if st.button("Login", use_container_width=True, key="btn_login"):
                if password == OWNER_PASSWORD:
                    st.session_state.is_owner = True
                    st.session_state.show_login = False
                    st.session_state.messages = []
                    st.success("Welcome back Samrat!")
                    st.rerun()
                else:
                    st.error("Wrong password!")
    else:
        st.success("Owner Mode Active")
        if st.button("Logout", use_container_width=True, key="btn_logout"):
            st.session_state.is_owner = False
            st.session_state.messages = []
            st.session_state.pdf_loaded = False
            st.session_state.chunks = None
            st.session_state.index = None
            st.rerun()

    st.divider()

    # PDF Upload — Owner only
    if st.session_state.is_owner:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

        if uploaded_file and not st.session_state.pdf_loaded:
            with st.spinner("Processing PDF..."):
                text = load_pdf(uploaded_file)
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = splitter.split_text(text)
                index, embeddings = create_vectorstore(chunks)
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.pdf_loaded = True
            st.success("PDF loaded - " + str(len(chunks)) + " chunks created!")

        if st.session_state.pdf_loaded:
            st.divider()
            st.markdown("**Mode:** Document Q&A")
            st.markdown("**Chunks:** " + str(len(st.session_state.chunks)))
            st.markdown("**Model:** Samrat AI")
            st.divider()
            if st.button("Clear Document", use_container_width=True, key="btn_clear_doc"):
                st.session_state.pdf_loaded = False
                st.session_state.chunks = None
                st.session_state.index = None
                st.rerun()

        st.divider()

        # Memory Viewer — Owner only
        with st.expander("View Memory"):
            memory = load_memory()
            if memory['facts']:
                st.markdown("**Known Facts:**")
                for fact in memory['facts']:
                    st.markdown("- " + fact)
            if memory['preferences']:
                st.markdown("**Preferences:**")
                for pref in memory['preferences']:
                    st.markdown("- " + pref)
            if not memory['facts'] and not memory['preferences']:
                st.caption("No memories yet - start chatting!")
            if st.button("Clear Memory", use_container_width=True, key="btn_clear_memory_owner"):
                from memory import clear_memory
                clear_memory()
                st.success("Memory cleared!")
                st.rerun()

        st.divider()
        if st.button("Clear Chat", use_container_width=True, key="btn_clear_chat"):
            st.session_state.messages = []
            st.rerun()

    # About — visible to everyone
    with st.expander("About Samrat AI"):
        st.markdown("**Samrat AI** is a fully private local AI assistant built by Samrat.")
        st.markdown("- 100% Private")
        st.markdown("- PDF Document Q&A")
        st.markdown("- RAG powered search")
        st.markdown("- Runs fully offline")
        st.markdown("- Zero cloud dependency")
        st.markdown("Built with Ollama, FAISS, LangChain and Streamlit")
    st.divider()

    # Memory Viewer
    with st.expander("View Memory"):
        memory = load_memory()
        if memory['facts']:
            st.markdown("**Known Facts:**")
            for fact in memory['facts']:
                st.markdown("- " + fact)
        if memory['preferences']:
            st.markdown("**Preferences:**")
            for pref in memory['preferences']:
                st.markdown("- " + pref)
        if not memory['facts'] and not memory['preferences']:
            st.caption("No memories yet — start chatting!")
        if st.button("Clear Memory", use_container_width=True, key="btn_clear_memory_stranger"):
            from memory import clear_memory
            clear_memory()
            st.success("Memory cleared!")
            st.rerun()

    st.divider()
    if st.button("Clear Chat", use_container_width=True, key="btn_clear_chat_2"):
        st.session_state.messages = []
        st.rerun()

# Welcome message
if st.session_state.is_owner:
    if not st.session_state.pdf_loaded:
        st.info("Welcome back Samrat! Upload a PDF or just start chatting.")
else:
    st.info("Welcome! You are chatting with Samrat AI — a personal AI assistant built by Samrat.")
    st.caption("This is a public demo. Login as owner for full access.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask Samrat AI anything..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            if st.session_state.pdf_loaded and st.session_state.is_owner:
                answer = ask_question(
                    prompt,
                    st.session_state.chunks,
                    st.session_state.index
                )
            else:
                # Owner gets memory, stranger does not
                if st.session_state.is_owner:
                    memory = load_memory()
                    memory_context = build_memory_prompt(memory)
                else:
                    memory_context = ""

                system_msg = {
                    "role": "system",
                    "content": (
                        "You are Samrat AI, a personal AI assistant built by Samrat.\n"
                        "You run 100% locally with complete privacy.\n"
                        "Be helpful, concise and intelligent.\n"
                        + memory_context

                    )
                }

                all_messages = [system_msg] + st.session_state.messages
                response = ollama.chat(
                    model='samrat-ai',
                    messages=all_messages
                )
                answer = response['message']['content']

                # Save memories only for owner
                if st.session_state.is_owner:
                    memory = extract_memories(st.session_state.messages, memory)
                    save_memory(memory)

        st.write(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

# Stranger watermark
if not st.session_state.is_owner:
    st.divider()
    st.caption("Demo by Samrat | Built with Ollama + FAISS + LangChain + Streamlit")
