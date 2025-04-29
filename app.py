# app.py

import streamlit as st
from src.loader import DocumentLoader
from src.chunker import TextChunker
from src.embedder import TextEmbedder
# We'll import VectorStore and rag_pipeline later!

st.set_page_config(page_title="NutriBot", page_icon="🥑", layout="wide")

st.title("NutriBot 🥑 - Your Nutrition Assistant")
st.write("Ask me anything from the Human Nutrition textbook!")

# 1. Load Document
@st.cache_data
def load_document():
    loader = DocumentLoader(
        file_path="human-nutrition-text.pdf",
        download_url="https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
    )
    return loader.load()

pages = load_document()
st.success(f"✅ Loaded {len(pages)} pages from the Nutrition textbook.")

# 2. Chunk Document
@st.cache_data
def chunk_document(pages):
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    return chunker.chunk_text(pages)

chunks = chunk_document(pages)
st.success(f"✅ Generated {len(chunks)} text chunks ready for embedding.")

# 3. Embed Chunks
@st.cache_data
def embed_chunks(chunks):
    embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")
    return embedder.embed(chunks)

embeddings = embed_chunks(chunks)
st.success(f"✅ Embedded all chunks. Shape: {embeddings.shape}")

# 4. Placeholder for Vector Store and Retrieval
st.info("🔜 Retrieval and RAG pipeline coming soon... Stay tuned!")

# (Later we will: initialize vector store, build retriever, answer queries.)
