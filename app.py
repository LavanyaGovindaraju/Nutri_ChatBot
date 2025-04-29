# app.py

import streamlit as st
import numpy as np
from src.loader import DocumentLoader
from src.chunker import TextChunker
from src.embedder import TextEmbedder
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="NutriBot 🥑", page_icon="🥑", layout="wide")

# Page title and description
st.title("NutriBot 🥑")
st.subheader("Your Personal Assistant for Nutrition Knowledge 📚")
st.markdown("Ask questions and get factual answers directly from the Human Nutrition textbook!")

# Load Document
@st.cache_data
def load_document():
    loader = DocumentLoader(
        file_path="human-nutrition-text.pdf",
        download_url="https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
    )
    return loader.load()

# Chunk Document
@st.cache_data
def chunk_document(pages):
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    return chunker.chunk_text(pages)

# Embed Chunks
@st.cache_data
def embed_chunks(chunks):
    embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")
    return embedder.embed(chunks)

# Initialize RAG Pipeline
@st.cache_resource
def initialize_rag(embeddings, chunks):
    rag = RAGPipeline(embedding_dim=embeddings.shape[1], model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    rag.add_documents(embeddings, chunks)
    return rag

# Processing Step (Background Loading)
with st.spinner("Preparing NutriBot..."):
    pages = load_document()
    chunks = chunk_document(pages)
    embeddings = embed_chunks(chunks)
    rag_pipeline = initialize_rag(embeddings, chunks)

# Divider
st.divider()

# Section: NutriBot Chat Interface
st.subheader("💬 Chat with NutriBot")

# Show entire history at the top
for user_msg, bot_msg in st.session_state.get("chat_history", []):
    st.chat_message("user").markdown(user_msg)
    st.chat_message("assistant").markdown(bot_msg)

# Capture new input at the bottom
query = st.chat_input("Ask a question about human nutrition...")

if query:
    # Display user's new message immediately
    st.chat_message("user").markdown(query)

    with st.spinner("NutriBot is thinking..."):
        embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")
        query_embedding = embedder.embed([query])
        query_embedding = np.array(query_embedding)
        answer = rag_pipeline.generate_answer(query, query_embedding)

    # Show the bot's reply below the user's input
    st.chat_message("assistant").markdown(answer)

    # Store the conversation
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append((query, answer))
