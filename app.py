# app.py
import streamlit as st
import torch

from src.loader import DocumentLoader
from src.chunker import SentenceChunker
from src.embedder import TextEmbedder
from src.rag_pipeline import RAGPipeline

@st.cache_resource
def load_pipeline():
    # Step 1: Load and parse PDF
    loader = DocumentLoader(
        file_path="human-nutrition-text.pdf",
        download_url="https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
    )
    pages = loader.load()

    # Step 2: Chunking
    chunker = SentenceChunker(chunk_size=10)
    chunks = chunker.chunk_pages(pages)

    # Step 3: Embeddings
    embedder = TextEmbedder()
    if not st.session_state.get("cached_embeddings"):
        embeddings = embedder.encode(chunks)
        embedder.save_embeddings(embeddings, chunks)
        st.session_state["cached_embeddings"] = (chunks, embeddings)
    else:
        chunks, embeddings = embedder.load_embeddings()
        embeddings = embeddings.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Step 4: RAG Pipeline
    rag = RAGPipeline(model_name="mistralai/Mistral-7B-Instruct-v0.1", use_faiss=True, faiss_index_path="faiss.index")
    return embedder, rag, chunks, embeddings

# Streamlit UI setup
st.set_page_config(page_title="NutriBot", layout="centered")
st.title("🥦 NutriBot - Your Nutrition Assistant")
st.markdown("Ask any question related to the nutrition textbook.")

# Load pipeline components
embedder, rag_pipeline, doc_chunks, doc_embeddings = load_pipeline()

# Input for user query
query = st.text_input("Enter your question:", key="query")
if query:
    query_embedding = embedder.encode([query])
    answer = rag_pipeline.generate_answer(query, query_embedding, doc_embeddings, doc_chunks)
    st.markdown("---")
    st.subheader("🧠 Answer")
    st.write(answer)

    # Show context used for answer
    with st.expander("🔍 Show retrieved context"):
        contexts = rag_pipeline.retrieve(query_embedding, doc_embeddings, doc_chunks, top_k=3)
        for i, ctx in enumerate(contexts):
            st.markdown(f"**Context {i+1}:** {ctx}")
