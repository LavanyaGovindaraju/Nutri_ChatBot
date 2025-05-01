# app.py
import streamlit as st
import torch

from src.loader import DocumentLoader
from src.chunker import SentenceChunker
from src.embedder import TextEmbedder
from src.rag_pipeline import RAGPipeline

MODEL_OPTIONS = {
    "TinyLlama (Fastest)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Phi-2 (Mid-size)": "microsoft/phi-2",
    "Falcon-7B-Instruct (Larger)": "tiiuae/falcon-7b-instruct",
    # Uncomment below if authenticated for gated models:
    # "Mistral-7B-Instruct (Gated)": "mistralai/Mistral-7B-Instruct-v0.1"
}

st.set_page_config(page_title="NutriBot", layout="centered")
st.title("🥦 NutriBot - Your Nutrition Assistant")

selected_model_name = st.selectbox("Choose your model:", list(MODEL_OPTIONS.keys()))
selected_model = MODEL_OPTIONS[selected_model_name]

@st.cache_resource
def load_pipeline(model_name):
    loader = DocumentLoader(
        file_path="human-nutrition-text.pdf",
        download_url="https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
    )
    pages = loader.load()

    chunker = SentenceChunker(chunk_size=10)
    chunks = chunker.chunk_pages(pages)

    embedder = TextEmbedder()
    if not st.session_state.get("cached_embeddings"):
        embeddings = embedder.encode(chunks)
        embedder.save_embeddings(embeddings, chunks)
        st.session_state["cached_embeddings"] = (chunks, embeddings)
    else:
        chunks, embeddings = embedder.load_embeddings()
        embeddings = embeddings.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    rag = RAGPipeline(model_name=model_name, use_faiss=True, faiss_index_path="faiss.index")
    return embedder, rag, chunks, embeddings

embedder, rag_pipeline, doc_chunks, doc_embeddings = load_pipeline(selected_model)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input
query = st.chat_input("Ask something about nutrition...")
if query:
    query_embedding = embedder.encode([query])
    answer = rag_pipeline.generate_answer(query, query_embedding, doc_embeddings, doc_chunks)
    st.session_state.chat_history.append((query, answer))

# Display chat
for user_q, bot_a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user_q)
    with st.chat_message("assistant"):
        st.markdown(bot_a)
