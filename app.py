# app.py
import streamlit as st
import torch

from src.loader import DocumentLoader
from src.chunker import SentenceChunker
from src.embedder import TextEmbedder
from src.langchain_pipeline import RAGPipeline

# ---------------------- UI CONFIGURATION ----------------------
st.set_page_config(
    page_title="NutriBot - Your Nutrition Assistant",
    page_icon="🥑",
    layout="centered",
    initial_sidebar_state="auto"
)

# Header with branding
st.markdown("""
# 🥑 NutriBot
Your personalized assistant for understanding human nutrition.
""")
st.divider()

# ---------------------- MODEL SELECTION ----------------------
MODEL_OPTIONS = {
    "TinyLlama (Fastest)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Phi-2 (Mid-size)": "microsoft/phi-2",
    "Falcon-7B-Instruct (Larger)": "tiiuae/falcon-7b-instruct",
    "Mistral-7B-Instruct (Gated)": "mistralai/Mistral-7B-Instruct-v0.1"
}

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    selected_model_name = st.selectbox("Choose a model:", list(MODEL_OPTIONS.keys()))
    selected_model = MODEL_OPTIONS[selected_model_name]
    top_k = st.slider("# of contexts to retrieve:", 1, 10, 3)

# ---------------------- LOAD COMPONENTS ----------------------
@st.cache_resource(show_spinner=True)
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

    rag = RAGPipeline(model_name=model_name, faiss_index_path="faiss.index")
    return embedder, rag, chunks, embeddings

embedder, rag_pipeline, doc_chunks, doc_embeddings = load_pipeline(selected_model)

# ---------------------- CHAT SECTION ----------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("""
### 💬 Chat with NutriBot
Ask anything related to the nutrition textbook.
""")

query = st.chat_input("Type your question here...")
if query:
    with st.spinner("NutriBot is thinking..."):
        answer, retrieved_contexts = rag_pipeline.chat(query)
        st.session_state.chat_history.append((query, answer, retrieved_contexts))

# Display prior chat
for user_q, bot_a, contexts in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"**You:** {user_q}")
    with st.chat_message("assistant"):
        st.markdown(f"**NutriBot:** {bot_a}")
        if contexts:
            with st.expander("🔍 Show retrieved context"):
                for i, ctx in enumerate(contexts):
                    short_ctx = ctx[:400] + "..." if len(ctx) > 400 else ctx
                    st.markdown(f"<div style='margin-bottom:1em;padding:0.5em;border-left:3px solid #4CAF50;background-color:#f9f9f9;'> <b>Context {i+1}:</b><br>{short_ctx}</div>", unsafe_allow_html=True)
