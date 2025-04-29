import streamlit as st
from src.loader import DocumentLoader
from src.chunker import TextChunker

st.set_page_config(page_title="NutriBot", page_icon="🥑", layout="wide")
st.title("NutriBot 🥑 - Your Nutrition Assistant")
st.write("Ask me anything from the Nutrition Textbook!")

# Load the document
@st.cache_data
def load_document():
    loader = DocumentLoader(
        file_path="human-nutrition-text.pdf",
        download_url="https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
    )
    return loader.load()

pages = load_document()
st.success(f"Loaded {len(pages)} pages from the Nutrition textbook!")

# Chunk the document
@st.cache_data
def chunk_document(pages):
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    return chunker.chunk_text(pages)

chunks = chunk_document(pages)
st.success(f"Generated {len(chunks)} text chunks ready for embedding!")
