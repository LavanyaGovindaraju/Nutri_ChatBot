import streamlit as st
from src.loader import DocumentLoader

st.set_page_config(page_title="NutriBot", page_icon="🥑", layout="wide")
st.title("NutriBot 🥑 - Your Nutrition Assistant")
st.write("Ask me anything from the Nutrition Textbook!")

# Load document (Download if missing)
@st.cache_data
def load_document():
    loader = DocumentLoader(
        file_path="human-nutrition-text.pdf",
        download_url="https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
    )
    return loader.load()

pages = load_document()

st.success(f"Loaded {len(pages)} pages from the Nutrition textbook!")
