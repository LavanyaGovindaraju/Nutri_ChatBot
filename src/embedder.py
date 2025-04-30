# src/embedder.py

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import torch

class TextEmbedder:
    """
    Encodes text chunks into dense vector embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name (str): Pretrained model from sentence-transformers to use.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Encodes a list of texts into embeddings.
        
        Args:
            texts (List[str]): List of text chunks.
        
        Returns:
            np.ndarray: Array of embeddings.
        """
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return embeddings
