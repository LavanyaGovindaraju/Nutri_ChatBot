# src/vector_store.py

import faiss
import numpy as np
from typing import List, Tuple

class VectorStore:
    """
    Simple FAISS-based vector store for storing and searching text embeddings.
    """

    def __init__(self, embedding_dim: int):
        """
        Args:
            embedding_dim (int): Dimension of the embeddings (e.g., 384 for MiniLM).
        """
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.texts = []

    def add(self, embeddings: np.ndarray, texts: List[str]):
        """
        Adds embeddings and corresponding texts to the vector store.
        
        Args:
            embeddings (np.ndarray): Array of embeddings to add.
            texts (List[str]): Corresponding texts for embeddings.
        """
        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Searches for the top_k most similar texts to the query embedding.

        Args:
            query_embedding (np.ndarray): Embedding of the query.
            top_k (int): Number of top results to return.
        
        Returns:
            List[Tuple[str, float]]: List of (text, similarity_score).
        """
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1:
                results.append((self.texts[idx], distance))
        return results
