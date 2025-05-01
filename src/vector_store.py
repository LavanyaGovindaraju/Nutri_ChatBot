# src/vector_store.py
import faiss
import numpy as np
import torch
from typing import List, Tuple

class VectorStore:
    """
    Simple FAISS-based vector store for storing and searching text embeddings,
    with persistence support.
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.texts = []

    def add(self, embeddings: np.ndarray, texts: List[str]):
        self.index.add(embeddings)
        self.texts.extend(texts)

    def add_tensor(self, embeddings: torch.Tensor, texts: List[str]):
        self.add(embeddings.cpu().numpy().astype(np.float32), texts)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        distances, indices = self.index.search(query_embedding, top_k)
        return [
            (self.texts[idx], distances[0][i])
            for i, idx in enumerate(indices[0]) if idx != -1
        ]

    def search_tensor(self, query_embedding: torch.Tensor, top_k: int = 5):
        return self.search(query_embedding.cpu().numpy().astype(np.float32), top_k)

    def save(self, index_path: str, text_path: str):
        faiss.write_index(self.index, index_path)
        with open(text_path, "w", encoding="utf-8") as f:
            for line in self.texts:
                f.write(line.replace("\n", " ") + "\n")

    def load(self, index_path: str, text_path: str):
        self.index = faiss.read_index(index_path)
        with open(text_path, "r", encoding="utf-8") as f:
            self.texts = [line.strip() for line in f.readlines()]
