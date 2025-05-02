# src/embedder.py
import os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    """
    Encodes a list of text chunks into dense vector embeddings using SentenceTransformer.
    Caches the results to CSV and supports torch Tensor loading.
    """

    def __init__(self, model_name="all-mpnet-base-v2", device=None):
        """
        Args:
            model_name (str): Name of the sentence-transformers model to use.
            device (str, optional): Force 'cuda' or 'cpu'. Defaults to auto-detection.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
        except NotImplementedError:
            # Safe fallback for meta tensor issue
            self.model = SentenceTransformer(model_name)  # no device param
            print("⚠️ Falling back to CPU mode due to meta tensor issue.")

    def encode(self, texts):
        """
        Generates embeddings for a list of texts.

        Args:
            texts (list[str]): List of text strings to embed.

        Returns:
            torch.Tensor: Embeddings as a torch tensor.
        """
        embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        return embeddings

    def save_embeddings(self, embeddings, texts, path="embeddings.csv"):
        """
        Saves embeddings and their associated text chunks to a CSV file.

        Args:
            embeddings (torch.Tensor): The generated embeddings.
            texts (list[str]): Corresponding texts.
            path (str): Output file path.
        """
        df = pd.DataFrame(embeddings.cpu().numpy())
        df.insert(0, "text", texts)
        df.to_csv(path, index=False)

    def load_embeddings(self, path="embeddings.csv"):
        """
        Loads embeddings and text chunks from CSV.

        Args:
            path (str): Path to the CSV file.

        Returns:
            tuple: (List of texts, torch.Tensor of embeddings)
        """
        df = pd.read_csv(path)
        texts = df["text"].tolist()
        emb_tensor = torch.tensor(df.drop(columns=["text"]).values)
        return texts, emb_tensor
