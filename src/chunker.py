# src/chunker.py

from typing import List

class TextChunker:
    """
    Splits text documents into smaller chunks.
    """

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        """
        Args:
            chunk_size (int): Maximum number of words per chunk.
            chunk_overlap (int): Number of words to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, texts: List[str]) -> List[str]:
        """
        Splits list of page texts into overlapping chunks.
        
        Args:
            texts (List[str]): List of page texts.
        
        Returns:
            List[str]: List of chunks.
        """
        chunks = []
        for text in texts:
            words = text.split()
            if len(words) == 0:
                continue
            start = 0
            while start < len(words):
                end = start + self.chunk_size
                chunk = " ".join(words[start:end])
                chunks.append(chunk)
                start += self.chunk_size - self.chunk_overlap
        return chunks
