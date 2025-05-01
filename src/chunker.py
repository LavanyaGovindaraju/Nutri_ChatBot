# src/chunker.py
import spacy
from tqdm.auto import tqdm

class SentenceChunker:
    """
    Uses spaCy's sentencizer to split text into sentences,
    then groups them into chunks of fixed size.
    """

    def __init__(self, chunk_size=10):
        """
        Args:
            chunk_size (int): Number of sentences per chunk.
        """
        self.chunk_size = chunk_size
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")

    def chunk_pages(self, pages):
        """
        Splits pages into sentence chunks.

        Args:
            pages (list): List of dictionaries with "text" fields.

        Returns:
            list: Sentence chunks extracted from all pages.
        """
        all_chunks = []

        for page in tqdm(pages, desc="Chunking text"):
            doc = self.nlp(page["text"])
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

            # Split into chunks of N sentences
            for i in range(0, len(sentences), self.chunk_size):
                chunk = " ".join(sentences[i:i + self.chunk_size])
                all_chunks.append(chunk)

        return all_chunks
