# src/rag_pipeline.py

from src.vector_store import VectorStore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import numpy as np
from typing import List
import torch

class RAGPipeline:
    """
    Combines retrieval (vector store) and generation (LLM) for answering questions.
    """

    def __init__(self, embedding_dim: int = 384, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.vector_store = VectorStore(embedding_dim=embedding_dim)

        print(f"[INFO] Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )

        print("[INFO] RAG Pipeline initialized with TinyLlama (optimized mode).")

    def add_documents(self, embeddings: np.ndarray, texts: List[str]):
        self.vector_store.add(embeddings, texts)

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 2) -> List[str]:
        """
        Retrieve top-k relevant chunks for a query (reduced to 2 for faster prompt building).
        """
        results = self.vector_store.search(query_embedding, top_k=top_k)
        return [text for text, _ in results]

    def generate_answer(self, query: str, query_embedding: np.ndarray) -> str:
        """
        Retrieve relevant context and generate an answer to the query.
        """
        # 1. Retrieve top-2 relevant chunks
        retrieved_texts = self.retrieve(query_embedding, top_k=2)
        context = " ".join(retrieved_texts)

        # 2. Build a smaller prompt
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        # 3. Generate a short, deterministic answer
        outputs = self.generator(
            prompt,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.7
        )
    


        generated_text = outputs[0]["generated_text"]
        answer = generated_text.replace(prompt, "").strip()
        return answer
