# src/langchain_pipeline.py

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import pipeline as hf_pipeline
import torch
import os
from typing import Tuple, List
from pathlib import Path

class RAGPipeline:
    """
    LangChain-based Retrieval-Augmented Generation (RAG) pipeline
    using HuggingFace Transformers and FAISS vector store.
    """

    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.1", faiss_index_path="faiss.index"):
        self.device = 0 if torch.cuda.is_available() else -1
        self.faiss_index_path = faiss_index_path

        # Load generation model into HuggingFace pipeline
        self.model_pipeline = hf_pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device=self.device,
            max_new_tokens=256,
            repetition_penalty=1.1,
            do_sample=True
        )

        # Wrap LLM for LangChain
        self.llm = HuggingFacePipeline(pipeline=self.model_pipeline)

        # Load embeddings and FAISS vector store
        self.embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        index_dir = Path(faiss_index_path)
        if not index_dir.exists():
            print(f"[INFO] FAISS index not found at {faiss_index_path}. Creating new one...")
            from src.embedder import TextEmbedder
            loader = TextEmbedder()
            chunks, embeddings = loader.load_embeddings()
            self.vectorstore = FAISS.from_texts(chunks, self.embedder)
            self.vectorstore.save_local(faiss_index_path)
        else:
            self.vectorstore = FAISS.load_local(
                faiss_index_path,
                self.embedder,
                allow_dangerous_deserialization=True
            )

        # Optional memory to retain chat history
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

        # Build LangChain's Conversational RAG pipeline
        self.qa_chain = ConversationalRetrievalChain.from_llm(
          llm=self.llm,
          retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
          memory=self.memory,
          return_source_documents=True,
          output_key="answer"
      )

    def chat(self, query: str) -> Tuple[str, List[str]]:
        """
        Generate answer and return source contexts.

        Returns:
            Tuple of (answer, list of retrieved context texts)
        """
        result = self.qa_chain({"question": query})
        answer = result["answer"]
        contexts = [doc.page_content for doc in result.get("source_documents", [])]
        return answer, contexts
