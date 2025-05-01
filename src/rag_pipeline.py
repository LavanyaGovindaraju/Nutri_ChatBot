# src/rag_pipeline.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.vector_store import VectorStore

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline with optional FAISS-based vector store
    and Hugging Face LLMs for answer generation.
    """

    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.1", use_faiss=True, faiss_index_path="faiss.index"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_faiss = use_faiss
        self.faiss_index_path = faiss_index_path
        self.vector_store = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def init_vector_store(self, doc_embeddings, doc_texts):
        dim = doc_embeddings.shape[1]
        store = VectorStore(embedding_dim=dim)

        if self.faiss_index_path and not store.texts:
            try:
                store.load(self.faiss_index_path, self.faiss_index_path + ".txt")
            except:
                store.add_tensor(doc_embeddings, doc_texts)
                store.save(self.faiss_index_path, self.faiss_index_path + ".txt")
        else:
            store.add_tensor(doc_embeddings, doc_texts)

        self.vector_store = store

    def retrieve(self, query_embedding, doc_embeddings, doc_texts, top_k=3):
        if self.use_faiss:
            if self.vector_store is None:
                self.init_vector_store(doc_embeddings, doc_texts)
            return [text for text, _ in self.vector_store.search_tensor(query_embedding, top_k)]
        else:
            from sentence_transformers.util import dot_score
            scores = dot_score(query_embedding, doc_embeddings)[0]
            top_indices = torch.topk(scores, k=top_k).indices.tolist()
            return [doc_texts[i] for i in top_indices]

    def format_prompt(self, query, contexts):
        context_block = "\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
        prompt = (
            f"You are a helpful assistant.\n"
            f"Use the following context to answer the question concisely.\n"
            f"{context_block}\n"
            f"\nQuestion: {query}\nAnswer:"
        )
        return prompt

    def generate_answer(self, query, query_embedding, doc_embeddings, doc_texts, top_k=3):
        contexts = self.retrieve(query_embedding, doc_embeddings, doc_texts, top_k)
        prompt = self.format_prompt(query, contexts)
        output = self.generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)[0]['generated_text']
        return output.replace(prompt, "").strip()
