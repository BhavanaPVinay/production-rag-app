import os
from typing import Any, Dict, List, Optional
import chromadb
from sentence_transformers import SentenceTransformer
import re


class RAGPipeline:
    """Core RAG engine managing embedding generation, vector storage, and similarity retrieval."""
    

    def filter_by_similarity(
        self,
        contexts: List[Dict[str, Any]],
        min_similarity: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """Keeps only retrieved chunks that meet the minimum similarity threshold."""
        return [item for item in contexts if item.get("similarity", 0.0) >= min_similarity]

    def _extractive_fallback_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Offline-safe fallback that selects the most relevant sentences from top chunks."""
        query_terms = {
            token.lower()
            for token in re.findall(r"[a-zA-Z0-9]+", query)
            if len(token) > 2
        }

        scored_sentences: List[tuple[int, str]] = []
        for ctx in contexts:
            doc_text = ctx.get("document", "")
            sentences = re.split(r"(?<=[.!?])\s+", doc_text.strip())
            for sent in sentences:
                sent_clean = sent.strip()
                if not sent_clean:
                    continue
                sent_tokens = {
                    token.lower()
                    for token in re.findall(r"[a-zA-Z0-9]+", sent_clean)
                    if len(token) > 2
                }
                overlap = len(query_terms & sent_tokens)
                scored_sentences.append((overlap, sent_clean))

        scored_sentences.sort(key=lambda pair: pair[0], reverse=True)
        top_sentences = [sent for score, sent in scored_sentences[:2] if score > 0]

        if not top_sentences and contexts:
            top_sentences = [contexts[0]["document"].strip()]

        return " ".join(top_sentences) if top_sentences else "Insufficient context to answer the question."

    def check_grounding(self, answer: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Checks what fraction of content words in the answer appear in the retrieved contexts."""
        context_corpus = " ".join(item.get("document", "") for item in contexts).lower()
        context_tokens = set(re.findall(r"[a-zA-Z0-9]+", context_corpus))

        stop_words = {
            "the", "and", "for", "with", "that", "this", "from", "into",
            "are", "was", "were", "can", "has", "have", "its", "via", "based"
        }
        answer_tokens = [
            tok.lower()
            for tok in re.findall(r"[a-zA-Z0-9]+", answer)
            if len(tok) > 2 and tok.lower() not in stop_words
        ]

        if not answer_tokens:
            return {"grounded_ratio": 0.0, "is_grounded": False, "ungrounded_terms": []}

        grounded = [tok for tok in answer_tokens if tok in context_tokens]
        ungrounded = sorted(set(tok for tok in answer_tokens if tok not in context_tokens))
        ratio = round(len(grounded) / len(answer_tokens), 4)

        return {
            "grounded_ratio": ratio,
            "is_grounded": ratio >= 0.75,
            "ungrounded_terms": ungrounded,
        }

    def generate_grounded_answer(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.35,
    ) -> Dict[str, Any]:
        """Retrieves chunks, filters weak matches, synthesizes an answer, and checks grounding."""
        raw_contexts = self.retrieve(query=query, top_k=top_k)
        filtered_contexts = self.filter_by_similarity(raw_contexts, min_similarity=min_similarity)

        if not filtered_contexts:
            return {
                "query": query,
                "answer": "No sufficiently relevant context was found in the indexed documents.",
                "citations": [],
                "retrieved_contexts": [],
                "filtered_out_count": len(raw_contexts),
                "grounding": {
                    "grounded_ratio": 0.0,
                    "is_grounded": False,
                    "ungrounded_terms": [],
                },
            }

        answer_text = self._extractive_fallback_answer(query=query, contexts=filtered_contexts)
        grounding_report = self.check_grounding(answer=answer_text, contexts=filtered_contexts)

        citations = [
            {
                "source": item.get("metadata", {}).get("source", "unknown"),
                "chunk_index": item.get("metadata", {}).get("chunk_index"),
                "similarity": item.get("similarity"),
            }
            for item in filtered_contexts
        ]

        return {
            "query": query,
            "answer": answer_text,
            "citations": citations,
            "retrieved_contexts": filtered_contexts,
            "filtered_out_count": len(raw_contexts) - len(filtered_contexts),
            "grounding": grounding_report,
        }


    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents_collection",
    ):
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Load PyTorch SentenceTransformer model for dense vector embeddings
        self.model = SentenceTransformer(self.model_name)

        # Initialize persistent ChromaDB storage
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def get_document_count(self) -> int:
        """Returns the current number of documents stored in the ChromaDB collection."""
        return self.collection.count()

    def index_file(self, file_path: str) -> int:
        """Reads a text file, splits by double newline paragraphs, and indexes into ChromaDB."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source file not found at: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split into distinct paragraph chunks
        raw_chunks = content.split("\n\n")
        chunks = [c.strip() for c in raw_chunks if c.strip()]

        if not chunks:
            return 0

        # Unique document identifiers and metadata
        ids = [f"doc_{i+1}" for i in range(len(chunks))]
        metadatas = [
            {"source": os.path.basename(file_path), "chunk_index": i + 1}
            for i in range(len(chunks))
        ]

        # Generate 384-dimensional dense embeddings
        embeddings = self.model.encode(chunks, convert_to_numpy=True).tolist()

        # Upsert documents and their vector representations into ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Encodes user query and retrieves top-k most semantically relevant documents."""
        # Check if collection has documents
        if self.collection.count() == 0:
            return []

        # Ensure top_k does not exceed total items in collection
        actual_k = min(top_k, self.collection.count())
        query_embedding = self.model.encode([query], convert_to_numpy=True).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: List[Dict[str, Any]] = []
        if results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, distances):
                # Cosine distance ranges from 0 (identical) to 2 (opposite)
                similarity = max(0.0, 1.0 - dist) if dist is not None else 0.0
                retrieved.append({
                    "document": doc,
                    "metadata": meta,
                    "distance": round(float(dist), 4) if dist is not None else 0.0,
                    "similarity": round(float(similarity), 4),
                })

        return retrieved

    def query_and_augment(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Retrieves matching chunks and builds an augmented prompt for LLM consumption."""
        contexts = self.retrieve(query=query, top_k=top_k)

        # Assemble retrieved context strings
        context_block = "\n\n---\n\n".join([item["document"] for item in contexts])

        augmented_prompt = (
            "Context Information:\n"
            f"{context_block if context_block else 'No relevant context found.'}\n\n"
            f"User Question: {query}\n\n"
            "Answer the question accurately based strictly on the context provided above."
        )

        return {
            "query": query,
            "retrieved_contexts": contexts,
            "augmented_prompt": augmented_prompt,
        }

