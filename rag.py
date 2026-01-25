from typing import Dict, Any, List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

SYSTEM_PROMPT = """You are a helpful assistant.
Answer ONLY using the provided context.
If the answer is not in the context, say: "I don't know based on the provided documents."
"""


def format_context(docs) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "unknown")
        parts.append(f"[{i}] source={source}\n{d.page_content}")
    return "\n\n".join(parts)


class RAGEngine:
    # Tune these once you collect a few example queries
    TOP_K = 4
    # For FAISS with L2 distance (common default): lower = more similar.
    # If best_score is ABOVE this threshold, retrieval is probably too weak to answer.
    RETRIEVAL_SCORE_THRESHOLD = 0.35

    def __init__(self, index_path: str = "index"):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.load_local(
            index_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

    def _compute_confidence(self, best_score: float) -> float:
        """
        Converts a FAISS distance score to a 0..1 confidence.
        Assumes: lower distance = better match.
        """
        t = self.RETRIEVAL_SCORE_THRESHOLD
        # best_score <= 0 is extremely strong (cap at 1.0)
        if best_score <= 0:
            return 1.0
        # Map best_score=t -> 0.0, best_score=0 -> 1.0
        conf = 1.0 - (best_score / t)
        return max(0.0, min(1.0, conf))

    def answer(self, question: str) -> Dict[str, Any]:
        question = (question or "").strip()
        if not question:
            return {
                "answer": "I don't know based on the provided documents.",
                "citations": [],
                "confidence": 0.0,
                "used_llm": False,
                "retrieval_score": None,
            }

        docs_with_scores = self.vector_store.similarity_search_with_score(question, k=self.TOP_K)

        if not docs_with_scores:
            return {
                "answer": "I don't know based on the provided documents.",
                "citations": [],
                "confidence": 0.0,
                "used_llm": False,
                "retrieval_score": None,
            }

        docs = [d for d, _ in docs_with_scores]

        # Cast numpy.float32 -> float for JSON serialization
        best_score = float(min(score for _, score in docs_with_scores))
        confidence = float(self._compute_confidence(best_score))

        if best_score > self.RETRIEVAL_SCORE_THRESHOLD:
            return {
                "answer": "I don't know based on the provided documents.",
                "citations": [],
                "confidence": 0.0,
                "used_llm": False,
                "retrieval_score": best_score,
            }

        context = format_context(docs)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]

        resp = self.llm.invoke(messages)

        citations = [
            {
                "rank": i + 1,
                "source": d.metadata.get("source", "unknown"),
                "snippet": d.page_content[:240],
                "score": float(docs_with_scores[i][1]),  # cast here too
            }
            for i, d in enumerate(docs)
        ]

        return {
            "answer": resp.content,
            "citations": citations,
            "confidence": confidence,
            "used_llm": True,
            "retrieval_score": best_score,
        }
