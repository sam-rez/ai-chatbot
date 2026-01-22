# rag.py
from typing import List, Dict, Any

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
    def __init__(self, index_path: str = "index"):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.load_local(
            index_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        self.llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

    def answer(self, question: str) -> Dict[str, Any]:
        docs = self.retriever.invoke(question)
        context = format_context(docs)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]

        resp = self.llm.invoke(messages)

        citations = [
            {
                "rank": i + 1,
                "source": d.metadata.get("source", "unknown"),
                "snippet": d.page_content[:240],
            }
            for i, d in enumerate(docs)
        ]

        return {
            "answer": resp.content,
            "citations": citations,
        }
