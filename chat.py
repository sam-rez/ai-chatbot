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


def main():
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        "index",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

    while True:
        question = input("\nAsk a question (or 'exit'): ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        docs = retriever.invoke(question)
        context = format_context(docs)

        messages = [
            {
                "role": "system", 
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]

        resp = llm.invoke(messages)
        print("\nAnswer:", resp.content)


if __name__ == "__main__":
    main()
