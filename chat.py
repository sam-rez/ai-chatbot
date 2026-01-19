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
    # load embeddings
    embeddings = OpenAIEmbeddings()

    # load FAISS from disk
    vector_store = FAISS.load_local(
        "index",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

    while True:
        # Wait for user input
        question = input("\nAsk a question (or 'exit'): ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        # Get relevant documents
        docs = retriever.invoke(question)

        # Build prompt with context
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

        # Call LLM
        resp = llm.invoke(messages)

        # Print answer
        print("\nAnswer:", resp.content)


if __name__ == "__main__":
    main()
