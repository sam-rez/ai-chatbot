from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def main():
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        "index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = ChatOpenAI(model="gpt-4o-mini")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever()
    )

    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() == "exit":
            break

        answer = qa.run(query)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()
