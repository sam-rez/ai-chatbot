# chat.py
from rag import RAGEngine

def main():
    rag = RAGEngine()

    while True:
        question = input("\nAsk a question (or 'exit'): ").strip()
        if question.lower() == "exit":
            break

        result = rag.answer(question)
        print("\nAnswer:", result["answer"])

if __name__ == "__main__":
    main()
