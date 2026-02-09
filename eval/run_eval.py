"""
RAG Evaluation Harness

Runs a fixed set of Q&A pairs through the RAG engine and scores each response
for faithfulness (grounded in context) and relevance (answers the question).
Uses LLM-as-judge with gpt-4o-mini for cost efficiency.

Usage:
    uv run python eval/run_eval.py
"""

import json
import sys
from pathlib import Path

# Add project root so we can import rag
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI

from rag import RAGEngine


FAITHFULNESS_PROMPT = """You are evaluating whether an AI answer is faithful to the provided context.

Context:
{context}

Question: {question}

Answer: {answer}

Does the answer contain ANY information that is not present in or directly inferable from the context?
Answer with exactly one word: Yes or No"""

RELEVANCE_PROMPT = """You are evaluating whether an AI answer relevantly addresses the question.

Question: {question}

Answer: {answer}

Does the answer directly address the question in a relevant way?
Answer with exactly one word: Yes or No"""


def load_eval_set(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def score_faithfulness(judge: ChatOpenAI, context: str, question: str, answer: str) -> bool:
    if not context or not answer:
        return False
    prompt = FAITHFULNESS_PROMPT.format(
        context=context, question=question, answer=answer
    )
    resp = judge.invoke([{"role": "user", "content": prompt}])
    return resp.content.strip().upper().startswith("NO")


def score_relevance(judge: ChatOpenAI, question: str, answer: str) -> bool:
    if not answer:
        return False
    prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
    resp = judge.invoke([{"role": "user", "content": prompt}])
    return resp.content.strip().upper().startswith("YES")


def main() -> None:
    eval_path = Path(__file__).parent / "eval_set.json"
    if not eval_path.exists():
        print(f"Error: {eval_path} not found")
        sys.exit(1)

    eval_set = load_eval_set(eval_path)
    rag = RAGEngine()
    judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    faithful_count = 0
    relevant_count = 0
    used_llm_count = 0
    results = []

    print(f"Running eval on {len(eval_set)} questions...\n")

    for i, item in enumerate(eval_set):
        question = item["question"]
        expected = item.get("expected_answer", "(no ground truth)")
        result = rag.answer(question, return_context=True)

        answer = result["answer"]
        context = result.get("context", "")
        used_llm = result.get("used_llm", False)

        if used_llm:
            used_llm_count += 1
            faithful = score_faithfulness(judge, context, question, answer)
            relevant = score_relevance(judge, question, answer)
            if faithful:
                faithful_count += 1
            if relevant:
                relevant_count += 1
        else:
            faithful = False
            relevant = False

        results.append(
            {
                "question": question,
                "expected": expected,
                "answer": answer[:200] + "..." if len(answer) > 200 else answer,
                "used_llm": used_llm,
                "faithful": faithful,
                "relevant": relevant,
            }
        )

        status = "✓" if (faithful and relevant) else "✗"
        print(f"  [{i + 1}/{len(eval_set)}] {status} {question[:50]}...")

    # Summary
    n = len(eval_set)
    faithful_pct = (faithful_count / used_llm_count * 100) if used_llm_count else 0
    relevant_pct = (relevant_count / used_llm_count * 100) if used_llm_count else 0

    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    print(f"  Total questions:       {n}")
    print(f"  Answered (LLM used):   {used_llm_count} ({used_llm_count / n * 100:.0f}%)")
    print(f"  Faithfulness:          {faithful_count}/{used_llm_count} ({faithful_pct:.0f}%)")
    print(f"  Relevance:             {relevant_count}/{used_llm_count} ({relevant_pct:.0f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
