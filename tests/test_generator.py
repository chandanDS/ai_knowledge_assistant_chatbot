import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.retriever import similarity_search
from rag.context_builder import build_context
from rag.generator import generate_answer


question = "What is LangChain architecture?"

print("=" * 80)
print(f"QUESTION: {question}")
print("=" * 80)

results = similarity_search(question)

print(f"\nRetrieved documents: {len(results)}")

context = build_context(results)

answer = generate_answer(
    question,
    context
)

print("\n" + "=" * 80)
print("FINAL ANSWER")
print("=" * 80)

print(answer)