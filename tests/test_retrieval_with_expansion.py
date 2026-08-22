from rag.retriever import (
    similarity_search_with_query_expansion
)


questions = [
    "What are the criticisms of LangChain?",
    "What is LangChain architecture?",
    "What are the security concerns in LangChain?",
    "How does LangChain simplify LLM application development?",
    "Who won the 2023 Cricket World Cup?",
    "What is the capital of France?"
]


for question in questions:

    print("\n" + "=" * 80)

    print(
        f"QUESTION: {question}"
    )

    print("=" * 80)

    expanded_query, results = (
        similarity_search_with_query_expansion(
            question
        )
    )

    print(
        f"\nExpanded query:"
        f"\n{expanded_query}"
    )

    print(
        f"\nRelevant documents:"
        f" {len(results)}"
    )

    for i, (doc, score) in enumerate(
        results,
        start=1
    ):

        print(
            f"\n{i}. Distance = {score:.4f}"
        )

        print(
            f"Page = "
            f"{doc.metadata.get('page')}"
        )

        print(
            doc.page_content[:300]
            .replace("\n", " ")
        )