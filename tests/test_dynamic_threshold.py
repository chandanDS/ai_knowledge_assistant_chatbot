from rag.retriever import (
    get_vectorstore,
    calculate_dynamic_threshold
)


questions = [
    "What is LangChain architecture?",
    "What are the security concerns in LangChain?",
    "What is LangChain?",
    "Who won the 2023 Cricket World Cup?",
    "What is the capital of France?"
]


vectorstore = get_vectorstore()


for question in questions:

    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    results = vectorstore.similarity_search_with_score(
        question,
        k=10
    )

    threshold = calculate_dynamic_threshold(results)

    print(
        f"Best distance: {results[0][1]:.4f}"
        if results
        else "No results"
    )

    print(
        f"Dynamic threshold: {threshold}"
    )

    if threshold is None:
        print("Decision: NO RAG")
    else:
        relevant = [
            (doc, score)
            for doc, score in results
            if score <= threshold
        ]

        print(
            f"Decision: RAG"
        )

        print(
            f"Relevant documents: {len(relevant)}"
        )

        for i, (doc, score) in enumerate(
            relevant,
            1
        ):
            print(
                f"{i}. Distance={score:.4f}"
            )