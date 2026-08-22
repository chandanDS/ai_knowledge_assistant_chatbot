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

    print("\n")
    print("=" * 80)
    print("QUESTION:", question)
    print("=" * 80)

    results = vectorstore.similarity_search_with_score(
        question,
        k=10
    )

    threshold = calculate_dynamic_threshold(results)

    print("Dynamic threshold:", threshold)

    print("\nRetrieved distances:")

    for i, (document, score) in enumerate(results, 1):

        status = (
            "KEEP"
            if score <= threshold
            else "REJECT"
        )

        print(
            f"{i}. "
            f"Distance={score:.4f} "
            f"→ {status}"
        )