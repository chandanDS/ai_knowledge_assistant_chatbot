from rag.retriever import get_vectorstore


questions = [
    "What is LangChain architecture?",
    "What are the security concerns in LangChain?",
    "What are the criticisms of LangChain?",
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

    # Take top 5 without threshold
    final_results = results[:5]

    print(
        f"Top 5 documents returned: "
        f"{len(final_results)}"
    )

    for i, (doc, score) in enumerate(
        final_results,
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
            doc.page_content[:250]
            .replace("\n", " ")
        )