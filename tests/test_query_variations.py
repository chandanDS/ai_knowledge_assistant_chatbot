from rag.retriever import get_vectorstore


queries = [
    "What are the criticisms of LangChain?",
    "What are the limitations and criticisms of LangChain?",
    "What are the limitations of LangChain?",
    "LangChain limitations complexity security",
]


vectorstore = get_vectorstore()


for question in queries:

    print("\n" + "=" * 80)

    print(
        f"QUERY: {question}"
    )

    print("=" * 80)

    results = vectorstore.similarity_search_with_score(
        question,
        k=5
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
            doc.page_content[:250]
            .replace("\n", " ")
        )