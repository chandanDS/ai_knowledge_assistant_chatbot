from rag.retriever import get_vectorstore


query_pairs = [
    (
        "What are the criticisms of LangChain?",
        "LangChain limitations criticisms complexity security"
    ),
    (
        "What is LangChain architecture?",
        "LangChain architecture components modular design"
    ),
    (
        "What are the security concerns in LangChain?",
        "LangChain security risks external integrations data exposure"
    ),
    (
        "How does LangChain simplify LLM application development?",
        "LangChain LLM application development components framework"
    ),
]


vectorstore = get_vectorstore()


for original_query, expanded_query in query_pairs:

    print("\n" + "=" * 80)

    print(
        f"ORIGINAL QUERY:\n{original_query}"
    )

    print(
        f"\nEXPANDED QUERY:\n{expanded_query}"
    )

    print("=" * 80)

    results = vectorstore.similarity_search_with_score(
        expanded_query,
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
            f"Page = {doc.metadata.get('page')}"
        )

        print(
            doc.page_content[:250]
            .replace("\n", " ")
        )