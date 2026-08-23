from rag.retriever import get_vectorstore


question = "What are the criticisms of LangChain?"


vectorstore = get_vectorstore()


results = vectorstore.similarity_search_with_score(
    question,
    k=10
)


print("\n")
print("=" * 80)
print("RAW FAISS RESULTS")
print("=" * 80)


for i, (doc, score) in enumerate(
    results,
    start=1
):

    print(
        f"\nRESULT {i}"
    )

    print(
        f"Distance: {score:.4f}"
    )

    print(
        f"Source: "
        f"{doc.metadata.get('source')}"
    )

    print(
        f"Page: "
        f"{doc.metadata.get('page')}"
    )

    print(
        "\nContent:"
    )

    print(
        doc.page_content[:500]
    )

    print(
        "\n" + "-" * 80
    )