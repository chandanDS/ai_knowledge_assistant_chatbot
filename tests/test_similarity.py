from rag.retriever import similarity_search


question = input(
    "Enter your question: "
)


results = similarity_search(
    question
)


print("\n")
print("=" * 80)

print(
    f"Relevant documents: {len(results)}"
)

print("=" * 80)


for i, (doc, score) in enumerate(
    results,
    start=1
):

    print(
        f"\nRESULT {i}"
    )

    print(
        f"Distance score: {score:.4f}"
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
        f"Chunk ID: "
        f"{doc.metadata.get('chunk_id')}"
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