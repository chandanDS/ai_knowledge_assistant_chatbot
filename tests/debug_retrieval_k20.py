from rag.retriever import get_vectorstore


question = "What are the criticisms of LangChain?"


vectorstore = get_vectorstore()


results = vectorstore.similarity_search_with_score(
    question,
    k=20
)


print("\n")
print("=" * 80)
print("RAW FAISS TOP 20 RESULTS")
print("=" * 80)


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
        doc.page_content[:300]
        .replace("\n", " ")
    )