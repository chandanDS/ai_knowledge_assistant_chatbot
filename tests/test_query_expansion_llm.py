from rag.query_expansion import expand_query


questions = [
    "What are the criticisms of LangChain?",
    "What is LangChain architecture?",
    "What are the security concerns in LangChain?",
    "How does LangChain simplify LLM application development?",
    "Who won the 2023 Cricket World Cup?",
]


for question in questions:

    print("\n" + "=" * 80)

    print(
        f"Original:\n{question}"
    )

    expanded = expand_query(
        question
    )

    print(
        f"\nExpanded:\n{expanded}"
    )