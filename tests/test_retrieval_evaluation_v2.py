from rag.retriever import similarity_search_with_query_expansion


TEST_CASES = [

    # =========================
    # RELEVANT QUESTIONS
    # =========================

    {
        "question": "What is LangChain?",
        "expected": True
    },

    {
        "question": "What is LangChain architecture?",
        "expected": True
    },

    {
        "question": "What are the components of LangChain?",
        "expected": True
    },

    {
        "question": "What are the criticisms of LangChain?",
        "expected": True
    },

    {
        "question": "What are the limitations of LangChain?",
        "expected": True
    },

    {
        "question": "What are the security concerns in LangChain?",
        "expected": True
    },

    {
        "question": "How does LangChain simplify LLM application development?",
        "expected": True
    },

    {
        "question": "What is LangGraph?",
        "expected": True
    },

    {
        "question": "What is LangServe?",
        "expected": True
    },

    {
        "question": "What is LangSmith?",
        "expected": True
    },


    # =========================
    # NON-RELEVANT QUESTIONS
    # =========================

    {
        "question": "Who won the 2023 Cricket World Cup?",
        "expected": False
    },

    {
        "question": "What is the capital of France?",
        "expected": False
    },

    {
        "question": "What is the population of India?",
        "expected": False
    },

    {
        "question": "What is the weather today?",
        "expected": False
    },

    {
        "question": "Who is the Prime Minister of India?",
        "expected": False
    },

    {
        "question": "What is the GDP of India?",
        "expected": False
    },

    {
        "question": "What is the price of gold today?",
        "expected": False
    },

    {
        "question": "Who won the FIFA World Cup?",
        "expected": False
    },

    {
        "question": "What is the capital of Japan?",
        "expected": False
    },

    {
        "question": "How many states are there in India?",
        "expected": False
    }
]


TP = 0
TN = 0
FP = 0
FN = 0


for test in TEST_CASES:

    question = test["question"]
    expected = test["expected"]

    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    expanded_query, results = (
        similarity_search_with_query_expansion(
            question
        )
    )

    retrieved = len(results) > 0

    print(f"Expanded query: {expanded_query}")
    print(f"Retrieved documents: {len(results)}")

    if results:

        print(
            f"Best distance: {results[0][1]:.4f}"
        )

    # =========================
    # CONFUSION MATRIX
    # =========================

    if expected and retrieved:

        TP += 1
        result = "TP"

    elif not expected and not retrieved:

        TN += 1
        result = "TN"

    elif not expected and retrieved:

        FP += 1
        result = "FP"

    else:

        FN += 1
        result = "FN"

    print(f"Expected relevant: {expected}")
    print(f"Result: {result}")


# =========================
# METRICS
# =========================

total = TP + TN + FP + FN

accuracy = (
    (TP + TN) / total
    if total > 0
    else 0
)

precision = (
    TP / (TP + FP)
    if (TP + FP) > 0
    else 0
)

recall = (
    TP / (TP + FN)
    if (TP + FN) > 0
    else 0
)

f1 = (
    2 * precision * recall
    / (precision + recall)
    if (precision + recall) > 0
    else 0
)


print("\n")
print("=" * 80)
print("RETRIEVAL EVALUATION V2")
print("=" * 80)

print(f"True Positive  : {TP}")
print(f"True Negative  : {TN}")
print(f"False Positive : {FP}")
print(f"False Negative : {FN}")

print(f"Accuracy       : {accuracy:.2%}")
print(f"Precision      : {precision:.2%}")
print(f"Recall         : {recall:.2%}")
print(f"F1 Score       : {f1:.2%}")