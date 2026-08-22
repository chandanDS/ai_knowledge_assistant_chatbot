import json

from rag.retriever import similarity_search


# ---------------------------------------------------------
# Load evaluation questions
# ---------------------------------------------------------

with open(
    "test_questions.json",
    "r",
    encoding="utf-8"
) as f:

    test_questions = json.load(f)


# ---------------------------------------------------------
# Evaluation counters
# ---------------------------------------------------------

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0


# ---------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------

for item in test_questions:

    question = item["question"]
    expected_relevant = item["expected_relevant"]

    results = similarity_search(question)

    retrieved_relevant = len(results) > 0


    # -----------------------------------------------------
    # Determine outcome
    # -----------------------------------------------------

    if expected_relevant and retrieved_relevant:

        true_positive += 1
        result = "TP"

    elif not expected_relevant and not retrieved_relevant:

        true_negative += 1
        result = "TN"

    elif not expected_relevant and retrieved_relevant:

        false_positive += 1
        result = "FP"

    else:

        false_negative += 1
        result = "FN"


    # -----------------------------------------------------
    # Print result
    # -----------------------------------------------------

    print("\n" + "=" * 80)

    print(
        f"Question: {question}"
    )

    print(
        f"Expected relevant: "
        f"{expected_relevant}"
    )

    print(
        f"Retrieved documents: "
        f"{len(results)}"
    )

    print(
        f"Result: {result}"
    )


# ---------------------------------------------------------
# Calculate metrics
# ---------------------------------------------------------

total = (
    true_positive
    + true_negative
    + false_positive
    + false_negative
)

accuracy = (
    (true_positive + true_negative) / total
    if total > 0
    else 0
)

precision = (
    true_positive
    / (true_positive + false_positive)
    if (true_positive + false_positive) > 0
    else 0
)

recall = (
    true_positive
    / (true_positive + false_negative)
    if (true_positive + false_negative) > 0
    else 0
)


# ---------------------------------------------------------
# Print summary
# ---------------------------------------------------------

print("\n")
print("=" * 80)
print("RETRIEVAL EVALUATION SUMMARY")
print("=" * 80)

print(
    f"True Positive  : {true_positive}"
)

print(
    f"True Negative  : {true_negative}"
)

print(
    f"False Positive : {false_positive}"
)

print(
    f"False Negative : {false_negative}"
)

print(
    f"Accuracy       : {accuracy:.2%}"
)

print(
    f"Precision      : {precision:.2%}"
)

print(
    f"Recall         : {recall:.2%}"
)

print("=" * 80)