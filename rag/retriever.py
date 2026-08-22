"""
RAG retrieval and FAISS vector-store management.
"""

from pathlib import Path
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# Configuration
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RETRIEVAL_K = 10

DEFAULT_DISTANCE_THRESHOLD = 0.75

FINAL_K = 5


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_STORE_DIR = (
    BASE_DIR / "vector_store" / "faiss_index"
)


# ============================================================
# Embeddings
# ============================================================

@lru_cache(maxsize=1)
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# ============================================================
# Vector Store
# ============================================================

@lru_cache(maxsize=1)
def get_vectorstore():

    embeddings = get_embeddings()

    vectorstore = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


# ============================================================
# Dynamic Threshold
# ============================================================

def calculate_dynamic_threshold(best_distance):

    if best_distance <= DEFAULT_DISTANCE_THRESHOLD:
        return DEFAULT_DISTANCE_THRESHOLD

    elif best_distance <= 0.85:
        return best_distance

    else:
        return None

# ============================================================
# Similarity Search
# ============================================================

def similarity_search(
    question,
    k=RETRIEVAL_K,
    threshold=None,
    final_k=FINAL_K,
):

    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search_with_score(
        question,
        k=k
    )

    if not results:
        return []

    best_distance = results[0][1]

    if threshold is None:
        threshold = calculate_dynamic_threshold(
            best_distance
        )

    if threshold is None:
        return []

    filtered_results = [
        (document, score)
        for document, score in results
        if score <= threshold
    ]

    return filtered_results[:final_k]


def get_retriever():
    """
    Return a LangChain retriever for basic similarity search.
    """

    vectorstore = get_vectorstore()

    return vectorstore.as_retriever(
        search_kwargs={
            "k": RETRIEVAL_K
        }
    )