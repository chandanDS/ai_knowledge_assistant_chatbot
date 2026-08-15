"""
RAG retrieval and vector-store management.
"""

from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFDirectoryLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import FAISS

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = (
    BASE_DIR / "documents"
)

VECTOR_STORE_DIR = (
    BASE_DIR / "vector_store" / "faiss_index"
)


EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


def create_vector_store():

    loader = PyPDFDirectoryLoader(
        str(DOCUMENTS_DIR)
    )

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(
        documents
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    VECTOR_STORE_DIR.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    vectorstore.save_local(
        str(VECTOR_STORE_DIR)
    )


def get_retriever():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore.as_retriever(
        search_kwargs={
            "k": 4
        }
    )