# ============================================================
# rag/ingest.py
# ============================================================

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

DOCUMENTS_PATH = PROJECT_ROOT / "documents"

VECTORSTORE_PATH = (
    PROJECT_ROOT
    / "vector_store"
    / "faiss_index"
)


# ============================================================
# LOAD DOCUMENTS
# ============================================================

documents = []

for pdf_file in DOCUMENTS_PATH.glob("*.pdf"):

    print(
        f"Loading: {pdf_file.name}"
    )

    loader = PyPDFLoader(
        str(pdf_file)
    )

    documents.extend(
        loader.load()
    )


# ============================================================
# SPLIT DOCUMENTS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(
    documents
)

print(
    f"Documents loaded: {len(documents)}"
)

print(
    f"Chunks created: {len(chunks)}"
)


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# CREATE FAISS INDEX
# ============================================================

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)


# ============================================================
# SAVE FAISS INDEX
# ============================================================

VECTORSTORE_PATH.mkdir(
    parents=True,
    exist_ok=True
)

vectorstore.save_local(
    str(VECTORSTORE_PATH)
)


print(
    f"FAISS index successfully created at:"
)

print(
    VECTORSTORE_PATH
)