from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_DIR = Path("index")


def load_retriever(pdf_stem: str, k: int = 4) -> VectorStoreRetriever:
    index_path = INDEX_DIR / pdf_stem
    if not index_path.exists():
        raise FileNotFoundError(
            f"Índice no encontrado: {index_path}\n"
            f"Ejecuta primero: python rag.py index <pdf>"
        )
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        str(index_path), embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})
