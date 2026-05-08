from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

from .embeddings import get_embeddings

INDEX_DIR = Path("index")


def load_retriever(pdf_stem: str, k: int = 4) -> VectorStoreRetriever:
    index_path = INDEX_DIR / pdf_stem
    if not index_path.exists():
        raise FileNotFoundError(
            f"Indice no encontrado: {index_path}\n"
            f"Ejecuta primero: python rag.py index <pdf>"
        )
    # allow_dangerous_deserialization: index files are written locally by this tool;
    # never load an index received from an untrusted source.
    vectorstore = FAISS.load_local(
        str(index_path), get_embeddings(), allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})
