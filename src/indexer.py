import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .embeddings import get_embeddings

INDEX_DIR = Path("index")
MAX_PDF_BYTES = 100 * 1024 * 1024  # 100 MB


def index_pdf(pdf_path: Path) -> None:
    if pdf_path.stat().st_size > MAX_PDF_BYTES:
        size_mb = pdf_path.stat().st_size / 1024 / 1024
        raise ValueError(f"PDF too large: {size_mb:.1f} MB (max 100 MB)")

    start = time.time()
    reader = PdfReader(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    docs: list[Document] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        for chunk_num, chunk in enumerate(splitter.split_text(text), start=1):
            docs.append(Document(
                page_content=chunk,
                metadata={"page": page_num, "chunk": chunk_num},
            ))

    if not docs:
        raise ValueError("No se pudo extraer texto del PDF.")

    vectorstore = FAISS.from_documents(docs, get_embeddings())

    # Path(stem).name strips residual separators to prevent path traversal
    safe_stem = Path(pdf_path.stem).name
    out_dir = INDEX_DIR / safe_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(out_dir))

    elapsed = time.time() - start
    print(f"Paginas: {len(reader.pages)} | Chunks: {len(docs)} | Tiempo: {elapsed:.1f}s")
