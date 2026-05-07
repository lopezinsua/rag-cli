import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

INDEX_DIR = Path("index")


def index_pdf(pdf_path: Path) -> None:
    start = time.time()
    reader = PdfReader(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

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

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    safe_stem = Path(pdf_path.stem).name  # strip any residual path separators
    out_dir = INDEX_DIR / safe_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(out_dir))

    elapsed = time.time() - start
    print(f"Páginas: {len(reader.pages)} | Chunks: {len(docs)} | Tiempo: {elapsed:.1f}s")
