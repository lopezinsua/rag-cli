#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _validated_pdf(path_str: str) -> Path:
    pdf = Path(path_str)
    if pdf.suffix.lower() != ".pdf":
        sys.exit(f"Error: '{pdf.name}' no es un archivo PDF.")
    if not pdf.exists():
        sys.exit(f"Error: '{pdf}' no existe.")
    return pdf


def cmd_index(args: argparse.Namespace) -> None:
    from src.indexer import index_pdf
    index_pdf(_validated_pdf(args.pdf))


def cmd_chat(args: argparse.Namespace) -> None:
    from src.indexer import index_pdf
    from src.chain import chat_loop
    pdf = _validated_pdf(args.pdf)
    index_path = Path("index") / pdf.stem
    if not index_path.exists():
        print(f"Índice no encontrado. Indexando '{pdf.name}'...")
        index_pdf(pdf)
    chat_loop(pdf)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG CLI — Chatea con tus PDFs en la terminal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python rag.py index apuntes.pdf\n"
            "  python rag.py chat apuntes.pdf"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Indexar un PDF")
    p_index.add_argument("pdf", help="Ruta al archivo PDF")

    p_chat = sub.add_parser("chat", help="Chatear con un PDF indexado")
    p_chat.add_argument("pdf", help="Ruta al archivo PDF")

    args = parser.parse_args()
    {"index": cmd_index, "chat": cmd_chat}[args.command](args)


if __name__ == "__main__":
    main()
