"""Tests for rag-cli indexer and retriever modules."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_index_pdf_no_text():
    mock_reader = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = ""
    mock_reader.pages = [mock_page]
    mock_path = MagicMock(spec=Path)
    mock_path.stat.return_value.st_size = 1024
    mock_path.stem = "dummy"

    with patch("src.indexer.PdfReader", return_value=mock_reader):
        from src.indexer import index_pdf
        with pytest.raises(ValueError, match="No se pudo extraer"):
            index_pdf(mock_path)


def test_index_pdf_too_large():
    mock_path = MagicMock(spec=Path)
    mock_path.stat.return_value.st_size = 200 * 1024 * 1024

    from src.indexer import index_pdf
    with pytest.raises(ValueError, match="too large"):
        index_pdf(mock_path)


def test_load_retriever_missing_index():
    from src.retriever import load_retriever
    with pytest.raises(FileNotFoundError, match="Indice no encontrado"):
        load_retriever("nonexistent_pdf_xyz123")
