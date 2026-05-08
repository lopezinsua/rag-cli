from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_load_retriever_missing_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.retriever import load_retriever
    with pytest.raises(FileNotFoundError, match="Indice no encontrado"):
        load_retriever("nonexistent_pdf")


def test_load_retriever_returns_retriever(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    index_dir = tmp_path / "index" / "my_doc"
    index_dir.mkdir(parents=True)

    mock_vectorstore = MagicMock()
    mock_retriever = MagicMock()
    mock_vectorstore.as_retriever.return_value = mock_retriever

    with patch("src.retriever.FAISS.load_local", return_value=mock_vectorstore), \
         patch("src.retriever.get_embeddings", return_value=MagicMock()):
        from src.retriever import load_retriever
        result = load_retriever("my_doc", k=3)

    mock_vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 3})
    assert result is mock_retriever


def test_load_retriever_default_k(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    index_dir = tmp_path / "index" / "my_doc"
    index_dir.mkdir(parents=True)

    mock_vectorstore = MagicMock()
    with patch("src.retriever.FAISS.load_local", return_value=mock_vectorstore), \
         patch("src.retriever.get_embeddings", return_value=MagicMock()):
        from src.retriever import load_retriever
        load_retriever("my_doc")

    mock_vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 4})
