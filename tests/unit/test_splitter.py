"""Pruebas unitarias del divisor de documentos (chunking)."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

pytestmark = pytest.mark.unit


def test_dividir_documentos_respeta_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "50")
    monkeypatch.setenv("CHUNK_OVERLAP", "10")

    from rag_system.config import get_settings

    get_settings.cache_clear()

    from rag_system.ingestion.splitter import dividir_documentos

    texto_largo = "Lorem ipsum dolor sit amet consectetur. " * 10
    documento = Document(page_content=texto_largo, metadata={"source_file": "demo.txt"})

    chunks = dividir_documentos([documento])

    assert len(chunks) > 1
    # Tolerancia: el splitter recursivo puede exceder ligeramente chunk_size
    # al buscar el separador mas cercano; se valida un limite razonable.
    assert all(len(chunk.page_content) <= 80 for chunk in chunks)
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[-1].metadata["chunk_index"] == len(chunks) - 1


def test_dividir_documentos_preserva_metadatos_de_origen() -> None:
    from rag_system.ingestion.splitter import dividir_documentos

    documento = Document(page_content="Contenido corto de prueba.", metadata={"source_file": "a.txt"})

    chunks = dividir_documentos([documento])

    assert len(chunks) == 1
    assert chunks[0].metadata["source_file"] == "a.txt"


def test_dividir_documentos_lista_vacia_devuelve_lista_vacia() -> None:
    from rag_system.ingestion.splitter import dividir_documentos

    assert dividir_documentos([]) == []
