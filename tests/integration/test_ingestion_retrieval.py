"""Prueba de integracion: ingesta real seguida de recuperacion, contra PGVector en Docker."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag_system.ingestion.embedder import almacenar_chunks
from rag_system.ingestion.splitter import dividir_documentos
from rag_system.retrieval.retriever import recuperar_contexto

pytestmark = pytest.mark.integration


def test_ingesta_y_recuperacion_end_to_end() -> None:
    documento = Document(
        page_content=(
            "PGVector es una extension de PostgreSQL que permite almacenar y "
            "consultar embeddings vectoriales de forma eficiente usando indices "
            "como IVFFlat o HNSW."
        ),
        metadata={"source_file": "pgvector_intro_test.txt"},
    )

    chunks = dividir_documentos([documento])
    ids_insertados = almacenar_chunks(chunks)
    assert len(ids_insertados) == len(chunks)

    resultados = recuperar_contexto("Que es PGVector?", k=1)

    assert len(resultados) == 1
    assert "PostgreSQL" in resultados[0].page_content
    assert resultados[0].metadata["source_file"] == "pgvector_intro_test.txt"


def test_recuperar_contexto_con_pregunta_vacia_lanza_error() -> None:
    with pytest.raises(ValueError):
        recuperar_contexto("   ", k=1)
