"""Pruebas unitarias de los nodos del grafo, con retriever y LLM mockeados."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from rag_system.graph.nodes import generate_node, retrieve_node
from rag_system.graph.state import RAGState

pytestmark = pytest.mark.unit


def test_retrieve_node_agrega_documentos_al_estado() -> None:
    documentos_falsos = [Document(page_content="contenido", metadata={"source_file": "a.txt"})]
    estado: RAGState = {"question": "pregunta?", "documents": [], "answer": ""}

    with patch(
        "rag_system.graph.nodes.recuperar_contexto", return_value=documentos_falsos
    ) as mock_retriever:
        nuevo_estado = retrieve_node(estado)

    mock_retriever.assert_called_once_with("pregunta?")
    assert nuevo_estado["documents"] == documentos_falsos
    assert nuevo_estado["question"] == "pregunta?"  # el resto del estado no se pierde


def test_generate_node_usa_llm_mockeado_y_actualiza_answer() -> None:
    estado: RAGState = {
        "question": "pregunta?",
        "documents": [Document(page_content="contexto", metadata={})],
        "answer": "",
    }

    llm_falso = MagicMock()
    llm_falso.invoke.return_value = AIMessage(content="respuesta simulada")

    with patch("rag_system.graph.nodes.obtener_llm", return_value=llm_falso):
        nuevo_estado = generate_node(estado)

    assert nuevo_estado["answer"] == "respuesta simulada"
    llm_falso.invoke.assert_called_once()
