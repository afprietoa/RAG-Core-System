"""Pruebas unitarias de la construccion de prompts para el nodo de generacion."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from rag_system.graph.prompts import construir_mensajes_generacion

pytestmark = pytest.mark.unit


def test_construir_mensajes_incluye_contexto_y_pregunta() -> None:
    documentos = [
        Document(page_content="Los gatos son mamiferos.", metadata={"source_file": "animales.txt"})
    ]

    mensajes = construir_mensajes_generacion("Que son los gatos?", documentos)

    assert len(mensajes) == 2
    assert isinstance(mensajes[0], SystemMessage)
    assert isinstance(mensajes[1], HumanMessage)
    assert "Los gatos son mamiferos." in mensajes[1].content
    assert "animales.txt" in mensajes[1].content
    assert "Que son los gatos?" in mensajes[1].content


def test_construir_mensajes_sin_contexto_indica_ausencia() -> None:
    mensajes = construir_mensajes_generacion("Alguna pregunta?", [])

    assert "No se recupero contexto relevante" in mensajes[1].content


def test_system_prompt_instruye_no_alucinar() -> None:
    mensajes = construir_mensajes_generacion("pregunta", [])

    contenido_sistema = mensajes[0].content
    assert "No tengo suficiente informacion" in contenido_sistema
