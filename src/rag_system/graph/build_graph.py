"""Ensamblado y compilacion del StateGraph que orquesta el flujo de consulta RAG."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from rag_system.graph.nodes import generate_node, retrieve_node
from rag_system.graph.state import RAGState


def construir_grafo() -> Any:
    """Ensambla el grafo lineal retrieve -> generate y lo compila.

    Returns:
        Grafo compilado, invocable con `.invoke({"question": ..., "documents": [], "answer": ""})`.
    """
    grafo = StateGraph(RAGState)

    grafo.add_node("retrieve", retrieve_node)
    grafo.add_node("generate", generate_node)

    grafo.add_edge(START, "retrieve")
    grafo.add_edge("retrieve", "generate")
    grafo.add_edge("generate", END)

    return grafo.compile()


def ejecutar_consulta(pregunta: str) -> RAGState:
    """Interfaz publica del flujo de consulta: ejecuta el grafo completo para una pregunta.

    Args:
        pregunta: pregunta del usuario en lenguaje natural.

    Returns:
        `RAGState` final, con `documents` (contexto recuperado) y `answer`
        (respuesta generada) poblados.
    """
    grafo = construir_grafo()
    estado_inicial: RAGState = {"question": pregunta, "documents": [], "answer": ""}
    return grafo.invoke(estado_inicial)
