"""Nodos del grafo de consulta RAG: recuperacion y generacion."""

from __future__ import annotations

from rag_system.graph.prompts import construir_mensajes_generacion
from rag_system.graph.state import RAGState
from rag_system.llm.provider import obtener_llm
from rag_system.retrieval.retriever import recuperar_contexto


def retrieve_node(state: RAGState) -> RAGState:
    """Nodo de recuperacion: busca los fragmentos mas relevantes para la pregunta."""
    documentos = recuperar_contexto(state["question"])
    return {**state, "documents": documentos}


def generate_node(state: RAGState) -> RAGState:
    """Nodo de generacion: produce la respuesta final usando el LLM y el contexto."""
    llm = obtener_llm()
    mensajes = construir_mensajes_generacion(state["question"], state["documents"])

    respuesta = llm.invoke(mensajes)

    return {**state, "answer": respuesta.content}
