"""Definicion del estado compartido entre los nodos del grafo de consulta RAG."""

from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict):
    """Estado que fluye a traves del grafo de LangGraph durante una consulta.

    Attributes:
        question: pregunta original del usuario, en lenguaje natural.
        documents: fragmentos recuperados de PGVector, poblados por `retrieve_node`.
        answer: respuesta final generada por el LLM, poblada por `generate_node`.
    """

    question: str
    documents: list[Document]
    answer: str
