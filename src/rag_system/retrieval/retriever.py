"""Recuperacion de contexto relevante desde PGVector para una pregunta dada."""

from __future__ import annotations

from langchain_core.documents import Document

from rag_system.config import get_settings
from rag_system.ingestion.embedder import obtener_vector_store


def recuperar_contexto(pregunta: str, k: int | None = None) -> list[Document]:
    """Busca los `k` fragmentos mas similares a `pregunta` en PGVector.

    Args:
        pregunta: texto de la consulta del usuario.
        k: numero de fragmentos a recuperar; si se omite, usa
            `RETRIEVAL_TOP_K` de la configuracion (config.py).

    Returns:
        Lista de `Document`, cada uno con `metadata["similarity_score"]`
        anadido (distancia: valores mas bajos indican mayor similitud,
        segun la metrica configurada por defecto en PGVector: distancia coseno).
    """
    if not pregunta or not pregunta.strip():
        raise ValueError("La pregunta no puede estar vacia")

    settings = get_settings()
    top_k = k or settings.retrieval_top_k

    vector_store = obtener_vector_store()
    resultados = vector_store.similarity_search_with_score(pregunta, k=top_k)

    documentos: list[Document] = []
    for documento, score in resultados:
        documento.metadata["similarity_score"] = round(float(score), 4)
        documentos.append(documento)

    return documentos
