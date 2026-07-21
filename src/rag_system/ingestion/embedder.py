"""Generacion de embeddings y almacenamiento de fragmentos en PGVector."""

from __future__ import annotations

import hashlib

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector

from rag_system.config import get_settings


def obtener_embeddings() -> Embeddings:
    """Instancia el modelo de embeddings segun `EMBEDDING_PROVIDER` (config.py).

    Los imports de cada proveedor estan dentro de cada rama para no forzar la
    instalacion de dependencias de proveedores que no se usan.
    """
    settings = get_settings()

    if settings.embedding_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.embedding_model)

    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model)

    raise ValueError(
        f"EMBEDDING_PROVIDER no soportado: {settings.embedding_provider!r} "
        "(valores validos: 'huggingface', 'openai')"
    )


def obtener_vector_store() -> PGVector:
    """Construye el vector store PGVector con el modelo de embeddings configurado.

    Usado tanto por la ingesta (para escribir) como por el retriever (para
    leer) — misma coleccion, mismo modelo de embeddings, para garantizar que
    las dimensiones de los vectores sean consistentes.
    """
    settings = get_settings()
    embeddings = obtener_embeddings()

    return PGVector(
        embeddings=embeddings,
        collection_name=settings.pgvector_collection,
        connection=settings.database_url,
        use_jsonb=True,
    )


def _id_determinista(chunk: Document) -> str:
    """Genera un ID estable a partir de la fuente, el indice y el contenido.

    Re-ingerir el mismo archivo con el mismo contenido produce los mismos IDs,
    por lo que `PGVector.add_documents` actualiza los registros existentes en
    vez de crear duplicados.
    """
    fuente = chunk.metadata.get("source_file", "desconocido")
    indice = chunk.metadata.get("chunk_index", 0)
    huella = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()[:16]
    return f"{fuente}:{indice}:{huella}"


def almacenar_chunks(chunks: list[Document]) -> list[str]:
    """Genera embeddings para los chunks y los almacena en PGVector.

    Returns:
        Lista de IDs (deterministas) de los documentos insertados/actualizados.

    Raises:
        ValueError: si `chunks` esta vacio.
    """
    if not chunks:
        raise ValueError("No hay chunks para almacenar")

    vector_store = obtener_vector_store()
    ids = [_id_determinista(chunk) for chunk in chunks]

    return vector_store.add_documents(chunks, ids=ids)
