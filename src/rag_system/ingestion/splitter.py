"""Division de documentos en fragmentos (chunks) para generar embeddings."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_system.config import get_settings


def dividir_documentos(documentos: list[Document]) -> list[Document]:
    """Divide una lista de documentos en fragmentos segun CHUNK_SIZE / CHUNK_OVERLAP.

    Cada fragmento resultante conserva los metadatos del documento original y
    recibe ademas `metadata["chunk_index"]`, un indice secuencial usado por
    `embedder.py` para generar IDs deterministas.
    """
    if not documentos:
        return []

    settings = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documentos)

    for indice, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = indice

    return chunks
