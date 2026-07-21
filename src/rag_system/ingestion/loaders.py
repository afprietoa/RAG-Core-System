"""Carga de documentos desde disco (.txt, .md, .pdf) para el flujo de ingesta."""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

_LOADER_POR_EXTENSION = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PyPDFLoader,
}


def cargar_documentos(ruta: str | Path) -> list[Document]:
    """Carga uno o varios documentos soportados desde un archivo o una carpeta.

    Args:
        ruta: ruta a un archivo individual o a una carpeta (se recorre recursivamente).

    Returns:
        Lista de `Document` de LangChain, cada uno con `metadata["source_file"]`
        indicando el nombre del archivo de origen.

    Raises:
        FileNotFoundError: si la ruta no existe.
        ValueError: si no se encuentra ningún documento soportado, o si un
            documento encontrado está vacío.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"La ruta no existe: {ruta}")

    if ruta.is_file():
        return _cargar_archivo(ruta)

    documentos: list[Document] = []
    for archivo in sorted(ruta.rglob("*")):
        if archivo.is_file() and archivo.suffix.lower() in _LOADER_POR_EXTENSION:
            documentos.extend(_cargar_archivo(archivo))

    if not documentos:
        extensiones = ", ".join(sorted(_LOADER_POR_EXTENSION))
        raise ValueError(
            f"No se encontraron documentos soportados ({extensiones}) en: {ruta}"
        )

    return documentos


def _cargar_archivo(archivo: Path) -> list[Document]:
    extension = archivo.suffix.lower()
    loader_cls = _LOADER_POR_EXTENSION.get(extension)
    if loader_cls is None:
        extensiones = ", ".join(sorted(_LOADER_POR_EXTENSION))
        raise ValueError(
            f"Extension no soportada: {extension} ({archivo}). "
            f"Extensiones validas: {extensiones}"
        )

    loader = loader_cls(str(archivo), encoding="utf-8") if loader_cls is TextLoader else loader_cls(str(archivo))
    documentos = loader.load()

    if not any(doc.page_content.strip() for doc in documentos):
        raise ValueError(f"El documento esta vacio: {archivo}")

    for doc in documentos:
        doc.metadata["source_file"] = archivo.name

    return documentos
