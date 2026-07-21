"""Orquestacion del flujo de ingesta: carga -> division -> embeddings -> almacenamiento."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from rag_system.ingestion.embedder import almacenar_chunks
from rag_system.ingestion.loaders import cargar_documentos
from rag_system.ingestion.splitter import dividir_documentos


@dataclass
class ReporteIngesta:
    """Resumen de una ejecucion de ingesta, usado para reportar resultados al usuario."""

    ruta: str
    documentos_cargados: int
    chunks_generados: int
    chunks_almacenados: int
    duracion_segundos: float


def ingerir(ruta: str | Path) -> ReporteIngesta:
    """Ejecuta el flujo completo de ingesta sobre un archivo o una carpeta.

    Args:
        ruta: archivo o carpeta con documentos `.txt`, `.md` o `.pdf`.

    Returns:
        `ReporteIngesta` con las metricas de la ejecucion.
    """
    inicio = time.perf_counter()

    documentos = cargar_documentos(ruta)
    chunks = dividir_documentos(documentos)
    ids_almacenados = almacenar_chunks(chunks)

    duracion = time.perf_counter() - inicio

    return ReporteIngesta(
        ruta=str(ruta),
        documentos_cargados=len(documentos),
        chunks_generados=len(chunks),
        chunks_almacenados=len(ids_almacenados),
        duracion_segundos=round(duracion, 2),
    )
