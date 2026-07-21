"""Interfaz de linea de comandos del sistema RAG: ingesta y consulta de documentos."""

from __future__ import annotations

import typer
from rich.console import Console

from rag_system.graph.build_graph import ejecutar_consulta
from rag_system.ingestion.pipeline import ingerir

app = typer.Typer(help="CLI del sistema RAG: ingesta de documentos y consulta con recuperacion de contexto.")
console = Console()


@app.command()
def ingest(
    ruta: str = typer.Argument(..., help="Archivo o carpeta con documentos .txt, .md o .pdf a ingerir."),
) -> None:
    """Ingiere uno o varios documentos: carga, divide, genera embeddings y almacena en PGVector."""
    try:
        reporte = ingerir(ruta)
    except (FileNotFoundError, ValueError) as error:
        console.print(f"[bold red]Error de ingesta:[/bold red] {error}")
        raise typer.Exit(code=1)

    console.print("[bold green]Ingesta completada[/bold green]")
    console.print(f"  Ruta procesada:      {reporte.ruta}")
    console.print(f"  Documentos cargados: {reporte.documentos_cargados}")
    console.print(f"  Chunks generados:    {reporte.chunks_generados}")
    console.print(f"  Chunks almacenados:  {reporte.chunks_almacenados}")
    console.print(f"  Duracion:            {reporte.duracion_segundos} s")


@app.command()
def query(
    pregunta: str = typer.Argument(..., help="Pregunta en lenguaje natural sobre los documentos ingeridos."),
) -> None:
    """Ejecuta el flujo de consulta: recupera contexto en PGVector y genera una respuesta con el LLM."""
    try:
        resultado = ejecutar_consulta(pregunta)
    except ValueError as error:
        console.print(f"[bold red]Error de consulta:[/bold red] {error}")
        raise typer.Exit(code=1)

    console.print(f"[bold cyan]Pregunta:[/bold cyan] {pregunta}\n")

    console.print("[bold yellow]Contexto recuperado (distancia coseno; menor = mas similar):[/bold yellow]")
    if not resultado["documents"]:
        console.print("  (sin resultados)")
    for i, doc in enumerate(resultado["documents"], start=1):
        fuente = doc.metadata.get("source_file", "desconocido")
        score = doc.metadata.get("similarity_score", "n/d")
        fragmento = doc.page_content[:200].replace("\n", " ")
        console.print(f"  [{i}] fuente={fuente} distancia={score}")
        console.print(f"      {fragmento}...")

    console.print("\n[bold green]Respuesta:[/bold green]")
    console.print(resultado["answer"])


if __name__ == "__main__":
    app()
