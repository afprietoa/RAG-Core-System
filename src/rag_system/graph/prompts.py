"""Construccion de los mensajes (system + human) para el nodo de generacion."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

SYSTEM_PROMPT = """\
Eres un asistente que responde preguntas basandote EXCLUSIVAMENTE en el \
contexto proporcionado. Sigue estas reglas de forma estricta:

1. Usa solo la informacion presente en el contexto para responder.
2. Si el contexto no contiene informacion suficiente para responder, dilo \
explicitamente: "No tengo suficiente informacion en los documentos para \
responder esa pregunta."
3. No inventes datos, cifras ni fuentes que no esten en el contexto.
4. Cuando sea posible, menciona el archivo fuente del que proviene la \
informacion (indicado como 'fuente' en cada fragmento del contexto).
5. Responde en espanol, de forma clara y concisa.
"""


def _formatear_contexto(documentos: list[Document]) -> str:
    if not documentos:
        return "(No se recupero contexto relevante)"

    bloques = []
    for i, doc in enumerate(documentos, start=1):
        fuente = doc.metadata.get("source_file", "desconocido")
        bloques.append(f"[Fragmento {i} | fuente: {fuente}]\n{doc.page_content}")

    return "\n\n".join(bloques)


def construir_mensajes_generacion(
    pregunta: str, documentos: list[Document]
) -> list[BaseMessage]:
    """Construye la lista de mensajes (system + human) para el LLM de generacion.

    Args:
        pregunta: pregunta original del usuario.
        documentos: fragmentos de contexto recuperados (pueden ser una lista vacia).

    Returns:
        Lista `[SystemMessage, HumanMessage]` lista para pasar a `llm.invoke(...)`.
    """
    contexto = _formatear_contexto(documentos)

    human_prompt = f"""Contexto recuperado:
{contexto}

Pregunta del usuario:
{pregunta}"""

    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ]
