"""Factory del modelo de lenguaje (LLM) usado en el nodo de generacion del grafo."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from rag_system.config import get_settings

_MAX_TOKENS_RESPUESTA = 2048
_TIMEOUT_SEGUNDOS = 60


def obtener_llm() -> BaseChatModel:
    """Instancia el LLM de generacion segun `LLM_PROVIDER` / `LLM_MODEL` (config.py).

    Returns:
        Un `BaseChatModel` de LangChain (`ChatAnthropic` o `ChatGoogleGenerativeAI`
        segun el proveedor configurado), listo para `.invoke(mensajes)`.

    Raises:
        ValueError: si `LLM_PROVIDER` no es un valor soportado (esto tambien lo
            valida `config.Settings`, asi que en la practica no deberia ocurrir).
    """
    settings = get_settings()

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            max_tokens=_MAX_TOKENS_RESPUESTA,
            timeout=_TIMEOUT_SEGUNDOS,
        )

    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
            max_output_tokens=_MAX_TOKENS_RESPUESTA,
            timeout=_TIMEOUT_SEGUNDOS,
        )

    raise ValueError(
        f"LLM_PROVIDER no soportado: {settings.llm_provider!r} "
        "(valores validos: 'anthropic', 'gemini')"
    )
