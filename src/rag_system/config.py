"""Configuración centralizada del sistema RAG, cargada y validada desde variables de entorno."""

from __future__ import annotations

from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Todas las variables de configuración del sistema, con valores por defecto razonables.

    Los nombres de los atributos se comparan de forma insensible a mayúsculas contra las
    variables de entorno (p. ej. el atributo ``anthropic_api_key`` se llena desde
    ``ANTHROPIC_API_KEY``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM ---
    llm_provider: str = "anthropic"
    llm_model: str = "claude-opus-4-8"
    anthropic_api_key: str | None = None
    google_api_key: str | None = None

    # --- Embeddings ---
    embedding_provider: str = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Base de datos vectorial ---
    database_url: str
    pgvector_collection: str = "documents"

    # --- Ingesta ---
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Recuperación ---
    retrieval_top_k: int = 4

    @model_validator(mode="after")
    def _validar_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "CHUNK_OVERLAP debe ser menor que CHUNK_SIZE "
                f"(recibido CHUNK_OVERLAP={self.chunk_overlap}, CHUNK_SIZE={self.chunk_size})"
            )
        if self.retrieval_top_k < 1:
            raise ValueError("RETRIEVAL_TOP_K debe ser un entero mayor o igual a 1")
        return self

    @model_validator(mode="after")
    def _validar_proveedor_embeddings(self) -> "Settings":
        proveedores_validos = {"huggingface", "openai"}
        if self.embedding_provider not in proveedores_validos:
            raise ValueError(
                f"EMBEDDING_PROVIDER inválido: {self.embedding_provider!r} "
                f"(valores válidos: {sorted(proveedores_validos)})"
            )
        return self

    @model_validator(mode="after")
    def _validar_proveedor_llm(self) -> "Settings":
        proveedores_validos = {"anthropic", "gemini"}
        if self.llm_provider not in proveedores_validos:
            raise ValueError(
                f"LLM_PROVIDER inválido: {self.llm_provider!r} "
                f"(valores válidos: {sorted(proveedores_validos)})"
            )
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY es obligatoria cuando LLM_PROVIDER=anthropic"
            )
        if self.llm_provider == "gemini" and not self.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY es obligatoria cuando LLM_PROVIDER=gemini"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Devuelve una instancia cacheada de Settings (lee `.env` una sola vez por proceso).

    Usar `get_settings.cache_clear()` en pruebas cuando se necesite recargar variables
    de entorno modificadas dinámicamente.
    """
    return Settings()
