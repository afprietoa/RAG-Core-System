"""Fixtures compartidas para toda la suite de pruebas."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _entorno_de_pruebas(monkeypatch: pytest.MonkeyPatch):
    """Garantiza variables de entorno minimas validas para `Settings()` en cada prueba.

    Las pruebas de integracion (marcador `integration`) sobrescriben `DATABASE_URL`
    para apuntar a la base de datos de pruebas en el puerto 5433 (docker-compose.test.yml).
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-no-valida")
    monkeypatch.setenv(
        "DATABASE_URL",
        os.environ.get(
            "TEST_DATABASE_URL",
            "postgresql+psycopg://rag_test_user:rag_test_password@localhost:5433/rag_test_db",
        ),
    )
    monkeypatch.setenv("PGVECTOR_COLLECTION", "documents_test")

    from rag_system.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
