"""Script de diagnóstico manual: verifica la conectividad con PostgreSQL/PGVector.

Uso:
    python scripts/verificar_conexion_db.py
"""

from __future__ import annotations

import sys

import psycopg


def main() -> int:
    try:
        conexion = psycopg.connect(
            host="localhost",
            port=5432,
            user="rag_user",
            password="rag_password",
            dbname="rag_db",
            connect_timeout=5,
        )
    except psycopg.OperationalError as error:
        print(f"[ERROR] No se pudo conectar a PostgreSQL: {error}")
        return 1

    with conexion:
        with conexion.cursor() as cursor:
            cursor.execute("SELECT 1;")
            resultado = cursor.fetchone()
            print(f"[OK] Conexion exitosa. SELECT 1 -> {resultado[0]}")

            cursor.execute(
                "SELECT extname FROM pg_extension WHERE extname = 'vector';"
            )
            extension = cursor.fetchone()
            if extension:
                print("[OK] Extension 'vector' esta instalada.")
            else:
                print("[ADVERTENCIA] Extension 'vector' NO esta instalada.")
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
