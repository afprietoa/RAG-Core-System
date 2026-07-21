# Sistema RAG — LangChain + LangGraph + PGVector

Sistema de Retrieval-Augmented Generation (RAG) desarrollado para la asignatura
"intro IA" (UNAL). Implementa dos flujos:

1. **Ingesta**: carga de documentos → división en fragmentos → generación de
   embeddings → almacenamiento en PostgreSQL con la extensión PGVector.
2. **Consulta**: recuperación de los fragmentos más relevantes para una
   pregunta → generación de la respuesta con un LLM, usando LangGraph para
   orquestar el flujo como un grafo de estados explícito (`retrieve` →
   `generate`).

El proveedor del LLM es intercambiable vía `LLM_PROVIDER` sin tocar el grafo,
el CLI ni los prompts: **Anthropic Claude** (`langchain-anthropic`) o
**Google Gemini** (`langchain-google-genai`).

## Arquitectura

```
Pregunta del usuario
        │
        ▼
 ┌─────────────┐      similarity_search      ┌──────────────┐
 │  retrieve   │ ───────────────────────────▶│   PGVector   │
 │   (nodo)    │◀─────────────────────────── │ (PostgreSQL) │
 └──────┬──────┘         chunks + score      └──────────────┘
        │ documentos
        ▼
 ┌─────────────┐        prompt + contexto      ┌──────────────┐
 │  generate   │ ─────────────────────────────▶│ Claude/Gemini│
 │   (nodo)    │◀────────────────────────────  │  (LLM API)   │
 └──────┬──────┘           respuesta            └──────────────┘
        │
        ▼
     Respuesta
```

Ambos nodos están definidos con LangGraph (`src/rag_system/graph/`) y comparten
un estado tipado (`RAGState`). El diseño e implementación completos están
documentados fase a fase en [`spec/`](spec/), empezando por
[`spec/plan-implementacion-rag.md`](spec/plan-implementacion-rag.md).

## Requisitos previos

- Python 3.11 o superior
- Docker Desktop (con Docker Compose v2)
- Una clave de API de Anthropic (https://console.anthropic.com/) **o** una
  clave de API de Google AI Studio (https://aistudio.google.com/apikey),
  según el proveedor de LLM que elijas en `LLM_PROVIDER`

## Instalación

```powershell
git clone <URL-de-este-repositorio>
cd RAG
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

> `requirements.txt` contiene las versiones exactas verificadas: la instalación
> completa (incluyendo `sentence-transformers`/`torch`) y la suite de pruebas
> unitarias se ejecutaron con éxito sobre esta combinación en Python 3.11
> (Windows). Ver la nota al inicio del archivo para el detalle de dos
> conflictos de dependencias transitivas resueltos durante el desarrollo
> (`langchain-core`, `pgvector`) y una incompatibilidad `typer`/`click`.

## Configuración

```powershell
Copy-Item .env.example .env
```

Edita `.env` y elige tu proveedor de LLM con `LLM_PROVIDER`:

- `LLM_PROVIDER=anthropic` → añade tu clave en `ANTHROPIC_API_KEY` y usa
  `LLM_MODEL=claude-opus-4-8` (o el modelo Claude que tengas habilitado).
- `LLM_PROVIDER=gemini` → añade tu clave en `GOOGLE_API_KEY` y usa
  `LLM_MODEL=gemini-flash-latest` (o el modelo Gemini que tengas habilitado —
  ver el comentario en `.env.example` para listar los modelos disponibles en
  tu clave; algunos alias fechados como `gemini-2.5-flash` ya no aceptan
  claves nuevas).

El resto de variables tiene valores por defecto funcionales (ver
`.env.example` para la descripción de cada una).

## Levantar la base de datos vectorial (PGVector vía Docker)

```powershell
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml ps
```

Debe mostrar el contenedor `rag-postgres` en estado `Up (healthy)`.

## Uso — Fase de ingesta

```powershell
python -m rag_system.cli ingest data/documents/pgvector_intro.txt
```

También acepta una carpeta completa: `python -m rag_system.cli ingest data/documents/`.

## Uso — Fase de consulta

```powershell
python -m rag_system.cli query "Que es PGVector y para que sirve en un sistema RAG?"
```

## Estado de verificación de este repositorio

Este proyecto se implementó y verificó de forma incremental. Lo siguiente **se
ejecutó realmente** en el entorno de desarrollo:

- Instalación completa de dependencias (`pip install -r requirements.txt` y
  `pip install -e .`) sin errores.
- Carga de configuración (`config.py`) y validación fail-fast al faltar una
  variable obligatoria.
- Suite de pruebas unitarias completa: **8/8 pasan**.
- Compilación del grafo de LangGraph (`construir_grafo()`).
- CLI funcional (`--help`, manejo de errores):

```
$ python -m rag_system.cli ingest "ruta/que/no/existe"
Error de ingesta: La ruta no existe: ruta\que\no\existe
$ echo $LASTEXITCODE
1
```

- Generación real de embeddings (carga del modelo `sentence-transformers/all-MiniLM-L6-v2`
  y cálculo de vectores) al ejecutar la prueba de integración: el flujo avanza
  correctamente hasta el punto de escribir en PGVector.
- **Invocación real del LLM de generación con Gemini** (`LLM_PROVIDER=gemini`,
  modelo `gemini-flash-latest`, vía `langchain-google-genai`), confirmando que
  `obtener_llm()` y el nodo `generate_node` funcionan contra la API real de
  Google, no solo con el mock de las pruebas unitarias:

```
$ python -c "
from rag_system.llm.provider import obtener_llm
from langchain_core.messages import HumanMessage
llm = obtener_llm()
print(llm.invoke([HumanMessage(content='Responde solo con la palabra: OK')]).content)
"
OK
```

**Docker no está disponible en el entorno donde se construyó este repositorio**,
por lo que los siguientes pasos —que sí están completamente implementados y
probados unitariamente con mocks— no pudieron ejecutarse de extremo a extremo
contra una base de datos real:

- `docker compose up` levantando PGVector.
- `python -m rag_system.cli ingest ...` completando la escritura en PGVector.
- `python -m rag_system.cli query ...` (recuperación + generación juntas —
  la generación por sí sola ya se verificó arriba).
- Las pruebas de integración (`pytest -m integration`) contra PGVector real
  (se confirmó que fallan de forma clara y controlada, sin colgarse, cuando
  no hay una base de datos disponible en el puerto configurado).

### Cómo completar la verificación (para quien tenga Docker y una API key)

```powershell
docker compose -f docker/docker-compose.yml up -d
python -m rag_system.cli ingest data/documents/pgvector_intro.txt
python -m rag_system.cli query "Que es PGVector y para que sirve en un sistema RAG?"
docker compose -f docker/docker-compose.test.yml up -d
pytest -m integration -v
```

Los formatos de salida esperados para `ingest` y `query`, con ejemplos
ilustrativos completos, están documentados en
[`spec/fase-04-ingesta-documentos.md`](spec/fase-04-ingesta-documentos.md) y
[`spec/fase-08-cli.md`](spec/fase-08-cli.md) respectivamente. **Antes de la
entrega final de la asignatura, reemplaza esta sección por la salida real**
obtenida al correr los comandos anteriores (ver
[`spec/fase-10-documentacion-repositorio.md`](spec/fase-10-documentacion-repositorio.md),
pasos 3 y 4, para el procedimiento exacto).

## Variables de entorno

| Variable | Descripción | Valor por defecto |
|---|---|---|
| `LLM_PROVIDER` | Proveedor del LLM de generación (`anthropic` o `gemini`) | `anthropic` |
| `LLM_MODEL` | Modelo del proveedor seleccionado | `claude-opus-4-8` |
| `ANTHROPIC_API_KEY` | Clave de API de Anthropic (obligatoria solo si `LLM_PROVIDER=anthropic`) | — |
| `GOOGLE_API_KEY` | Clave de API de Google AI Studio (obligatoria solo si `LLM_PROVIDER=gemini`) | — |
| `EMBEDDING_PROVIDER` | Proveedor de embeddings (`huggingface` u `openai`) | `huggingface` |
| `EMBEDDING_MODEL` | Modelo de embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| `DATABASE_URL` | Cadena de conexión a PostgreSQL/PGVector | `postgresql+psycopg://rag_user:rag_password@localhost:5432/rag_db` |
| `PGVECTOR_COLLECTION` | Nombre de la colección de vectores | `documents` |
| `CHUNK_SIZE` | Tamaño de fragmento en caracteres | `1000` |
| `CHUNK_OVERLAP` | Solapamiento entre fragmentos | `150` |
| `RETRIEVAL_TOP_K` | Número de fragmentos recuperados por consulta | `4` |

Ver `.env.example` para la plantilla completa.

## Pruebas

```powershell
pytest -m unit -v                                            # rapidas, sin Docker (8/8 verificadas)
docker compose -f docker/docker-compose.test.yml up -d
pytest -m integration -v                                     # requieren Docker
```

## Estructura del proyecto

```
RAG/
├── src/rag_system/       # codigo fuente del sistema
│   ├── config.py         # configuracion centralizada (pydantic-settings)
│   ├── ingestion/        # carga, division, embeddings y almacenamiento
│   ├── retrieval/        # recuperacion de contexto desde PGVector
│   ├── graph/             # estado y nodos de LangGraph
│   ├── llm/               # proveedor del LLM (Claude o Gemini, via LLM_PROVIDER)
│   └── cli.py             # interfaz de linea de comandos
├── data/documents/        # documentos de ejemplo
├── docker/                # docker-compose para PGVector (dev y test)
├── tests/                 # pruebas unitarias e integracion
└── spec/                  # documentacion detallada por fase de implementacion
```

## Limitaciones conocidas

- El corpus de ejemplo incluido (`data/documents/pgvector_intro.txt`) es
  deliberadamente pequeño; con corpus más grandes y temáticamente variados la
  discriminación del retriever mejora.
- Los documentos PDF escaneados (sin capa de texto) no se procesan — el
  pipeline no incluye OCR.
- No hay soporte multi-turno (memoria de conversación) — cada consulta al CLI
  es independiente.
- La eliminación de contenido obsoleto tras editar un documento fuente ya
  ingerido requiere limpieza manual (ver `spec/fase-04-ingesta-documentos.md`,
  tabla de errores).
- Ver la sección "Estado de verificación" arriba para lo que aún requiere
  Docker y una clave de API real para probarse de extremo a extremo.

## Documentación de diseño e implementación

El proceso completo de diseño e implementación, fase por fase, con código
completo, comandos y solución de errores, está en [`spec/`](spec/):

- [`plan-implementacion-rag.md`](spec/plan-implementacion-rag.md) — plan general
- [`fase-01-preparacion.md`](spec/fase-01-preparacion.md)
- [`fase-02-dependencias-configuracion.md`](spec/fase-02-dependencias-configuracion.md)
- [`fase-03-docker-pgvector.md`](spec/fase-03-docker-pgvector.md)
- [`fase-04-ingesta-documentos.md`](spec/fase-04-ingesta-documentos.md)
- [`fase-05-recuperacion-contexto.md`](spec/fase-05-recuperacion-contexto.md)
- [`fase-06-estado-grafo-langgraph.md`](spec/fase-06-estado-grafo-langgraph.md)
- [`fase-07-generacion-llm.md`](spec/fase-07-generacion-llm.md)
- [`fase-08-cli.md`](spec/fase-08-cli.md)
- [`fase-09-pruebas.md`](spec/fase-09-pruebas.md)
- [`fase-10-documentacion-repositorio.md`](spec/fase-10-documentacion-repositorio.md)
