# Context-Aware Document Q&A API

A containerized microservice for **context-aware document Q&A** using Retrieval-Augmented Generation (RAG). Upload a PDF, chunk and embed it into ChromaDB, then ask questions and get answers grounded in your document—with reduced hallucination by constraining the LLM to retrieved context.

## Features

- PDF upload with automated text extraction and chunking
- Vector storage and similarity search via **ChromaDB**
- RAG pipeline with **LangChain**
- Dual LLM providers: **Ollama** (local) or **Google Gemini** (cloud)
- Embeddings via **Ollama** on the host
- Interactive API docs at `/docs`
- **Docker** support for easy deployment

## Tech Stack

| Layer | Technology |
|-------|------------|
| API | Python, FastAPI, Uvicorn |
| RAG | LangChain |
| Vector DB | ChromaDB |
| PDF parsing | PyPDF |
| LLM | Ollama, Google Gemini |
| Containers | Docker, Docker Compose |

## Prerequisites

- **Python 3.12+** (local development)
- **Ollama** — required for embeddings (and optional for answers)
- **Docker Desktop** — for containerized runs
- **Gemini API key** — optional, if using `LLM_PROVIDER=gemini`

### Ollama models

Pull the models used by this project:

```bash
ollama pull deepseek-r1:1.5b
ollama pull deepseek-r1:latest
```

## Project Structure

```
doc-qa-api/
├── app/
│   ├── main.py      # FastAPI routes
│   ├── rag.py       # PDF processing, embeddings, Q&A
│   └── models.py    # Request/response schemas
├── chroma_db/       # Persisted vectors (gitignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run.py
└── .env             # Secrets (gitignored — create from example below)
```

## Configuration

Create a `.env` file in the project root:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:latest
GEMINI_API_KEY=your-gemini-api-key
CHROMA_DB_PATH=./chroma_db
LLM_PROVIDER=gemini
```

| Variable | Description |
|----------|-------------|
| `OLLAMA_BASE_URL` | Ollama server URL |
| `OLLAMA_MODEL` | Model for Ollama-generated answers |
| `GEMINI_API_KEY` | Google Gemini API key |
| `CHROMA_DB_PATH` | Directory for ChromaDB persistence |
| `LLM_PROVIDER` | `ollama` or `gemini` (default answer provider) |

> **Note:** Embeddings always use Ollama, even when answers use Gemini.

## Run with Docker (recommended)

1. Start **Docker Desktop** and **Ollama** on your machine.
2. Create `.env` as shown above.
3. From the project root:

```bash
docker-compose up --build
```

4. Open **http://localhost:8000/docs**

Docker Compose overrides `OLLAMA_BASE_URL` to `http://host.docker.internal:11434` so the container can reach Ollama on your host.

## Run locally (without Docker)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
python run.py
```

API: **http://localhost:8000**  
Docs: **http://localhost:8000/docs**

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/upload` | Upload a PDF (multipart form, field: `file`) |
| `POST` | `/ask?filename=doc.pdf` | Ask a question about an uploaded document |

### Example: upload

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-document.pdf"
```

### Example: ask

```bash
curl -X POST "http://localhost:8000/ask?filename=your-document.pdf" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What is this document about?\", \"provider\": \"gemini\"}"
```

`provider` is optional; if omitted, the value from `LLM_PROVIDER` in `.env` is used.

## How It Works

1. **Upload** — PDF text is extracted, split into chunks, embedded with Ollama, and stored in ChromaDB.
2. **Ask** — The question is embedded, top relevant chunks are retrieved, and the LLM answers using only that context.
3. **Response** — Returns the answer, provider used, and source chunk previews.

## License

MIT (or specify your license here.)
