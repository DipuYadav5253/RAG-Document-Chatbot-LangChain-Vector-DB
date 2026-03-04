# RAG Document Chatbot 🤖

> Ask questions about any PDF document using AI — powered by LangChain, FAISS, Groq (LLaMA 3), and FastAPI

## Architecture
```
PDF → LangChain → FAISS Vector Store → Groq LLaMA 3 → FastAPI → Answer
```

## Tech Stack
| Tool | Purpose |
|---|---|
| LangChain | RAG pipeline orchestration |
| FAISS | Vector store for document chunks |
| HuggingFace | Sentence embeddings |
| Groq (LLaMA 3.3) | Free, fast LLM inference |
| FastAPI | REST API backend |
| MLflow | Experiment tracking |
| Docker | Containerisation |

## Quick Start
Create .env file:
bashGROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
### Option 1 — Docker (recommended)
```bash
git clone https://github.com/DipuYadav5253/rag-chatbot
cd rag-chatbot
cp .env.example .env  # add your GROQ_API_KEY
docker-compose up --build
```

### Option 2 — Local
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

## API Endpoints
| Endpoint | Method | Description |
|---|---|---|
| /health | GET | Check API status |
| /upload-pdf | POST | Upload PDF document |
| /ask | POST | Ask a question |
| /ask-tracked | POST | Ask with MLflow tracking |

## How It Works
1. Upload any PDF via `/upload-pdf`
2. LangChain splits it into 1000-character chunks
3. HuggingFace embeds chunks into vectors
4. FAISS stores vectors locally
5. Questions matched to top 6 relevant chunks
6. Groq LLaMA 3 generates accurate answer from context

## MLflow Tracking
Every experiment tracked: question, model, response time, answer length.
```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5001
```

## Example
```json
POST /ask
{"question": "What are the key skills?"}

Response:
{
  "answer": "The key skills include Python, Django, FastAPI...",
  "sources": ["chunk 1 content...", "chunk 2 content..."]
}
```

## Author
**Dipu Yadav** — M.Sc. Software Engineering, ESIGELEC France  
[LinkedIn](https://www.linkedin.com/in/dipu-yadav-5b1b7b214) | [GitHub](https://github.com/DipuYadav5253)