# love.exe 💌 — RAG Conversational Chatbot

A prototype chatbot that retrieves similar previously indexed stories and uses them as context for LLM-generated answers. This is a software demonstration, not professional counseling.

## How it works

1. **Index stories:** create sentence embeddings and store text in FAISS.
2. **Retrieve:** find relevant indexed stories for a new question.
3. **Generate:** provide retrieved context to Google Gemini through a LangChain conversational chain.
4. **Continue the conversation:** retain previous turns through the project's memory module.
5. **Serve:** provide a browser interface using FastAPI (`web_app.py`).

**Indexing stories is not model fine-tuning.** This application embeds and retrieves documents without retraining the underlying LLM.

## Technologies

Python · FastAPI · LangChain · Google Gemini (`gemini-2.0-flash` configured in `chain.py`) · FAISS · Sentence Transformers

## Getting started

```bash
git clone https://github.com/minjoooo303/love.exe.git
cd love.exe
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Enter your Gemini key as `GOOGLE_API_KEY` in `.env` and run:

```bash
python web_app.py
```

Open http://localhost:8000. Dependencies and external model availability can change; this is a prototype rather than a verified production deployment.

## Files

| File | Purpose |
| --- | --- |
| `web_app.py` | FastAPI routes and HTML UI |
| `main.py` | CLI entry point |
| `chain.py` | LLM and conversational RAG chain |
| `retriever.py` | Similarity retrieval |
| `vector_store.py` | FAISS story indexing |
| `memory.py` | Conversation context |
| `prompts.py` | Prompts |

## Data and security

Never commit `.env`, real conversation logs, or personal stories. The local FAISS index and generated logs are excluded by `.gitignore`. Public repository history should be reviewed separately for any previously committed sensitive information.
