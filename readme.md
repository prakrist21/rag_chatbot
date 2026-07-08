# RAG Chatbot

A Retrieval-Augmented Generation (RAG) system that answers questions about PDF documents using LangChain, local embeddings, and Groq LLM inference.

## Project structure

```
.
├── docs/                  # Place PDF files here
├── rag_chatbot/           # v1 — single-file prototype
│   ├── chatbot.py         #   Loads, chunks, embeds, queries in-memory
│   └── readme.md
├── rag_chatbot_v2/        # v2 — two-phase ingestion & querying
│   ├── injest.py          #   Loads PDFs → chunks → embeds → persists to disk
│   ├── query.py           #   Loads persisted vector store → interactive CLI
│   └── readme.md
├── chroma_db/             # Persisted vector store (auto-generated, git-ignored)
├── requirements.txt       # Shared Python dependencies
├── .env                   # GROQ_API_KEY (git-ignored)
└── .gitignore
```

## Versions

| | v1 (`rag_chatbot`) | v2 (`rag_chatbot_v2`) |
|---|---|---|
| Files | Single `chatbot.py` | `injest.py` + `query.py` |
| Vector store | In-memory (rebuilds every run) | Persisted to `./chroma_db` |
| PDFs | Single file (hardcoded) | All PDFs from `docs/` |
| Interface | Hardcoded sample queries | Interactive CLI loop |
| Chunk size | 1000 chars, 200 overlap | 500 chars, 50 overlap |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate           # on Mac/Linux
pip install -r requirements.txt

# Add your PDFs to docs/
# Create .env with GROQ_API_KEY=your_key

# v1
python rag_chatbot/chatbot.py

# v2
python rag_chatbot_v2/injest.py
python rag_chatbot_v2/query.py
```

## Notes

- Embeddings run locally via `sentence-transformers/all-MiniLM-L6-v2` (~80MB download on first run).
- Requires a free Groq API key from [console.groq.com](https://console.groq.com/).
- `.env` and `chroma_db/` are git-ignored, never commit API keys.
