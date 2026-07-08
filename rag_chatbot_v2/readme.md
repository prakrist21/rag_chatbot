# rag_chatbot_v2

An improved RAG chatbot that answers questions about PDF documents with a persistent vector store and interactive CLI.

## How it works

Two separate scripts handle ingestion and querying, so you only embed once.

### Ingestion (`injest.py`)

1. **Load** : `PyPDFLoader` reads **all** PDF files from the `docs/` folder.
2. **Chunk** : `RecursiveCharacterTextSplitter` splits each document into smaller chunks (500 chars, 50 overlap).
3. **Embed** : Each chunk is converted into a vector using `sentence-transformers/all-MiniLM-L6-v2`.
4. **Persist** : Embeddings are saved to a local `./chroma_db` directory on disk.

### Querying (`query.py`)

1. **Load** : The persisted Chroma database is loaded from disk (no re-embedding needed).
2. **Retrieve** : The top-3 most relevant chunks are fetched via similarity search.
3. **Generate** : Retrieved chunks (with page numbers) are passed as context to `llama-3.3-70b-versatile` via the Groq API.
4. **Loop** : An interactive CLI asks for questions until you type `quit`.

## Tech stack

- [LangChain](https://www.langchain.com/) : orchestration
- [HuggingFace Sentence Transformers](https://www.sbert.net/) : local embeddings
- [Chroma](https://www.trychroma.com/) : vector store (persisted to disk)
- [Groq](https://groq.com/) : fast LLM inference

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/prakrist21/rag_chatbot_v2.git
cd rag_chatbot_v2
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # on Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

> Get a free API key from [console.groq.com](https://console.groq.com/).

### 5. Add your PDFs

Place one or more PDF files inside a `docs/` folder in the project root.

### 6. Ingest the documents

```bash
python injest.py
```

This loads, chunks, embeds, and persists all PDFs from `docs/` into `./chroma_db`.

### 7. Query

```bash
python query.py
```

Type your questions interactively. Enter `quit` to exit.

## Example

```
Ask a question (or 'quit'): Where is the store located?

[Page 3] The store is located at 123 Main Street, Springfield.
```

## Notes

- Embeddings run locally (no external API needed), so the first run will download the `all-MiniLM-L6-v2` model (~80MB).
- The vector store is persisted to `./chroma_db` , you only need to run `injest.py` once (or again when you add/update PDFs).
- Smaller chunk size (500) with less overlap (50) compared to v1, better suited for precise retrieval.
- `.env` and `chroma_db/` are git-ignored, never commit API keys.
