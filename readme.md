# rag_chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot that answers questions about a PDF document. 

## How it works

1. **Load** — `PyPDFLoader` reads a PDF file into raw documents.
2. **Chunk** — `RecursiveCharacterTextSplitter` splits the document into overlapping chunks (1000 chars, 200 overlap) so context isn't cut off mid-thought.
3. **Embed** — Each chunk is converted into a vector using `sentence-transformers/all-MiniLM-L6-v2` (runs locally, no API calls needed for embeddings).
4. **Store & Retrieve** — Embeddings are stored in a Chroma vector database. At query time, the most relevant chunks are retrieved via similarity search.
5. **Generate** — The retrieved chunks are passed as context to `llama-3.3-70b-versatile` via the Groq API, which generates the final answer.

## Tech stack

- [LangChain](https://www.langchain.com/) — orchestration
- [HuggingFace Sentence Transformers](https://www.sbert.net/) — local embeddings
- [Chroma](https://www.trychroma.com/) — vector store
- [Groq](https://groq.com/) — fast LLM inference

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/prakrist21/rag_chatbot.git
cd rag_chatbot
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

### 5. Add your PDF

Place the PDF you want to query in the project root and update the filename in the script (default: `test.pdf`).

### 6. Run

```bash
python main.py
```

## Example queries

The script currently asks a few sample questions:

```python
ask_chatbot("Where is the store located")
ask_chatbot("What is the usp of the company")
ask_chatbot("What are some products available in the store")
ask_chatbot("Are there any vacancy available in the company")
```

Swap these out or wrap `ask_chatbot()` in a loop / CLI input for interactive use.

## Notes

- Embeddings run locally (no external API needed), so the first run will download the `all-MiniLM-L6-v2` model (~80MB).
- The vector store is currently in-memory — it rebuilds from the PDF on every run. Persistence (`persist_directory`) can be added for larger documents.
- `.env` is git-ignored — never commit API keys.