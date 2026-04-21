# llama.cpp RAG Pipeline

A local Retrieval-Augmented Generation (RAG) pipeline using llama.cpp for inference and Chroma for vector storage. Query your PDF documents using a local LLM with no cloud dependencies.

## Features

- **Local-only inference** — runs entirely on your machine
- **PDF document support** — extracts and processes PDF files (including encrypted ones)
- **Vector embeddings** — uses HuggingFace's `all-MiniLM-L6-v2` for fast, accurate embeddings
- **Persistent storage** — embeds documents once, reuses across sessions
- **OpenAI-compatible API** — works with any llama-server instance

## Requirements

- Python 3.13+
- [llama-server](https://github.com/ggerganov/llama.cpp) running locally
- ~2GB RAM for embeddings model

## Setup

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/alyavasilyeva/llama-cpp-rag.git
cd llama-cpp-rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start llama-server

In a separate terminal, run your model. Example:

```bash
llama-server -hf bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF:Q4_K_M
```

By default, it runs on `http://localhost:8080`. If using a different port, update `main.py`:

```python
llm = ChatOpenAI(
    base_url="http://localhost:YOUR_PORT/v1",
    ...
)
```

### 3. Run the RAG Pipeline

```bash
python main.py
```

**First run:** Loads PDFs, creates embeddings (~1-2 minutes)  
**Subsequent runs:** Loads cached embeddings (~5 seconds)

## Usage

Once running, you'll see an interactive prompt:

```
Ask a question: What is IEC 61083?
```

The pipeline will:
1. Search `chroma_db/` for relevant document chunks
2. Send them to llama-server for context
3. Return an answer with source documents

Type `quit` to exit.

## Project Structure

```
llama-cpp-rag/
├── main.py              # RAG pipeline script
├── requirements.txt     # Python dependencies
├── resources/           # PDF documents
│   └── *.pdf           # Your documents to query
├── chroma_db/          # Vector store (created on first run)
└── README.md           # This file
```

## How It Works

1. **Load** — PyPDFLoader extracts text from PDFs
2. **Split** — RecursiveCharacterTextSplitter chunks documents (1000 chars, 200 overlap)
3. **Embed** — HuggingFaceEmbeddings converts text to vectors
4. **Store** — Chroma persists embeddings to disk
5. **Retrieve** — On query, find 3 most similar chunks
6. **Generate** — llama-server answers using retrieved context

## Adding More Documents

Drop new PDFs in the `resources/` folder and delete `chroma_db/`:

```bash
rm -rf chroma_db
python main.py
```

The pipeline will rebuild with all documents.

## Troubleshooting

**"Error code: 404"**
- Ensure llama-server is running on the correct port
- Update `base_url` in `main.py` to match your server port

**"cryptography>=3.1 is required"**
- Install: `pip install cryptography`
- Some PDFs are encrypted; this decrypts them

**"No module named 'langchain'"**
- Ensure virtual environment is activated: `source venv/bin/activate`

## Performance Notes

- First embeddings run takes 1-2 minutes (one-time)
- Queries depend on llama-server speed (typically 5-10s)
- Chroma stores embeddings in `chroma_db/` (check disk space)
- Retrieves top-3 chunks per query (adjust in `setup_rag_chain()`)

## License

MIT
