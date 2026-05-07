# rag-cli

Chat with any PDF from the terminal. Built on LangChain, FAISS, and Llama 3.3 via Groq.

```
$ python rag.py index apuntes_nlp.pdf
Páginas: 42 | Chunks: 187 | Tiempo: 4.2s

$ python rag.py chat apuntes_nlp.pdf
Chateando con 'apuntes_nlp.pdf'. Escribe 'exit' para salir.

> ¿Qué es la atención multi-cabeza?
La atención multi-cabeza divide el embedding en H subspacios
independientes, permitiendo al modelo atender distintas relaciones
en paralelo. Los resultados se concatenan al final.
[Fuente: p.12, chunk 3 | p.13, chunk 1]

> exit
Hasta luego.
```

## How it works

1. **Index** — the PDF is parsed, split into 500-token chunks with 50-token overlap, and each chunk is embedded with `sentence-transformers/all-MiniLM-L6-v2` (local, no API needed). The vectors are stored locally with FAISS.
2. **Retrieve** — for each question, the top-4 most semantically similar chunks are fetched from the index.
3. **Generate** — Llama 3.3 70B (via Groq) receives the retrieved context, the conversation history (last 6 turns), and the question, then streams the answer.

## Quickstart

```bash
git clone https://github.com/lopezinsua/rag-cli
cd rag-cli
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY (free at console.groq.com)
python rag.py chat your_document.pdf
```

## Features

- Streaming responses (character by character)
- Source citations after each answer (`[Fuente: p.X, chunk Y]`)
- Conversation memory across turns within a session
- Auto-indexes on first `chat` if index is missing
- Works with any PDF in Spanish or English

## Requirements

- Python 3.11+
- Groq API key (free at [console.groq.com](https://console.groq.com))

---

By [López Insua](https://github.com/lopezinsua)
