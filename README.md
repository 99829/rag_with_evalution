# 📄 Multi-Document RAG Assistant

A production-style Retrieval-Augmented Generation (RAG) system that answers questions across multiple documents (PDF, DOCX, TXT) with source citations, hybrid retrieval, cross-encoder reranking, and full LangSmith evaluation/tracing.

**🔗 Live Demo:** [ragwithevalution-ayjvvu3nvk4dbqeaahkdfy.streamlit.app](https://ragwithevalution-ayjvvu3nvk4dbqeaahkdfy.streamlit.app)

---

## What this project demonstrates

Most RAG tutorials stop at "embed and retrieve." This project goes further — it's built the way a production RAG system would actually be designed and evaluated:

- **Hybrid retrieval** (dense vector search + BM25 keyword search, merged via Reciprocal Rank Fusion) instead of embeddings alone
- **Cross-encoder reranking** on top of hybrid retrieval for higher precision
- **LangGraph orchestration** with a query classifier, adaptive retrieval depth, and an automated citation-grounding check that catches hallucinated sources
- **LangSmith evaluation loop** with a 12-question golden dataset and custom evaluators (citation accuracy, groundedness) — not just "it works," but *measured* against expected answers and sources
- **Two interfaces**: a Streamlit UI for interactive use, and a FastAPI backend (`/documents`, `/query`, `/health`) for programmatic access

---

## Architecture

```
[Documents: PDF/DOCX/TXT] → [Chunking + Metadata Tagging] → [Chroma Vector Store + BM25 Index]
                                                                        |
[User Question] → [Query Classifier] → [Hybrid Retrieval (RRF)] → [Cross-Encoder Reranker]
                                                                        |
                                                            [LangGraph: Generate Answer]
                                                                        |
                                                       [Citation Grounding Verification]
                                                                        |
                                                          [Cited Answer + Source Chunks]
                                                                        |
                                                    [LangSmith: Tracing + Eval Metrics]
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Ingestion | `pypdf`, `python-docx`, LangChain text splitters |
| Retrieval | ChromaDB (dense), `rank-bm25` (sparse), Reciprocal Rank Fusion |
| Reranking | `sentence-transformers` cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| Generation | LangGraph, Groq (`openai/gpt-oss-120b` with fallback chain) |
| Evaluation | LangSmith (golden dataset + custom evaluators) |
| Interfaces | Streamlit (UI), FastAPI (REST API) |
| Deployment | Streamlit Community Cloud |

---

## Evaluation Results

Ran a 12-question golden dataset spanning single-document, multi-document, and 3-way cross-document synthesis questions:

- **11/12 (92%)** passed both citation accuracy and citation groundedness checks
- The one failing case was diagnosed to a smaller fallback model drifting from the required citation format on a complex, table-formatted multi-part answer — documented as a known limitation with a proposed prompt-level fix

Full traces and eval runs are logged in LangSmith under the `multidoc-rag` project.

---

## Project Structure

```
rag_project/
├── app/
│   ├── ingestion/       # loaders, chunking, ingestion pipeline
│   ├── retrieval/       # vector store, BM25, hybrid retriever, reranker
│   ├── generation/      # LangGraph orchestration, prompts, LLM client
│   ├── eval/            # golden dataset, evaluators, eval runner
│   ├── api/             # FastAPI app
│   └── ui/              # Streamlit app
├── data/sample_docs/    # sample documents for testing
├── tests/                # test scripts
├── requirements.txt
└── .env.example
```

---

## Setup

**1. Clone and create a virtual environment (Python 3.10 recommended):**
```bash
git clone https://github.com/99829/rag_with_evalution.git
cd rag_with_evalution
python -m venv venv
venv\Scripts\activate      # Windows
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Set up environment variables** — copy `.env.example` to `.env` and fill in:
```
GROQ_API_KEY=your_groq_key          # free at console.groq.com
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=multidoc-rag
```

**4. Run the Streamlit app:**
```bash
streamlit run app/ui/streamlit_app.py
```

**5. Or run the FastAPI backend:**
```bash
uvicorn app.api.main:app --reload
```
Interactive API docs available at `http://127.0.0.1:8000/docs`.

**6. Run the evaluation suite:**
```bash
python -m app.eval.run_eval
```

---

## Key Design Decisions

- **Hybrid over pure-vector retrieval**: dense embeddings miss exact keyword/entity matches (names, IDs); BM25 catches those. RRF merges both without needing to tune a manual weighting.
- **Citation verification as a separate graph node**: rather than trusting the LLM's citations, a dedicated node cross-checks every cited `(doc, page)` against what was actually retrieved — catching hallucinated sources automatically.
- **Groq over OpenAI**: free-tier inference with a multi-model fallback chain, prioritizing accessibility without sacrificing reliability.

---

## Author

**Harsh Sarvaiya** — [GitHub](https://github.com/99829) · [LinkedIn](https://linkedin.com/in/harsh-sarvaiya-2b9549297)
