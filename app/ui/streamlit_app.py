
# Streamlit UI: upload documents, ask questions, and see the retrieved
# chunks alongside the answer (transparency into what the model saw).

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

# MUST be the very first Streamlit command in the script.
st.set_page_config(page_title="Multi-Doc RAG Assistant", layout="wide")

import os

try:
    for key, value in st.secrets.items():
        os.environ[key] = str(value)
except Exception:
    pass  # No secrets.toml locally — expected. .env handles it instead.

import tempfile
from dotenv import load_dotenv
load_dotenv()

from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_index import BM25Index
from app.retrieval.hybrid_retriever import HybridRetriever
from app.generation.graph import build_rag_graph
from app.ingestion.pipeline import ingest_files

@st.cache_resource
def get_pipeline():
    vector_store = VectorStore()
    bm25_index = BM25Index()
    retriever = HybridRetriever(vector_store, bm25_index)
    rag_app = build_rag_graph(retriever)
    return vector_store, bm25_index, retriever, rag_app


vector_store, bm25_index, retriever, rag_app = get_pipeline()

st.title("📄 Multi-Document RAG Assistant")
st.caption("Upload multiple documents, then ask questions that span across them.")

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF / DOCX / TXT files", accept_multiple_files=True,
        type=["pdf", "docx", "txt"],
    )
    if uploaded_files and st.button("Ingest documents"):
        tmp_dir = tempfile.mkdtemp()
        saved_paths = []
        for f in uploaded_files:
            path = Path(tmp_dir) / f.name
            path.write_bytes(f.read())
            saved_paths.append(str(path))
        with st.spinner("Ingesting..."):
            num_chunks = ingest_files(saved_paths, vector_store, bm25_index)
        st.success(f"Ingested {len(uploaded_files)} file(s), {num_chunks} chunks added.")

        st.divider()
    st.subheader("Indexed documents")
    docs = vector_store.list_documents_with_ids()
    if docs:
        for doc in docs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"• {doc['doc_name']}")
            with col2:
                if st.button("🗑️", key=f"delete_{doc['doc_id']}", help=f"Remove {doc['doc_name']}"):
                    vector_store.delete_document(doc['doc_id'])
                    bm25_index.delete(doc['doc_id'])
                    st.rerun()
    else:
        st.write("No documents indexed yet.")
    st.caption(f"Total chunks: {vector_store.count()}")

st.header("Ask a question")
question = st.text_input("Your question", placeholder="e.g. Compare the leave and travel notice periods")

if st.button("Ask") and question:
    if vector_store.count() == 0:
        st.warning("Please upload and ingest at least one document first.")
    else:
        with st.spinner("Retrieving + generating answer..."):
            result = rag_app.invoke({
                "question": question,
                "query_type": None,
                "retrieved_chunks": [],
                "reranked_chunks": [],
                "answer": None,
                "model_used": None,
                "citations_verified": None,
                "flagged_citations": [],
            })

        st.subheader("Answer")
        st.write(result["answer"])

        if result["citations_verified"]:
            st.success("✅ All citations verified against retrieved sources")
        else:
            st.error(f"⚠️ Unverified citations detected: {result['flagged_citations']}")

        st.caption(f"Query classified as: **{result['query_type']}** | Model used: {result['model_used']}")

        with st.expander("🔍 Retrieved source chunks (transparency view)"):
            for i, chunk in enumerate(result["reranked_chunks"], start=1):
                meta = chunk["metadata"]
                st.markdown(f"**[{i}] {meta['doc_name']}, p.{meta['page_number']}** "
                            f"(rerank score: {chunk.get('rerank_score', 0):.3f})")
                st.text(chunk["text"])
                st.divider()