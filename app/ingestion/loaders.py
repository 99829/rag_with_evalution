"""
Document loaders for the multi-doc RAG system.
Supports PDF, DOCX, and TXT. Each loader returns a list of
"page records": {"text": str, "page_number": int, "doc_name": str}
"""
from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader
from docx import Document as DocxDocument


def load_pdf(file_path: str) -> List[Dict]:
    """Load a PDF and return one record per page."""
    reader = PdfReader(file_path)
    doc_name = Path(file_path).name
    records = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            records.append({
                "text": text,
                "page_number": page_num,
                "doc_name": doc_name,
            })
    return records


def load_docx(file_path: str) -> List[Dict]:
    """Load a DOCX file. DOCX has no native 'page' concept, so we
    treat every ~500 words as a pseudo-page for citation purposes."""
    doc_name = Path(file_path).name
    docx_obj = DocxDocument(file_path)
    full_text = "\n".join(p.text for p in docx_obj.paragraphs if p.text.strip())

    words = full_text.split()
    words_per_pseudo_page = 500
    records = []
    for i in range(0, len(words), words_per_pseudo_page):
        chunk_words = words[i:i + words_per_pseudo_page]
        pseudo_page = (i // words_per_pseudo_page) + 1
        records.append({
            "text": " ".join(chunk_words),
            "page_number": pseudo_page,
            "doc_name": doc_name,
        })
    return records


def load_txt(file_path: str) -> List[Dict]:
    doc_name = Path(file_path).name
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    return [{
        "text": text,
        "page_number": 1,
        "doc_name": doc_name,
    }]


LOADER_MAP = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_txt,
}


def load_document(file_path: str) -> List[Dict]:
    """Dispatch to the correct loader based on file extension."""
    ext = Path(file_path).suffix.lower()
    if ext not in LOADER_MAP:
        raise ValueError(f"Unsupported file type: {ext}")
    return LOADER_MAP[ext](file_path)


def load_directory(dir_path: str) -> List[Dict]:
    """Load every supported file in a directory."""
    all_records = []
    for file_path in sorted(Path(dir_path).iterdir()):
        if file_path.suffix.lower() in LOADER_MAP:
            all_records.extend(load_document(str(file_path)))
    return all_records