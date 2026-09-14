SYSTEM_PROMPT = """You are a careful research assistant answering questions using ONLY the
provided document excerpts. Follow these rules strictly:

1. Only use information present in the excerpts below. Do not use outside knowledge.
2. Every factual claim in your answer must be followed by a citation in the
   EXACT format [doc_name, p.X] using standard square brackets ( [ and ] ),
   never any other bracket style (no full-width, curly, or angle brackets).
3. If the excerpts do not contain enough information to answer, say so clearly
   instead of guessing.
4. If the question requires combining information from multiple documents,
   synthesize it explicitly and cite each source used.
"""

ANSWER_PROMPT_TEMPLATE = """Question: {question}

Retrieved excerpts:
{context}

Write a clear, well-cited answer following the system rules."""


def format_context(chunks) -> str:
    """Format retrieved chunks into a numbered, citable context block."""
    lines = []
    for i, c in enumerate(chunks, start=1):
        meta = c["metadata"]
        lines.append(
            f"[{i}] ({meta['doc_name']}, p.{meta['page_number']})\n{c['text']}"
        )
    return "\n\n".join(lines)