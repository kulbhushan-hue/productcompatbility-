"""
RAG layer: TF-IDF retrieval over notes/docs text + Claude for answer synthesis.

We deliberately use a lightweight TF-IDF retriever (scikit-learn) instead of
a heavy transformer embedding model -- for a corpus of compatibility notes
and short doc snippets (hundreds to low thousands of short rows), TF-IDF
retrieval is fast, dependency-light, and perfectly adequate. Swap in a real
vector DB (Chroma/FAISS + sentence-transformers, or Claude/OpenAI
embeddings) if your corpus grows into the tens of thousands of documents.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import anthropic

CLAUDE_MODEL = "claude-sonnet-4-6"


@dataclass
class DocChunk:
    text: str
    source_url: str
    vendor: str
    product_name: str


@dataclass
class RagIndex:
    chunks: list[DocChunk] = field(default_factory=list)
    vectorizer: TfidfVectorizer | None = None
    matrix: object = None

    @classmethod
    def build(cls, df: pd.DataFrame) -> "RagIndex":
        chunks = []
        for _, row in df.iterrows():
            text_parts = [
                f"{row.get('vendor','')} product {row.get('product_name','')} version {row.get('product_version','')}",
                f"Category: {row.get('category','')}",
                f"Compatibility item: {row.get('item','')} -> status {row.get('status','')}",
            ]
            if isinstance(row.get("notes"), str) and row.get("notes").strip():
                text_parts.append(f"Notes: {row['notes']}")
            chunks.append(
                DocChunk(
                    text=". ".join(text_parts),
                    source_url=row.get("source_url", ""),
                    vendor=row.get("vendor", ""),
                    product_name=row.get("product_name", ""),
                )
            )
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        matrix = vectorizer.fit_transform([c.text for c in chunks])
        return cls(chunks=chunks, vectorizer=vectorizer, matrix=matrix)

    def retrieve(self, query: str, top_k: int = 8) -> list[DocChunk]:
        if not self.chunks:
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in top_idx if sims[i] > 0]


SYSTEM_PROMPT = """You are a mainframe (z/OS) product-compatibility assistant.
You are given:
1. A STRUCTURED, exact list of products confirmed compatible with a given z/OS
   version (this is ground truth -- trust it completely for the YES/NO facts).
2. Some RETRIEVED supporting notes/context (PTF requirements, caveats, source
   links) that may add useful nuance.

Write a clear, well-organized answer for the user:
- Group results by vendor (IBM, Broadcom).
- List each compatible product with its version and any caveat from the notes.
- If the structured list is empty, say so plainly and suggest checking the
  product name spelling or a different z/OS version -- do not invent products.
- Always mention that this reflects the last-scraped data and the user should
  verify against the live vendor page for anything going into production,
  and include the source URLs you were given.
- Never state a product is compatible unless it appears in the structured list.
- Keep it concise: a short intro line, then a clean grouped list.
"""


def generate_answer(
    zos_version: str,
    structured_matches: pd.DataFrame,
    retrieved_chunks: list[DocChunk],
    product_filter: str | None = None,
) -> str:
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    if structured_matches.empty:
        structured_text = "No compatible products found in the structured database for this z/OS version" + (
            f" and product filter '{product_filter}'." if product_filter else "."
        )
    else:
        structured_text = structured_matches.to_string(index=False)

    context_text = "\n".join(
        f"- [{c.vendor} | {c.product_name}] {c.text} (source: {c.source_url})" for c in retrieved_chunks
    )

    user_prompt = f"""z/OS version requested: {zos_version}
Product filter (if any): {product_filter or "none - list all compatible products"}

STRUCTURED MATCHES (ground truth):
{structured_text}

RETRIEVED SUPPORTING CONTEXT:
{context_text if context_text else "(none retrieved)"}

Write the final answer for the user now."""

    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in resp.content if hasattr(block, "text"))
