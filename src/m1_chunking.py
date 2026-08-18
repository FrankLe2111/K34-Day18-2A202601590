from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import glob
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = dict(metadata or {})
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n+", text) if s.strip()]
    if not sentences:
        return []

    def lexical_similarity(left: str, right: str) -> float:
        left_words = set(re.findall(r"\w+", left.lower()))
        right_words = set(re.findall(r"\w+", right.lower()))
        if not left_words or not right_words:
            return 0.0
        return len(left_words & right_words) / (len(left_words | right_words) or 1)

    similarities = [lexical_similarity(sentences[i - 1], sentences[i]) for i in range(1, len(sentences))]
    threshold_to_use = threshold * 0.5
    try:
        from sentence_transformers import SentenceTransformer

        embeddings = SentenceTransformer("all-MiniLM-L6-v2").encode(sentences)
        from math import sqrt

        def cosine(left, right):
            numerator = sum(a * b for a, b in zip(left, right))
            denominator = sqrt(sum(a * a for a in left) * sum(b * b for b in right))
            return numerator / denominator if denominator else 0.0

        similarities = [cosine(embeddings[i - 1], embeddings[i]) for i in range(1, len(sentences))]
        threshold_to_use = threshold
    except Exception:
        pass

    groups = [[sentences[0]]]
    for sentence, similarity in zip(sentences[1:], similarities):
        if similarity < threshold_to_use:
            groups.append([])
        groups[-1].append(sentence)
    return [Chunk(" ".join(group), {**metadata, "strategy": "semantic"}) for group in groups]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")

    metadata = dict(metadata or {})
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    parent_texts = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > parent_size:
            if current:
                parent_texts.append(current)
                current = ""
            parent_texts.extend(paragraph[i:i + parent_size] for i in range(0, len(paragraph), parent_size))
        elif not current:
            current = paragraph
        elif len(current) + 2 + len(paragraph) <= parent_size:
            current += "\n\n" + paragraph
        else:
            parent_texts.append(current)
            current = paragraph
    if current:
        parent_texts.append(current)

    parents = []
    children = []
    for parent_text in parent_texts:
        parent_id = f"parent_{len(parents)}"
        parents.append(Chunk(parent_text, {**metadata, "chunk_type": "parent", "parent_id": parent_id}))
        children.extend(
            Chunk(child_text, {**metadata, "chunk_type": "child", "parent_id": parent_id}, parent_id)
            for child_text in (parent_text[i:i + child_size] for i in range(0, len(parent_text), child_size))
        )
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = dict(metadata or {})
    headers = []
    offset = 0
    fence = None
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            fence = None if fence == marker else marker if fence is None else fence
        elif fence is None:
            header = re.match(r"#{1,3}\s+.+$", line.rstrip("\r\n"))
            if header:
                headers.append((offset, offset + header.end(), header.group()))
        offset += len(line)
    if not headers:
        return [Chunk(text.strip(), {**metadata, "section": "", "strategy": "structure"})] if text.strip() else []

    chunks = []
    if text[:headers[0][0]].strip():
        chunks.append(Chunk(text[:headers[0][0]].strip(), {**metadata, "section": "", "strategy": "structure"}))
    for index, (start, _, header_text) in enumerate(headers):
        end = headers[index + 1][0] if index + 1 < len(headers) else len(text)
        section_text = text[start:end].strip()
        chunks.append(Chunk(section_text, {**metadata, "section": header_text.strip(), "strategy": "structure"}))
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
