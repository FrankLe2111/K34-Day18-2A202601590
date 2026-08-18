from __future__ import annotations

"""Module 5: Enrichment Pipeline."""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""

    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|\n+", text) if sentence.strip()]


def _extractive_summary(text: str) -> str:
    sentences = _sentences(text)
    return " ".join(sentences[:2]) if sentences else text


def _fallback_questions(text: str, n_questions: int) -> list[str]:
    return [f"{sentence.rstrip('.!?')}?" for sentence in _sentences(text)[:n_questions]]


def summarize_chunk(text: str) -> str:
    """Create a short summary, with an extractive fallback."""
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return response.choices[0].message.content.strip()
        except Exception as error:
            print(f"  ⚠️  OpenAI summarize failed: {error}")
    return _extractive_summary(text)


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """Generate questions that the chunk can answer."""
    if n_questions <= 0:
        return []
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng."},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            questions = response.choices[0].message.content.strip().splitlines()
            return [question.strip().lstrip("0123456789.-) ") for question in questions if question.strip()][:n_questions]
        except Exception as error:
            print(f"  ⚠️  OpenAI HyQA failed: {error}")
    return _fallback_questions(text, n_questions)


def contextual_prepend(text: str, document_title: str = "") -> str:
    """Prepend a short document context while preserving the original text."""
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu."},
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=80,
            )
            context = response.choices[0].message.content.strip()
            return f"{context}\n\n{text}"
        except Exception as error:
            print(f"  ⚠️  OpenAI contextual failed: {error}")
    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


def extract_metadata(text: str) -> dict:
    """Extract basic topic metadata, using a stable fallback without an API."""
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            content = response.choices[0].message.content.strip()
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
            result = json.loads(content)
            return result if isinstance(result, dict) else {}
        except Exception as error:
            print(f"  ⚠️  OpenAI metadata failed: {error}")
    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


def _fallback_enrichment(text: str, source: str) -> dict:
    return {
        "summary": _extractive_summary(text),
        "questions": _fallback_questions(text, 3),
        "context": f"Trích từ {source}." if source else "Đoạn thông tin chính sách nội bộ.",
        "metadata": {"topic": "general", "entities": [], "category": "policy", "language": "vi"},
    }


def _enrich_single_call(text: str, source: str) -> dict:
    """Get summary, questions, context, and metadata in one LLM call."""
    fallback = _fallback_enrichment(text, source)
    if not OPENAI_API_KEY:
        return fallback
    try:
        from openai import OpenAI

        response = OpenAI().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """Phân tích đoạn văn và trả về JSON:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}"""},
                {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
            ],
            max_tokens=400,
        )
        content = response.choices[0].message.content.strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
        result = json.loads(content)
        return result if isinstance(result, dict) else fallback
    except Exception as error:
        print(f"  ⚠️  Enrichment API failed: {error}")
        return fallback


def enrich_chunks(chunks: list[dict], methods: list[str] | None = None) -> list[EnrichedChunk]:
    """Run combined or individually selected enrichment methods."""
    methods = ["combined"] if methods is None else list(methods)
    enriched = []
    for index, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")
        if "combined" in methods:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context = result.get("context", "")
            enriched_text = f"{context}\n\n{text}" if context else text
            auto_metadata = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_metadata = extract_metadata(text) if "metadata" in methods else {}
            if "summary" in methods and summary and enriched_text == text:
                enriched_text = f"{summary}\n\n{text}"
        if not isinstance(questions, list):
            questions = [str(questions)]
        if not isinstance(auto_metadata, dict):
            auto_metadata = {}
        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_metadata},
            method="+".join(methods),
        ))
        if (index + 1) % 10 == 0 or index + 1 == len(chunks):
            print(f"  Enriched {index + 1}/{len(chunks)} chunks...", flush=True)
    return enriched


if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm."
    print(summarize_chunk(sample))
