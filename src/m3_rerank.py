from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import re
import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


def _lexical_score(query: str, text: str) -> float:
    query_tokens = set(re.findall(r"\w+", query.lower()))
    document_tokens = set(re.findall(r"\w+", text.lower()))
    return len(query_tokens & document_tokens) / (len(query_tokens) or 1)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._model_attempted = False

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self._model_attempted:
            self._model_attempted = True
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except Exception:
                self._model = None
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents with CrossEncoder, falling back to token overlap."""
        if not documents or top_k <= 0:
            return []

        model = self._load_model()
        if model is None:
            scores = [_lexical_score(query, document.get("text", "")) for document in documents]
        else:
            pairs = [(query, document["text"]) for document in documents]
            raw_scores = model.predict(pairs)
            if isinstance(raw_scores, (int, float)):
                scores = [float(raw_scores)]
            else:
                try:
                    scores = [float(score) for score in raw_scores]
                except TypeError:
                    scores = [float(raw_scores)]
            if len(scores) != len(documents):
                scores = scores[:len(documents)] + [0.0] * max(0, len(documents) - len(scores))

        ranked = sorted(zip(scores, documents), key=lambda item: item[0], reverse=True)
        return [
            RerankResult(
                text=document["text"],
                original_score=float(document.get("score", 0.0)),
                rerank_score=float(score),
                metadata=document.get("metadata", {}),
                rank=rank,
            )
            for rank, (score, document) in enumerate(ranked[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative when flashrank is installed."""

    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []
        try:
            from flashrank import Ranker, RerankRequest

            self._model = self._model or Ranker()
            passages = [{"text": document["text"]} for document in documents]
            results = self._model.rerank(RerankRequest(query=query, passages=passages))
            ranked = []
            for result in results[:top_k]:
                index = next((i for i, document in enumerate(documents) if document["text"] == result.passage.text), 0)
                ranked.append((float(result.score), documents[index]))
        except Exception:
            ranked = [(_lexical_score(query, document.get("text", "")), document) for document in documents]
            ranked.sort(key=lambda item: item[0], reverse=True)

        return [
            RerankResult(
                text=document["text"],
                original_score=float(document.get("score", 0.0)),
                rerank_score=score,
                metadata=document.get("metadata", {}),
                rank=rank,
            )
            for rank, (score, document) in enumerate(ranked[:top_k])
        ]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark reranker latency over n_runs."""
    if n_runs <= 0:
        return {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    for result in CrossEncoderReranker().rerank(query, docs):
        print(f"[{result.rank}] {result.rerank_score:.4f} | {result.text}")
