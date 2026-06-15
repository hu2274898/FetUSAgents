"""RAG retrieval for the specific VQA pipeline.

Wraps a Chroma vector store backed by OpenAI embeddings. The store is
loaded once per process via a singleton (:class:`RAGRetriever`), and
``_try_get_rag`` returns ``None`` when the optional ``langchain``
dependencies aren't installed so the pipeline can still run without
RAG.

:func:`build_rag_query` composes the retrieval query from the user's
question and the analyst agent's visual-evidence breakdown — the
analyst is multimodal and emits medical jargon ("thalami", "cavum
septi pellucidi", ...) that is far more retrieval-friendly than a
per-task seed table.
"""
from __future__ import annotations

import os
from typing import List, Optional

from .types import VQASample

try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    HAS_RAG = True
except ImportError:
    HAS_RAG = False
    print("[WARN] langchain_openai / langchain_chroma not installed — RAG disabled")


class RAGRetriever:
    """Singleton wrapper around a Chroma vector store of medical knowledge."""

    _instance: Optional["RAGRetriever"] = None

    def __init__(
        self,
        persist_dir: str = "./fetal_ultrasound_knowledge_db",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = "text-embedding-3-small",
    ):
        # Resolve credentials from the environment so we never commit them.
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        api_base = api_base or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for RAG retrieval; set it in the "
                "environment or pass api_key= explicitly."
            )

        if not HAS_RAG:
            raise ImportError(
                "langchain_openai and langchain_chroma are required for RAG"
            )
        if not os.path.exists(persist_dir):
            raise FileNotFoundError(f"RAG database not found: {persist_dir}")

        self.embedding_model = OpenAIEmbeddings(
            model=model,
            openai_api_key=api_key,
            openai_api_base=api_base,
            check_embedding_ctx_length=False,
        )
        self.db = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedding_model,
        )
        print(f"[RAG] Knowledge base loaded from {persist_dir}")

    @classmethod
    def get_instance(cls, **kwargs) -> "RAGRetriever":
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        try:
            results = self.db.similarity_search(query, k=k)
            return [doc.page_content.strip() for doc in results if doc.page_content.strip()]
        except Exception as e:
            print(f"  [RAG ERROR] {e}")
            return []


def _try_get_rag() -> Optional[RAGRetriever]:
    """Return a RAG retriever if available, else ``None``."""
    if not HAS_RAG:
        return None
    try:
        return RAGRetriever.get_instance()
    except Exception as e:
        print(f"  [RAG INIT ERROR] {e}")
        return None


# =========================
# RAG query construction
# =========================
# Cap on the analyst snippet length we splice into the query. Embedding
# models effectively window ~512 tokens; anything past ~600 chars
# dilutes the medical-keyword signal without helping recall.
_ANALYSIS_QUERY_CAP = 600


def build_rag_query(sample: VQASample, analysis_text: str = "") -> str:
    """Compose a retrieval query from the user question + analyst output.

    The analyst is a multimodal LLM agent that has just inspected the
    image and produced a visual-evidence breakdown. Its phrasing tends
    to surface the exact anatomical jargon Chroma's vector store is
    indexed against ("thalami", "cavum septi pellucidi", ...), so it's
    a much better retrieval anchor than a hand-written seed phrase
    keyed on ``sample.task_name``.

    ``analysis_text`` defaults to ``""`` for the rare callers that
    don't have analyst output yet — in that case we fall back to the
    raw question and let the embedding model do its best.
    """
    base = sample.question
    if analysis_text:
        snippet = analysis_text[:_ANALYSIS_QUERY_CAP]
        return f"{base}\n\nAnalysis: {snippet}"
    return base
