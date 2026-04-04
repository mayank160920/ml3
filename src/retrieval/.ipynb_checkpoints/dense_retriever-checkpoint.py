"""
---------------------------------
RAG page routing via dense cosine similarity search.

Responsibility: return the top-K most relevant page numbers for a
given section configuration. No extraction, no paid LLM calls beyond
the free NVIDIA NIM embedding endpoint.

Query construction strategy
-----------------------------
Multi-signal queries maximise retrieval accuracy by combining:
  1. Section name          — structural signal
  2. Section keywords      — domain anchoring
  3. Entity names          — what we are looking for
  4. Entity descriptions   — semantic signal
  5. Entity example values — format and domain context

This ensures the embedding captures both structural and semantic signals,
bridging the gap between config descriptions and actual document vocabulary
(e.g., "annual amount before coverage begins" → "deductible").

Fallback mechanism
------------------
If initial top-K retrieval yields low-confidence extractions, the retriever
can be called again with an expanded top-K (default: 4) to cast a wider
net over the document.
"""
from __future__ import annotations

import chromadb

from config.config_parser import SectionConfig
from models.nvidia_client import NvidiaEmbeddingClient


class DenseRetriever:
    """
    Retrieves the most relevant page numbers for a section via cosine search.

    Parameters
    ----------
    embedding_client : NvidiaEmbeddingClient (uses query mode for embeddings)
    collection       : ChromaDB collection built by IndexBuilder
    default_top_k    : number of pages to retrieve by default
    fallback_top_k   : expanded page count used during fallback retrieval
    """

    def __init__(
        self,
        embedding_client: NvidiaEmbeddingClient,
        collection: chromadb.Collection,
        default_top_k: int = 2,
        fallback_top_k: int = 4,
    ) -> None:
        self.embedding_client = embedding_client
        self.collection = collection
        self.default_top_k = default_top_k
        self.fallback_top_k = fallback_top_k

    # ── public API ────────────────────────────────────────────────────────────

    def retrieve_for_section(
        self,
        section: SectionConfig,
        top_k: int | None = None,
    ) -> list[int]:
        """
        Retrieve the top-K most relevant page numbers for a section.

        Parameters
        ----------
        section : SectionConfig with name, keywords, and entities
        top_k   : number of pages to return (defaults to self.default_top_k)

        Returns
        -------
        list of page numbers sorted in ascending order
        """
        k = top_k if top_k is not None else self.default_top_k
        query = self._build_query(section)
        return self._search(query, k)

    def retrieve_fallback(self, section: SectionConfig) -> list[int]:
        """
        Expanded retrieval with fallback_top_k pages.

        Called when initial extraction confidence is below threshold for
        one or more entities in the section.

        Parameters
        ----------
        section : SectionConfig

        Returns
        -------
        list of page numbers sorted in ascending order
        """
        return self.retrieve_for_section(section, top_k=self.fallback_top_k)

    def retrieve_by_query(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """
        Search by a raw query string and return (page_number, similarity) pairs.

        Parameters
        ----------
        query : retrieval query string
        top_k : number of results (defaults to self.default_top_k)

        Returns
        -------
        list of (page_number, cosine_similarity) tuples, sorted by similarity
        descending
        """
        k = top_k if top_k is not None else self.default_top_k
        query_vec = self.embedding_client.embed_query(query)

        # Guard: ChromaDB requires n_results ≤ collection size
        n = min(k, self.collection.count())
        if n == 0:
            return []

        result = self.collection.query(
            query_embeddings=[query_vec],
            n_results=n,
            include=["metadatas", "distances"],
        )

        pairs: list[tuple[int, float]] = []
        for meta, dist in zip(
            result["metadatas"][0],
            result["distances"][0],
        ):
            page_num = int(meta["page_number"])
            similarity = 1.0 - dist    # cosine distance → similarity
            pairs.append((page_num, similarity))

        # Sort descending by similarity
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    # ── private ───────────────────────────────────────────────────────────────

    def _build_query(self, section: SectionConfig) -> str:
        """
        Compose a multi-signal retrieval query from section config.

        Combines five signal types for maximum semantic coverage:
          section_name | keywords | entity_names | descriptions | examples
        """
        parts = [
            section.section_name.replace("_", " "),
            " ".join(section.section_keywords),
            " ".join(
                e.entity_name.replace("_", " ")
                for e in section.entities
            ),
            " ".join(
                e.entity_description
                for e in section.entities
            ),
            " ".join(
                e.entity_example_value
                for e in section.entities
                if e.entity_example_value
            ),
        ]
        return " ".join(p for p in parts if p.strip())

    def _search(self, query: str, top_k: int) -> list[int]:
        """
        Embed query and perform cosine similarity search in ChromaDB.

        Returns page numbers sorted ascending.
        """
        pairs = self.retrieve_by_query(query, top_k=top_k)
        return sorted(page_num for page_num, _ in pairs)
