"""
src/retrieval/index_builder.py
--------------------------------
Builds an in-memory ChromaDB dense vector index from OCR-structured pages.

Each document produces one temporary ChromaDB collection that is
destroyed after the document is fully processed — no persistent storage,
no cross-document contamination.

Embedding model: nvidia/llama-3.2-nemoretriever-300m-embed-v1
  - Retrieval-optimised training objective
  - 300M parameters — fast inference, low memory
  - Free tier via NVIDIA NIM

Index format per document page:
  - id       : str(page_number)
  - embedding: dense float vector from NemoRetriever (passage mode)
  - document : composite_index text from OCR StructuredPage
  - metadata : {"page_number": int}
"""
from __future__ import annotations

import time

import chromadb

from models.nvidia_client import NvidiaEmbeddingClient
from shared_types import StructuredPage


class IndexBuilder:
    """
    Embeds page composite index texts and stores them in ChromaDB.

    Parameters
    ----------
    embedding_client : NvidiaEmbeddingClient for NemoRetriever embeddings
    collection_name  : ChromaDB collection name (default: 'doc_pages')
    """

    DEFAULT_COLLECTION = "doc_pages"

    def __init__(
        self,
        embedding_client: NvidiaEmbeddingClient,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        self.embedding_client = embedding_client
        self.collection_name = collection_name
        self._chroma_client = chromadb.Client()   # ephemeral in-memory client
        self._collection: chromadb.Collection | None = None

    # ── public API ────────────────────────────────────────────────────────────

    def build(
        self,
        structured_pages: dict[int, StructuredPage],
    ) -> chromadb.Collection:
        """
        Embed all page index texts and store vectors in ChromaDB.

        Steps
        -----
        1. Drop any existing collection with the same name
        2. Create a new cosine-similarity collection
        3. Embed all page composite_index texts via NemoRetriever (passage mode)
        4. Add vectors + metadata to the collection

        Parameters
        ----------
        structured_pages : mapping of page_number → StructuredPage

        Returns
        -------
        chromadb.Collection ready for cosine similarity queries
        """
        # Tear down any stale collection before rebuilding
        self._drop_collection()

        self._collection = self._chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        page_numbers = sorted(structured_pages.keys())
        texts = [structured_pages[p].index_text for p in page_numbers]

        # Embed as passages (documents being indexed, not queries)
        vectors = self.embedding_client.embed_passages(texts)

        self._collection.add(
            ids=[str(p) for p in page_numbers],
            embeddings=vectors,
            documents=texts,
            metadatas=[{"page_number": p} for p in page_numbers],
        )
        return self._collection

    def build_from_pdf_fallback(
        self,
        pdf_path: str,
        collection_name: str | None = None,
    ) -> chromadb.Collection:
        """
        Build an index directly from a PDF using PyMuPDF text extraction
        as a fallback when PaddleOCR is unavailable.

        Parameters
        ----------
        pdf_path        : path to the PDF
        collection_name : override the default collection name

        Returns
        -------
        chromadb.Collection
        """
        import fitz
        from ocr.ocr_engine import OCREngine

        if collection_name:
            self.collection_name = collection_name

        ocr = OCREngine()
        structured_pages: dict[int, StructuredPage] = {}

        with fitz.open(pdf_path) as doc:
            for page_idx, page in enumerate(doc, start=1):
                raw_text = page.get_text("text")
                sp = ocr.process_text(raw_text, page_number=page_idx)
                structured_pages[page_idx] = sp

        return self.build(structured_pages)

    def get_collection(self) -> chromadb.Collection:
        """
        Return the current active ChromaDB collection.

        Raises
        ------
        RuntimeError : if build() has not been called yet
        """
        if self._collection is None:
            raise RuntimeError(
                "No index built. Call build() before get_collection()."
            )
        return self._collection

    def destroy(self) -> None:
        """
        Delete the ChromaDB collection to free memory.

        Called automatically by the pipeline after extraction is complete
        for a document to prevent cross-document contamination.
        """
        self._drop_collection()
        self._collection = None

    # ── private ───────────────────────────────────────────────────────────────

    def _drop_collection(self) -> None:
        """Silently drop the collection if it exists."""
        try:
            self._chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
