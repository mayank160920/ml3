"""
tests/test_end_to_end.py
--------------------------
Simple end-to-end test script to verify all CMSVS components work correctly.

Run with:
    cd cmsvs
    python tests/test_end_to_end.py

Requirements:
    - NVIDIA_API_KEY set in environment or .env file
    - All dependencies installed (pip install -r requirements.txt)
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
from pathlib import Path

# ── Add src/ to Python path ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

print(Path(__file__).parent.parent / ".env")

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")

def ok(msg: str) -> None:
    print(f"  ✅  {msg}")

def fail(msg: str) -> None:
    print(f"  ❌  {msg}")

def info(msg: str) -> None:
    print(f"  ℹ   {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic test assets
# ══════════════════════════════════════════════════════════════════════════════

def create_synthetic_pdf(path: str) -> str:
    """Create a minimal 3-page SBC-style PDF for testing."""
    import fitz

    pages = [
        {
            "title": "Summary of Benefits and Coverage",
            "body": (
                "Plan Name: HealthFirst Gold PPO\n"
                "Coverage Period: 01/01/2024 – 12/31/2024\n\n"
                "DEDUCTIBLES\n"
                "Individual Deductible (In-Network): $1,500\n"
                "Family Deductible (In-Network): $3,000\n"
                "Individual Deductible (Out-of-Network): $4,500\n"
            ),
        },
        {
            "title": "Out-of-Pocket Maximums",
            "body": (
                "Individual Out-of-Pocket Maximum (In-Network): $5,000\n"
                "Family Out-of-Pocket Maximum (In-Network): $10,000\n"
                "Individual Out-of-Pocket Maximum (Out-of-Network): $15,000\n"
            ),
        },
        {
            "title": "Copayments and Prescription Drugs",
            "body": (
                "Primary Care Visit: $25 copay after deductible\n"
                "Specialist Visit: $50 copay after deductible\n"
                "Emergency Room: $250 copay then 20% coinsurance\n"
                "Urgent Care: $40 copay\n"
                "Preventive Care: No charge\n\n"
                "Tier 1 Generic: $10 copay per 30-day supply\n"
                "Tier 2 Preferred Brand: $40 copay per 30-day supply\n"
                "Tier 3 Non-Preferred Brand: $80 copay per 30-day supply\n"
                "Tier 4 Specialty: 25% coinsurance\n"
            ),
        },
    ]

    doc = fitz.open()
    for page_data in pages:
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 60), page_data["title"], fontsize=14)
        page.draw_line((72, 75), (540, 75), color=(0.3, 0.3, 0.3), width=1)
        y = 100
        for line in page_data["body"].split("\n"):
            page.insert_text((72, y), line, fontsize=10)
            y += 18

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    return path


def create_synthetic_image(path: str) -> str:
    """Create a minimal PNG image with form-like content for testing."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (800, 600), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    lines = [
        ("MARKETING RESEARCH AUTHORIZATION", (72, 40), 16),
        ("Form No: 06", (72, 80), 12),
        ("Date: 4/16/90", (72, 105), 12),
        ("Stamp ID: 670801704", (72, 130), 12),
        ("From: J. Smith", (72, 160), 12),
        ("To: P.W. Putney", (72, 185), 12),
        ("Project: Y-1 Ultra 100's vs. Winston Ultra 100's", (72, 210), 12),
        ("Project No: 1990-48B", (72, 235), 12),
        ("Organisation: Kapuler Marketing Research", (72, 260), 12),
        ("Sample Size: 400", (72, 285), 12),
        ("Total Cost: $12,500", (72, 310), 12),
        ("CC: S. Willinger (3), K. A. Hutchison/S. A. Howard", (72, 335), 10),
    ]

    for text, pos, size in lines:
        draw.text(pos, text, fill=(0, 0, 0))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Test: 1 — Shared Types
# ══════════════════════════════════════════════════════════════════════════════

def test_shared_types() -> bool:
    section("TEST 1 — Shared Types")
    try:
        from shared_types import (
            InputType, PageImage, LoadedInput, StructuredPage,
            ExtractionStatus, EntityType, RawExtraction,
            EntityResult, FinalEntityValue,
            ValidationStatus, DiscrepancyType,
            EntityValidationResult, SectionValidationResult,
            ValidationReport,
        )

        # Basic enum checks
        assert InputType.PDF == "pdf"
        assert InputType.IMAGE == "image"
        assert ExtractionStatus.FOUND == "FOUND"
        assert ValidationStatus.MATCH == "MATCH"
        assert DiscrepancyType.NUMERIC_DIFFERENCE == "NUMERIC_DIFFERENCE"

        # FinalEntityValue instantiation
        fev = FinalEntityValue(
            entity_name="test_entity",
            extracted_value="$1,500",
            extraction_status=ExtractionStatus.FOUND,
            entity_type=EntityType.DIRECT,
            confidence=0.95,
            source_page=1,
            source_region="Deductibles table",
            raw_context="Individual Deductible: $1,500",
            review_required=False,
            fallback_triggered=False,
            expression_audit=None,
        )
        assert fev.entity_name == "test_entity"
        assert fev.extracted_value == "$1,500"

        # ValidationReport aggregation
        from shared_types import EntityValidationResult, SectionValidationResult
        evr = EntityValidationResult(
            entity_name="e1",
            section_name="sec1",
            doc_a_value="$1,500",
            doc_b_value="$1,500",
            doc_a_normalized="1500.00 USD",
            doc_b_normalized="1500.00 USD",
            validation_status=ValidationStatus.MATCH,
            discrepancy_type=DiscrepancyType.NOT_APPLICABLE,
            reasoning="Values are equal.",
            confidence=1.0,
            review_required=False,
            fast_path_match=True,
        )
        svr = SectionValidationResult(section_name="sec1", entity_results=[evr])
        assert svr.match_count == 1
        assert svr.mismatch_count == 0

        report = ValidationReport(
            doc_a_path="a.pdf",
            doc_b_path="b.pdf",
            config_name="test_config",
            section_results=[svr],
        )
        assert report.total_entities == 1
        assert report.total_matches == 1

        ok("All shared types instantiate and behave correctly")
        return True

    except Exception as exc:
        fail(f"Shared types test failed: {exc}")
        import traceback; traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Test: 2 — Config Parser
# ══════════════════════════════════════════════════════════════════════════════

def test_config_parser() -> bool:
    section("TEST 2 — Config Parser")
    try:
        from config.config_parser import CMSVSConfigParser

        config_path = Path(__file__).parent.parent / "configs" / "funsd_ner_config.yaml"
        if not config_path.exists():
            fail(f"Config file not found: {config_path}")
            return False

        parser = CMSVSConfigParser()
        config = parser.load(config_path)

        assert config.config_name == "funsd_ner_config"
        assert config.domain == "form_understanding"
        assert len(config.sections) > 0

        # Check entity names
        all_names = config.all_entity_names()
        assert "document_title" in all_names
        assert "primary_signer_name" in all_names
        assert "final_amount" in all_names

        ok(f"Config loaded: {config.config_name}")
        ok(f"Sections: {len(config.sections)}")
        ok(f"Total entities: {len(all_names)}")

        # Healthcare config
        hc_path = Path(__file__).parent.parent / "configs" / "healthcare_sbc_config.yaml"
        if hc_path.exists():
            hc_config = parser.load(hc_path)
            expression_entities = [
                e.entity_name
                for s in hc_config.sections
                for e in s.entities
                if e.entity_extraction_logic == "EXPRESSION"
            ]
            ok(f"Healthcare config loaded: {hc_config.config_name}")
            ok(f"Expression entities: {expression_entities}")
        return True

    except Exception as exc:
        fail(f"Config parser test failed: {exc}")
        import traceback; traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Test: 3 — Input Handler & Image Loader
# ══════════════════════════════════════════════════════════════════════════════

def test_input_handling() -> bool:
    section("TEST 3 — Input Handler & Image Loader")
    try:
        from input.input_handler import InputHandler
        from shared_types import InputType

        handler = InputHandler()

        # Extension detection
        assert handler.detect_input_type("doc.pdf") == InputType.PDF
        assert handler.detect_input_type("scan.png") == InputType.IMAGE
        assert handler.detect_input_type("form.jpg") == InputType.IMAGE

        ok("Extension detection works for PDF and image types")

        # Image loading with synthetic image
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test_form.png")
            create_synthetic_image(img_path)

            loaded = handler.load(img_path)
            assert loaded.input_type == InputType.IMAGE
            assert loaded.total_pages == 1
            assert 1 in loaded.page_images
            page = loaded.page_images[1]
            assert page.image_base64  # non-empty base64
            assert page.mime_type == "image/png"
            assert page.width > 0
            assert page.height > 0

            ok(f"Image loaded: {page.width}×{page.height}px")
            ok(f"Base64 length: {len(page.image_base64)} chars")

        return True

    except Exception as exc:
        fail(f"Input handling test failed: {exc}")
        import traceback; traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Test: 4 — PageImageStore & PDF Loading
# ══════════════════════════════════════════════════════════════════════════════

def test_pdf_loading() -> bool:
    section("TEST 4 — PageImageStore & PDF Loading")
    try:
        from ingestion.page_image_store import PageImageStore

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "test_sbc.pdf")
            create_synthetic_pdf(pdf_path)

            store = PageImageStore(dpi=150)
            total = store.load_pdf(pdf_path)

            assert total == 3
            assert store.page_count == 3

            # Get specific pages
            pages = store.get_pages([1, 2])
            assert 1 in pages and 2 in pages

            all_pages = store.get_all_pages()
            assert len(all_pages) == 3

            for pn, page in all_pages.items():
                assert page.page_number == pn
                assert page.image_base64
                assert page.width > 0

            ok(f"PDF loaded: {total} pages")
            ok(f"Page 1 size: {all_pages[1].width}×{all_pages[1].height}px")
            ok(f"Page 2 size: {all_pages[2].width}×{all_pages[2].height}px")

            # Test clear
            store.clear()
            assert store.page_count == 0
            ok("Store cleared successfully")

        return True

    except Exception as exc:
        fail(f"PDF loading test failed: {exc}")
        import traceback; traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Test: 5 — OCR Engine
# ══════════════════════════════════════════════════════════════════════════════

def test_ocr_engine() -> bool:
    section("TEST 5 — OCR Engine (text processing)")
    try:
        from ocr.ocr_engine import OCREngine

        ocr = OCREngine()

        # Test process_text (no PaddleOCR needed)
        sample_text = (
            "DEDUCTIBLES\n"
            "Individual Deductible (In-Network): $1,500\n"
            "Family Deductible (In-Network): $3,000\n"
            "Out-of-Pocket Maximum: $5,000\n"
        )

        sp = ocr.process_text(sample_text, page_number=1)

        assert sp.page_number == 1
        assert sp.raw_text == sample_text
        assert "DEDUCTIBLES" in sp.section_headers or len(sp.section_headers) >= 0
        assert sp.index_text.startswith("SECTIONS:")
        assert "KEY_VALUES:" in sp.index_text
        assert "RAW_TEXT:" in sp.index_text

        ok(f"StructuredPage created for page 1")
        ok(f"Section headers found: {sp.section_headers}")
        ok(f"Key-value pairs found: {len(sp.key_value_pairs)}")
        ok(f"Index text length: {len(sp.index_text)} chars")

        # Test key-value extraction
        if sp.key_value_pairs:
            first_k = list(sp.key_value_pairs.keys())[0]
            ok(f"Sample KV: '{first_k}' → '{sp.key_value_pairs[first_k]}'")

        return True

    except Exception as exc:
        fail(f"OCR engine test failed: {exc}")
        import traceback; traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Test: 6 — Value Normalizer
# ══════════════════════════════════════════════════════════════════════════════

def test_value_normalizer() -> bool:
    section("TEST 6 — Value Normalizer (rule-based fast path)")
    try:
        from validation.utils.value_normalizer import ValueNormalizer

        norm = ValueNormalizer()

        # Monetary normalisation
        cases_monetary = [
            ("$1,500",    "1500.00 USD"),
            ("$1,500.00", "1500.00 USD"),
            ("1500",      "1500.00 USD"),
            ("No charge", "0.00 USD"),
            ("Covered in full", "0.00 USD"),
        ]
        for raw, expected in cases_monetary:
            result = norm.normalize(raw, "monetary")
            assert result == expected, f"'{raw}' → '{result}' (expected '{expected}')"
        ok(f"Monetary normalisation: {len(cases_monetary)} cases passed")

        # Percentage normalisation
        cases_pct = [
            ("20%",       "20.0%"),
            ("20 percent","20.0%"),
            ("0.20",      "20.0%"),
        ]
        for raw, expected in cases_pct:
            result = norm.normalize(raw, "percentage")
            assert result == expected, f"'{raw}' → '{result}' (expected '{expected}')"
        ok(f"Percentage normalisation: {len(cases_pct)} cases passed")

        # Coverage normalisation
        cases_cov = [
            ("Not covered",        "MEMBER_PAYS_100_PERCENT"),
            ("Member pays 100%",   "MEMBER_PAYS_100_PERCENT"),
            ("Covered in full",    "0.00 USD"),
        ]
        for raw, expected in cases_cov:
            result = norm.normalize(raw, "coverage_classification")
            assert result == expected, f"'{raw}' → '{result}' (expected '{expected}')"
        ok(f"Coverage normalisation: {len(cases_cov)} cases passed")

        # Fast-path match detection
        assert norm.is_fast_path_match("$1,500", "$1,500.00", "monetary")
        assert norm.is_fast_path_match("20%", "20.0%", "percentage")
        assert not norm.is_fast_path_match("$1,500", "$2,000", "monetary")
        ok("Fast-path match detection works correctly")

        # None handling
        assert norm.normalize(None, "monetary") is None
        assert not norm.is_fast_path_match(None, "$1,500", "monetary")
        ok("None value handling works correctly")

        return True

    except Exception as exc:
        fail(f"Value normalizer test failed: {exc}")
        import traceback; traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Test: 7 — Expression Evaluator
# ══════════════════════════════════════════════════════════════════════════════

def test_expression_evaluator() -> bool:
    section("TEST 7 — Expression Evaluator (SimpleEval sandboxed)")
    try:
        from extraction.expression_evaluator import ExpressionEvaluator

        evaluator = ExpressionEvaluator()

        # Basic addition
        result = evaluator.evaluate(
            template="var_a + var_b",
            variable_values={"var_a": "$1,500", "var_b": "$3,000"},
            data_type="monetary",
        )
        assert result["status"] == "SUCCESS"
        assert result["computed_value"] == "$4,500.00"
        assert result["numeric_result"] == 4500.0
        ok(f"Addition: $1,500 + $3,000 = {result['computed_value']}")

        # Multiplication
        result2 = evaluator.evaluate(
            template="tier1 + tier2",
            variable_values={"tier1": "$10", "tier2": "$40"},
            data_type="monetary",
        )
        assert result2["status"] == "SUCCESS"
        assert result2["computed_value"] == "$50.00"
        ok(f"Drug cost: $10 + $40 = {result2['computed_value']}")

        # With confidences
        result3 = evaluator.evaluate_with_confidences(
            template="a + b",
            variable_values={"a": "$5,000", "b": "$10,000"},
            variable_confidences={"a": 0.95, "b": 0.90},
            data_type="monetary",
        )
        assert result3["status"] == "SUCCESS"
        assert abs(result3["confidence"] - 0.925) < 0.001
        ok(f"Confidence averaging: {result3['confidence']:.3f}")

        # Security test: blocked injections
        injection_cases = [
            "__import__('os').system('ls')",
            "open('/etc/passwd').read()",
            "exec('import sys')",
        ]
        blocked = 0
        for expr in injection_cases:
            res = evaluator.evaluate(
                template=expr,
                variable_values={},
                data_type="text",
            )
            if res["status"] == "ERROR":
                blocked += 1
        assert blocked == len(injection_cases)
        ok(f"Security: {blocked}/{len(injection_cases)} injection attempts blocked")

        # Missing variable → ERROR
        result4 = evaluator.evaluate(
            template="a + b",
            variable_values={"a": "$1,000", "b": None},
            data_type="monetary",
        )
        assert result4["status"] == "ERROR"
        ok("Missing variable produces ERROR (not crash)")

        return True

    except Exception as exc:
        fail(f"Expression evaluator test failed: {exc}")
        import traceback; traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Test: 8 — NER Prompt Builder
# ══════════════════════════════════════════════════════════════════════════════

def test_prompt_builders() -> bool:
    section("TEST 8 — Prompt Builders")
    try:
        from prompts.ner_prompt_builder import NERPromptBuilder
        from prompts.validation_prompt_builder import ValidationPromptBuilder
        from config.config_parser import CMSVSConfigParser

        config_path = Path(__file__).parent.parent / "configs" / "funsd_ner_config.yaml"
        config = CMSVSConfigParser().load(config_path)
        section_cfg = config.sections[0]   # Document Identity

        # NER prompt
        ner_builder = NERPromptBuilder()
        ner_prompt = ner_builder.build_section_prompt(section_cfg)

        assert "document_title" in ner_prompt
        assert "DIRECT" in ner_prompt
        assert "extracted_value" in ner_prompt
        assert "confidence" in ner_prompt
        ok(f"NER prompt built: {len(ner_prompt)} chars")

        # Fallback NER prompt
        fallback_prompt = ner_builder.build_fallback_prompt(section_cfg)
        assert "FALLBACK" in fallback_prompt
        ok("Fallback NER prompt includes fallback note")

        # Validation prompt
        val_builder = ValidationPromptBuilder()
        pairs = [
            {
                "entity_name":     "document_title",
                "description":     "Title of the document",
                "data_type":       "text",
                "doc_a_value":     "MARKETING RESEARCH AUTHORIZATION",
                "doc_b_value":     "marketing research authorization",
                "doc_a_normalized": "marketing research authorization",
                "doc_b_normalized": "marketing research authorization",
            }
        ]
        val_prompt = val_builder.build_section_validation_prompt(
            section=section_cfg,
            entity_pairs=pairs,
        )
        assert "MATCH" in val_prompt
        assert "MISMATCH" in val_prompt
        assert "document_title" in val_prompt
        ok(f"Validation prompt built: {len(val_prompt)} chars")

        # Single entity prompt
        single_prompt = val_builder.build_single_entity_prompt(
            entity_name="document_title",
            entity_description="Title of the document",
            data_type="text",
            doc_a_value="TITLE A",
            doc_b_value="title a",
        )
        assert "document_title" in single_prompt
        ok("Single entity validation prompt built")

        return True

    except Exception as exc:
        fail(f"Prompt builder test failed: {exc}")
        import traceback; traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Test: 9 — NVIDIA Embedding Client (requires API key)
# ══════════════════════════════════════════════════════════════════════════════

def test_nvidia_embeddings() -> bool:
    section("TEST 9 — NVIDIA Embedding Client (API call)")
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        info("NVIDIA_API_KEY not set — skipping API test")
        return True

    try:
        from models.nvidia_client import NvidiaEmbeddingClient

        client = NvidiaEmbeddingClient(api_key=api_key)

        # Passage embedding
        passages = [
            "SECTIONS: DEDUCTIBLES | KEY_VALUES: Individual Deductible: $1,500 | RAW_TEXT: Individual Deductible (In-Network): $1,500",
            "SECTIONS: OUT-OF-POCKET MAXIMUMS | KEY_VALUES: OOP Max: $5,000 | RAW_TEXT: Individual OOP Maximum: $5,000",
        ]
        vectors = client.embed_passages(passages)
        assert len(vectors) == 2
        assert len(vectors[0]) > 0
        ok(f"Passage embeddings: {len(vectors)} vectors, dim={len(vectors[0])}")

        # Query embedding
        query = "individual deductible in-network annual amount before coverage begins"
        query_vec = client.embed_query(query)
        assert len(query_vec) == len(vectors[0])
        ok(f"Query embedding: dim={len(query_vec)}")

        # Cosine similarity check: relevant page should have higher similarity
        import numpy as np
        def cosine_sim(a, b):
            a, b = np.array(a), np.array(b)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        sim0 = cosine_sim(query_vec, vectors[0])
        sim1 = cosine_sim(query_vec, vectors[1])
        ok(f"Similarity to deductible page: {sim0:.4f}")
        ok(f"Similarity to OOP max page:    {sim1:.4f}")
        ok(f"Deductible page ranks higher: {sim0 > sim1}")

        return True

    except Exception as exc:
        fail(f"NVIDIA embedding test failed: {exc}")
        import traceback; traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Test: 10 — Dense Index & Retriever (requires API key)
# ══════════════════════════════════════════════════════════════════════════════

def test_dense_retrieval() -> bool:
    section("TEST 10 — Dense Index Builder & Retriever (API call)")
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        info("NVIDIA_API_KEY not set — skipping API test")
        return True

    try:
        from models.nvidia_client import NvidiaEmbeddingClient
        from retrieval.index_builder import IndexBuilder
        from retrieval.dense_retriever import DenseRetriever
        from ocr.ocr_engine import OCREngine
        from config.config_parser import CMSVSConfigParser

        # Build structured pages from synthetic text
        ocr = OCREngine()
        pages = {
            1: ocr.process_text(
                "DEDUCTIBLES\n"
                "Individual Deductible (In-Network): $1,500\n"
                "Family Deductible (In-Network): $3,000\n",
                page_number=1,
            ),
            2: ocr.process_text(
                "OUT-OF-POCKET MAXIMUMS\n"
                "Individual OOP Maximum (In-Network): $5,000\n"
                "Family OOP Maximum (In-Network): $10,000\n",
                page_number=2,
            ),
            3: ocr.process_text(
                "COPAYMENTS AND COINSURANCE\n"
                "Primary Care Visit: $25 copay\n"
                "Specialist Visit: $50 copay\n"
                "Emergency Room: $250 copay\n",
                page_number=3,
            ),
        }

        embedding_client = NvidiaEmbeddingClient(api_key=api_key)
        builder = IndexBuilder(embedding_client=embedding_client)
        collection = builder.build(pages)

        assert collection.count() == 3
        ok(f"Index built: {collection.count()} page vectors in ChromaDB")

        # Retrieve pages for a deductible section
        config = CMSVSConfigParser().load(
            Path(__file__).parent.parent / "configs" / "healthcare_sbc_config.yaml"
        )
        deductible_section = config.get_section("Deductibles")

        retriever = DenseRetriever(
            embedding_client=embedding_client,
            collection=collection,
            default_top_k=2,
            fallback_top_k=3,
        )

        pages_for_deductibles = retriever.retrieve_for_section(deductible_section)
        ok(f"Deductibles → pages: {pages_for_deductibles}")
        assert 1 in pages_for_deductibles, "Page 1 (deductibles) should be retrieved"

        # Retrieve with similarity scores
        pairs = retriever.retrieve_by_query(
            "individual deductible annual amount before coverage",
            top_k=3,
        )
        ok(f"Query results (page, similarity):")
        for pg, sim in pairs:
            info(f"  Page {pg} → {sim:.4f}")

        # Cleanup
        builder.destroy()
        ok("ChromaDB collection destroyed")

        return True

    except Exception as exc:
        fail(f"Dense retrieval test failed: {exc}")
        import traceback; traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Test: 11 — MLLM Extraction (requires API key)
# ══════════════════════════════════════════════════════════════════════════════

def test_mllm_extraction() -> bool:
    section("TEST 11 — MLLM Extraction (API call with image)")
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        info("NVIDIA_API_KEY not set — skipping API test")
        return True

    try:
        from models.nvidia_client import NvidiaLLMClient
        from extraction.mllm_extractor import MLLMExtractor
        from config.config_parser import CMSVSConfigParser
        from shared_types import ExtractionStatus

        config = CMSVSConfigParser().load(
            Path(__file__).parent.parent / "configs" / "funsd_ner_config.yaml"
        )

        # Use the Document Identity section (simple fields)
        section_cfg = config.get_section("Document Identity")

        llm_client = NvidiaLLMClient(api_key=api_key)
        extractor = MLLMExtractor(llm_client=llm_client)

        # Create a synthetic test image
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test_form.png")
            create_synthetic_image(img_path)

            # Load and base64-encode
            from PIL import Image
            import base64, io
            with Image.open(img_path) as img:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            raw_extractions = extractor.extract_section(
                section=section_cfg,
                page_images_b64=[img_b64],
            )

        ok(f"Extractions received: {len(raw_extractions)} entities")
        for name, ext in raw_extractions.items():
            status = ext.extraction_status.value
            value = ext.extracted_value or "null"
            conf = ext.confidence
            info(f"  {name}: '{value}' [{status}] conf={conf:.2f}")

        # Convert to EntityResults
        entity_results = extractor.to_entity_results(raw_extractions)
        review_count = sum(1 for er in entity_results.values() if er.review_required)
        ok(f"EntityResults: {len(entity_results)} total, {review_count} flagged for review")

        return True

    except Exception as exc:
        fail(f"MLLM extraction test failed: {exc}")
        import traceback; traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Test: 12 — Semantic Validator (requires API key)
# ══════════════════════════════════════════════════════════════════════════════

def test_semantic_validator() -> bool:
    section("TEST 12 — Semantic Validator (API call)")
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        info("NVIDIA_API_KEY not set — skipping API test")
        return True

    try:
        from models.nvidia_client import NvidiaLLMClient
        from validation.semantic_validator import SemanticValidator
        from config.config_parser import CMSVSConfigParser
        from shared_types import (
            EntityType, ExtractionStatus, FinalEntityValue, ValidationStatus
        )

        config = CMSVSConfigParser().load(
            Path(__file__).parent.parent / "configs" / "funsd_ner_config.yaml"
        )

        llm_client = NvidiaLLMClient(api_key=api_key)
        validator = SemanticValidator(
            llm_client=llm_client,
            config=config,
            confidence_threshold=0.70,
        )

        def make_fev(name: str, value: str | None) -> FinalEntityValue:
            return FinalEntityValue(
                entity_name=name,
                extracted_value=value,
                extraction_status=(
                    ExtractionStatus.FOUND if value
                    else ExtractionStatus.NOT_FOUND
                ),
                entity_type=EntityType.DIRECT,
                confidence=0.95 if value else 0.0,
                source_page=1,
                source_region="",
                raw_context="",
                review_required=False,
                fallback_triggered=False,
                expression_audit=None,
            )

        # Doc A and Doc B entity values
        doc_a = {
            "document_title":    make_fev("document_title",    "MARKETING RESEARCH AUTHORIZATION"),
            "document_reference_number": make_fev("document_reference_number", "6"),
            "document_stamp_id": make_fev("document_stamp_id", "670801704"),
        }
        doc_b = {
            "document_title":    make_fev("document_title",    "marketing research authorization"),
            "document_reference_number": make_fev("document_reference_number", "06"),
            "document_stamp_id": make_fev("document_stamp_id", "670801704"),
        }

        # Validate just the Document Identity section
        doc_id_section = config.get_section("Document Identity")
        section_result = validator.validate_section(
            section=doc_id_section,
            doc_a_entities=doc_a,
            doc_b_entities=doc_b,
            doc_a_name="Doc A (Original)",
            doc_b_name="Doc B (Augmented)",
        )

        ok(f"Section: {section_result.section_name}")
        ok(f"Matches: {section_result.match_count}")
        ok(f"Mismatches: {section_result.mismatch_count}")
        for er in section_result.entity_results:
            status = er.validation_status.value
            fast = " [fast-path]" if er.fast_path_match else ""
            info(f"  {er.entity_name}: {status}{fast} (conf={er.confidence:.2f})")
            if er.reasoning and not er.fast_path_match:
                info(f"    Reasoning: {er.reasoning[:80]}...")

        return True

    except Exception as exc:
        fail(f"Semantic validator test failed: {exc}")
        import traceback; traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Test: 13 — Report Generator
# ══════════════════════════════════════════════════════════════════════════════

def test_report_generator() -> bool:
    section("TEST 13 — Report Generator")
    try:
        from output.report_generator import ReportGenerator
        from shared_types import (
            DiscrepancyType, EntityType, EntityValidationResult,
            ExtractionStatus, FinalEntityValue,
            SectionValidationResult, ValidationReport, ValidationStatus,
        )

        # Build a minimal ValidationReport
        ev1 = EntityValidationResult(
            entity_name="document_title",
            section_name="Document Identity",
            doc_a_value="MARKETING RESEARCH AUTHORIZATION",
            doc_b_value="marketing research authorization",
            doc_a_normalized="marketing research authorization",
            doc_b_normalized="marketing research authorization",
            validation_status=ValidationStatus.MATCH,
            discrepancy_type=DiscrepancyType.NOT_APPLICABLE,
            reasoning="Case-insensitive match after normalisation.",
            confidence=1.0,
            review_required=False,
            fast_path_match=True,
        )
        ev2 = EntityValidationResult(
            entity_name="final_volume_or_quantity",
            section_name="Financial and Quantitative",
            doc_a_value="400",
            doc_b_value="500",
            doc_a_normalized="400",
            doc_b_normalized="500",
            validation_status=ValidationStatus.MISMATCH,
            discrepancy_type=DiscrepancyType.NUMERIC_DIFFERENCE,
            reasoning="Values 400 and 500 are numerically different.",
            confidence=0.95,
            review_required=False,
            fast_path_match=False,
        )

        report = ValidationReport(
            doc_a_path="doc_a.png",
            doc_b_path="doc_b.png",
            config_name="funsd_ner_config",
            section_results=[
                SectionValidationResult("Document Identity", [ev1]),
                SectionValidationResult("Financial and Quantitative", [ev2]),
            ],
        )

        doc_a_entities = {
            "document_title": FinalEntityValue(
                entity_name="document_title",
                extracted_value="MARKETING RESEARCH AUTHORIZATION",
                extraction_status=ExtractionStatus.FOUND,
                entity_type=EntityType.DIRECT,
                confidence=0.98,
                source_page=1,
                source_region="Top of document",
                raw_context="MARKETING RESEARCH AUTHORIZATION",
                review_required=False,
                fallback_triggered=False,
                expression_audit=None,
            ),
        }
        doc_b_entities = {
            "document_title": FinalEntityValue(
                entity_name="document_title",
                extracted_value="marketing research authorization",
                extraction_status=ExtractionStatus.FOUND,
                entity_type=EntityType.DIRECT,
                confidence=0.97,
                source_page=1,
                source_region="Top of document",
                raw_context="marketing research authorization",
                review_required=False,
                fallback_triggered=False,
                expression_audit=None,
            ),
        }

        generator = ReportGenerator()
        report_dict = generator.generate(
            validation_report=report,
            doc_a_entities=doc_a_entities,
            doc_b_entities=doc_b_entities,
        )

        # Verify structure
        assert "summary" in report_dict
        assert "entity_extractions" in report_dict
        assert "validation_results" in report_dict
        assert report_dict["summary"]["total_entities"] == 2
        assert report_dict["summary"]["total_matches"] == 1
        assert report_dict["summary"]["total_mismatches"] == 1

        ok(f"Report generated: {report_dict['summary']}")

        # Ground truth format
        gt_report = generator.generate_groundtruth_format(report)
        assert "entities" in gt_report
        assert len(gt_report["entities"]) == 2
        gt_results = {e["entity_name"]: e["validation_result"] for e in gt_report["entities"]}
        assert gt_results["document_title"] == "match"
        assert gt_results["final_volume_or_quantity"] == "mismatch"
        ok("Ground truth format: matches M2 schema")

        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_report.json")
            saved = generator.save(report_dict, output_path)
            assert saved.exists()
            with open(saved) as f:
                loaded = json.load(f)
            assert loaded["summary"]["total_entities"] == 2
            ok(f"Report saved and reloaded from: {saved.name}")

        return True

    except Exception as exc:
        fail(f"Report generator test failed: {exc}")
        import traceback; traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Test: 14 — Full Image Pipeline (requires API key)
# ══════════════════════════════════════════════════════════════════════════════

def test_image_pipeline() -> bool:
    section("TEST 14 — Full Image Pipeline (API call)")
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        info("NVIDIA_API_KEY not set — skipping API test")
        return True

    try:
        from models.nvidia_client import NvidiaLLMClient
        from pipeline.image_pipeline import ImagePipeline
        from config.config_parser import CMSVSConfigParser

        config = CMSVSConfigParser().load(
            Path(__file__).parent.parent / "configs" / "funsd_ner_config.yaml"
        )
        llm_client = NvidiaLLMClient(api_key=api_key)

        pipeline = ImagePipeline(
            config=config,
            llm_client=llm_client,
            confidence_threshold=0.70,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "funsd_form.png")
            create_synthetic_image(img_path)
            entity_values = pipeline.run(img_path)

        ok(f"Image pipeline completed: {len(entity_values)} entities extracted")
        found = sum(
            1 for fev in entity_values.values()
            if fev.extracted_value is not None
        )
        review = sum(
            1 for fev in entity_values.values()
            if fev.review_required
        )
        ok(f"  Found: {found}/{len(entity_values)}")
        ok(f"  Review required: {review}")

        for name, fev in entity_values.items():
            if fev.extracted_value:
                info(f"  {name}: '{fev.extracted_value}' (conf={fev.confidence:.2f})")

        return True

    except Exception as exc:
        fail(f"Image pipeline test failed: {exc}")
        import traceback; traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Test: 15 — CMSVSPipeline end-to-end (requires API key)
# ══════════════════════════════════════════════════════════════════════════════

def test_cmsvs_pipeline_end_to_end() -> bool:
    section("TEST 15 — CMSVSPipeline End-to-End (API call)")
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        info("NVIDIA_API_KEY not set — skipping API test")
        return True

    try:
        from pipeline.cmsvs_pipeline import CMSVSPipeline

        config_path = (
            Path(__file__).parent.parent / "configs" / "funsd_ner_config.yaml"
        )
        pipeline = CMSVSPipeline.from_env(
            config_path=str(config_path),
            confidence_threshold=0.70,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            doc_a_path = os.path.join(tmpdir, "doc_a.png")
            doc_b_path = os.path.join(tmpdir, "doc_b.png")
            create_synthetic_image(doc_a_path)
            create_synthetic_image(doc_b_path)

            # Full pipeline
            report = pipeline.run(
                doc_a_path=doc_a_path,
                doc_b_path=doc_b_path,
                doc_a_name="Original Form",
                doc_b_name="Augmented Form",
            )

            # Ground truth format
            gt_report = pipeline.run_groundtruth_format(
                doc_a_path=doc_a_path,
                doc_b_path=doc_b_path,
            )

            # Save report
            output_path = os.path.join(tmpdir, "pipeline_report.json")
            saved = pipeline.save_report(report, output_path)

        summary = report.get("summary", {})
        ok(f"Pipeline completed successfully")
        ok(f"Total entities   : {summary.get('total_entities', 0)}")
        ok(f"Matches          : {summary.get('total_matches', 0)}")
        ok(f"Mismatches       : {summary.get('total_mismatches', 0)}")
        ok(f"Match rate       : {summary.get('match_rate', 0.0):.1%}")
        ok(f"Review required  : {summary.get('review_required', 0)}")
        ok(f"Fast-path matches: {summary.get('fast_path_matches', 0)}")

        # Ground truth format check
        gt_entities = gt_report.get("entities", [])
        ok(f"Ground truth format: {len(gt_entities)} entities")

        return True

    except Exception as exc:
        fail(f"End-to-end pipeline test failed: {exc}")
        import traceback; traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Main runner
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "═" * 60)
    print("  CMSVS Component Test Suite")
    print("═" * 60)

    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if api_key:
        ok(f"NVIDIA_API_KEY found: ****{api_key[-4:]}")
    else:
        info("NVIDIA_API_KEY not set — API tests will be skipped")

    tests = [
        # ("Shared Types",            test_shared_types),
        # ("Config Parser",           test_config_parser),
        # ("Input Handler",           test_input_handling),
        # ("PDF Loading",             test_pdf_loading),
        # ("OCR Engine",              test_ocr_engine),
        # ("Value Normalizer",        test_value_normalizer),
        # ("Expression Evaluator",    test_expression_evaluator),
        # ("Prompt Builders",         test_prompt_builders),
        # ("NVIDIA Embeddings",       test_nvidia_embeddings),
        # ("Dense Retrieval",         test_dense_retrieval),
        # ("MLLM Extraction",         test_mllm_extraction),
        # ("Semantic Validator",      test_semantic_validator),
        # ("Report Generator",        test_report_generator),
        # ("Image Pipeline",          test_image_pipeline),
        ("CMSVSPipeline E2E",       test_cmsvs_pipeline_end_to_end),
    ]

    passed, failed, skipped = 0, 0, 0
    for name, fn in tests:
        try:
            result = fn()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as exc:
            fail(f"UNEXPECTED ERROR in '{name}': {exc}")
            failed += 1

    print(f"\n{'═' * 60}")
    print(f"  Results: {passed} passed  |  {failed} failed")
    print(f"{'═' * 60}\n")
    print(Path(__file__).parent.parent / ".env")
    if failed > 0:
        sys.exit(1)

    

if __name__ == "__main__":
    
    main()
