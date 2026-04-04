
## Milestone 4: Model Training and Experiments

**Cross-Document Validation using Document Intelligence**

*Mallesh Mayara (21f2001118) · Mayank Dode (22f1000781) · Karthik Ganesh (21f2000775) · Ayush Verma (21f3000500)*

---

## Abstract

This document presents the work completed in Milestone 4 of the Configurable Multimodal Semantic Validation System (CMSVS) project. The primary achievement of this milestone is the delivery of a **fully working end-to-end solution** — the complete pipeline from raw document input to structured validation report is operational and demonstrated via a local Streamlit interface.

A key architectural decision made during this milestone is the **consolidation of the entire inference stack under NVIDIA NIM free-tier services exclusively**. All embedding calls use **llama-3.2-nemoretriever-300m-embed-v1** and all MLLM/LLM inference calls use **llama-4-maverick-17b-128e-instruct** — both served through the NVIDIA NIM free-tier endpoint. This eliminates provider-switching complexity and concentrates all inference through a single reliable provider.

Systematic fine-tuning, formal accuracy evaluation, and detailed error analysis are planned for Milestone 5.

---

## 1. Milestone Objectives

Milestone 4 is defined by the following requirements per the project guidelines:

> **(1) Train initial models**
> **(2) Experiment with hyperparameters, optimization methods, and regularization techniques**

In the context of the CMSVS system — which leverages pre-trained LLMs via API rather than gradient-based training — these requirements map to:

- **Working Solution Delivery:** A fully functional end-to-end pipeline running locally with a Streamlit demo interface
- **Model Selection:** Evaluating and finalising the inference model for extraction and validation tasks
- **Initial Experimentation:** Preliminary parameter choices for retrieval, confidence thresholds, and prompt design
- **Provider Consolidation:** Simplifying the inference stack to a single provider for reliability and maintainability

Formal hyperparameter tuning, accuracy benchmarking, and systematic ablation studies are scoped for **Milestone 5**.

---

## 2. Provider Consolidation: NVIDIA NIM Exclusive Stack

### 2.1 Decision

The M3 architecture specified Groq API as the primary LLM provider with NVIDIA NIM as fallback. During M4 implementation, the team consolidated to **NVIDIA NIM exclusively** for the following reasons:

- The NVIDIA NIM free tier provides sufficient throughput and daily quota for the entire project evaluation dataset
- Removing the dual-provider design eliminates provider-switching logic, simplifies the codebase, and removes a source of inconsistent behavior between runs
- A single API key and a single client class reduces configuration overhead
- All required capabilities — dense embeddings and multimodal LLM inference — are available under the same provider

### 2.2 Consolidated Model Stack

| Function | Model | Provider | Cost |
|---|---|---|---|
| Dense Embeddings | llama-3.2-nemoretriever-300m-embed-v1 | NVIDIA NIM (Free) | $0 |
| MLLM Extraction (Visual) | llama-4-maverick-17b-128e-instruct | NVIDIA NIM (Free) | $0 |
| LLM Validation (Text) | llama-4-maverick-17b-128e-instruct | NVIDIA NIM (Free) | $0 |

### 2.3 Why llama-4-maverick-17b-128e-instruct

| Criterion | Assessment |
|---|---|
| Multimodal capability | Native vision support — accepts base64-encoded page images directly |
| Context window | 128K tokens — handles dense multi-column SBC documents without truncation |
| JSON output reliability | Consistent structured JSON responses with low schema violation rate |
| Instruction following | Follows multi-step CoT validation prompt sequences accurately |
| Architecture | Mixture-of-Experts (128 experts) — activates specialized pathways for structured document tasks |
| Free-tier availability | Available on NVIDIA NIM at zero cost |

---

## 3. Working Solution Overview

### 3.1 What Has Been Built

The complete CMSVS pipeline is operational end-to-end. The following capabilities are working in the current implementation:

- ✅ PDF and image document ingestion
- ✅ PaddleOCR-based text extraction for dense vector indexing
- ✅ NVIDIA NIM embedding-based RAG page routing via ChromaDB
- ✅ Section-batched MLLM visual extraction using llama-4-maverick
- ✅ Expression entity computation via SimpleEval
- ✅ Rule-based value pre-normalization
- ✅ Section-wise Chain-of-Thought semantic validation
- ✅ Structured JSON validation report generation
- ✅ Streamlit local demo interface

### 3.2 End-to-End Pipeline Flow (As Implemented)

```
Document A (PDF/Image) ──┐
                          ├──► Input Routing
Document B (PDF/Image) ──┘         │
                                    ▼
                          ┌─── PDF Path ───┐
                          │  OCR → Index   │
                          │  RAG Retrieval │
                          └───────┬────────┘
                                  │  (Image path bypasses above)
                                  ▼
                          MLLM Extraction
                          (llama-4-maverick)
                                  │
                                  ▼
                          Expression Engine
                          (SimpleEval)
                                  │
                                  ▼
                          Section-Wise Validation
                          (CoT via llama-4-maverick)
                                  │
                                  ▼
                          Structured JSON Report
                          + Streamlit Display
```

### 3.3 Streamlit Interface

The local Streamlit interface provides:

- File upload for Document A and Document B (PDF or image)
- Configuration file selection
- Real-time pipeline execution with progress indicators
- Extraction results display per document
- Section-wise validation report with per-entity status
- Downloadable JSON report output

---

## 4. Initial Parameter Choices

The following parameters were selected based on initial testing during M4 implementation. These represent starting values that will be systematically evaluated in Milestone 5.

| Parameter | Current Value | Rationale |
|---|---|---|
| Retrieval top-K | 3 | Balances page coverage against token consumption |
| Fallback top-K | 5 | Expanded retrieval on low-confidence extraction |
| Confidence threshold | 0.75 | Mid-point starting value for escalation trigger |
| Extraction prompt style | Structured with CoT steps | Initial testing showed better table parsing |
| Validation granularity | Section-wise | Confirmed from M3 design — retains relational context |
| Max JSON retry attempts | 3 | Sufficient to recover from transient schema violations |

These parameter choices will be subject to systematic ablation experiments in M5 to identify optimal values with quantitative justification.

---

## 5. Initial Observations

During M4 implementation and initial test runs on a small subset of document pairs, the following qualitative observations were noted. These will be formally measured and analyzed in M5.

**Extraction:**
- The MLLM accurately identifies and extracts values from standard two-column (In-Network / Out-of-Network) SBC table layouts
- Documents with three-tier network structures occasionally produce ambiguous entity-to-column mappings — identified as a prompt engineering improvement area for M5
- EXPRESSION variable extraction performs well when component values appear in clearly separated table cells

**Validation:**
- The rule-based pre-normalization fast path correctly handles the majority of format-only differences without needing an MLLM call
- Section-wise CoT validation produces readable, traceable reasoning chains that clearly identify the basis for each MATCH or MISMATCH decision
- Coverage classification entities (covered / not covered) are handled reliably by the equivalents mapping

**System Behavior:**
- Pipeline completes a full PDF document pair in approximately 25–35 seconds end-to-end on the local Streamlit interface
- NVIDIA NIM rate limits were encountered during rapid successive test runs — mitigated by adding a short inter-call delay

---

## 6. Pending Items for Milestone 5

The following items are explicitly scoped for M5:

| Item | Description |
|---|---|
| Retrieval top-K ablation | Systematic comparison of K=2, K=3, K=4, K=5 on full evaluation dataset |
| Confidence threshold tuning | Precision-recall tradeoff analysis across threshold values |
| Prompt template comparison | Structured vs CoT extraction prompt accuracy comparison |
| Token consumption profiling | Per-stage, per-document-pair token measurement |
| Formal accuracy evaluation | Precision, recall, F1-score against M2 ground truth annotations |
| Error analysis | Root cause attribution for extraction and validation failures |
| Numeric parser improvements | Handling non-standard monetary value formats in expression variables |
| Coverage equivalents expansion | Additional insurance industry abbreviations and phrasings |

---

## 7. Directory Structure

```
cmsvs/
├── README.md
├── requirements.txt
├── .env.example
├── worklog.md
│
├── configs/
│   ├── healthcare_sbc_config.yaml
│   ├── funsd_ner_config.yaml
│   └── schema/
│       ├── config_schema.json
│       └── output_schema.json
│
├── src/
│   ├── shared_types.py
│   ├── input/
│   │   ├── input_handler.py
│   │   └── image_loader.py
│   ├── ingestion/
│   │   ├── document_processor.py
│   │   └── page_image_store.py
│   ├── ocr/
│   │   └── ocr_engine.py
│   ├── retrieval/
│   │   ├── index_builder.py
│   │   └── dense_retriever.py
│   ├── config/
│   │   └── config_parser.py
│   ├── prompts/
│   │   ├── ner_prompt_builder.py
│   │   └── validation_prompt_builder.py
│   ├── models/
│   │   └── nvidia_client.py
│   ├── extraction/
│   │   ├── mllm_extractor.py
│   │   ├── expression_evaluator.py
│   │   └── expression_orchestrator.py
│   ├── validation/
│   │   ├── utils/
│   │   │   └── value_normalizer.py
│   │   └── semantic_validator.py
│   ├── output/
│   │   └── report_generator.py
│   └── pipeline/
│       ├── pdf_pipeline.py
│       ├── image_pipeline.py
│       └── cmsvs_pipeline.py
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── funsd/
│   │   ├── original/
│   │   └── augmented/
│   └── healthcare_sbc/
│       ├── doc_a_sbc/
│       ├── doc_b_benefit_grids/
│       │   ├── unaugmented/
│       │   └── augmented/
│       └── ground_truth/
│
├── tests/
│   ├── test_input_handler.py
│   ├── test_config_parser.py
│   ├── test_ocr_engine.py
│   ├── test_index_builder.py
│   ├── test_dense_retriever.py
│   ├── test_expression_evaluator.py
│   ├── test_mllm_extractor.py
│   ├── test_semantic_validator.py
│   └── test_end_to_end.py
│
├── notebooks/
│   ├── 01_input_routing_demo.ipynb
│   ├── 02_pdf_rag_pipeline_demo.ipynb
│   ├── 03_image_direct_pipeline_demo.ipynb
│   ├── 04_expression_extraction_demo.ipynb
│   └── 05_end_to_end_validation_demo.ipynb
│
└── scripts/
    └── run_pipeline.py
```

---

## 8. Team Contributions

| Member | Milestone 4 Contributions |
|---|---|
| **Mayank Dode** (22f1000781) | Input routing layer implementation and refinements; PDF rendering and image preprocessing for multi-format inputs; input handler integration testing for the consolidated NVIDIA NIM pipeline |
| **Ayush Verma** (21f3000500) | Extraction layer implementation with llama-4-maverick response format compatibility; expression evaluator integration and testing; initial extraction accuracy observations on test document pairs |
| **Karthik Ganesh** (21f2000775) | Dense retrieval architecture implementation (IndexBuilder with ChromaDB, DenseRetriever with NemoRetriever embeddings); NVIDIA NIM provider consolidation — Groq client removal and unified NVIDIA NIM client implementation; API provider evaluation and llama-4-maverick selection rationale; retrieval testing and demo notebook |
| **Mallesh Mayara** (21f2001118) | Configuration architecture (CMSVSConfigParser, EntityConfig dataclasses, SectionConfig); healthcare SBC and FUNSD YAML configuration files; shared type contracts (shared_types.py); report generator; JSON schema definitions; section-wise semantic validation architecture (SemanticValidator, CoT validation prompt builder, ValueNormalizer rule engine); pipeline integration (PDFPipeline, ImagePipeline, CMSVSPipeline master orchestrator); Streamlit demo interface; end-to-end testing; milestone documentation |

---

## 9. Milestone Summary

### Completed Objectives
- ✅ Full end-to-end working solution implemented and running locally via Streamlit
- ✅ NVIDIA NIM exclusive inference stack — embeddings and MLLM/LLM under single provider
- ✅ llama-4-maverick-17b-128e-instruct selected and integrated for extraction and validation
- ✅ All pipeline layers operational — input routing, OCR, retrieval, extraction, expression engine, validation, report generation
- ✅ Initial parameter values established as baseline for M5 experimentation

### What Comes Next — Milestone 5
- 🔄 Systematic parameter tuning — retrieval top-K, confidence threshold, prompt templates
- 🔄 Formal accuracy evaluation against M2 ground truth annotations
- 🔄 Token consumption profiling per pipeline stage
- 🔄 Error analysis and targeted accuracy improvements
- 🔄 Precision, recall, and F1-score reporting per entity and per scenario type

---

*Document Version: 1.0 | Milestone: M4 — Model Training and Experiments*
*Indian Institute of Technology Madras — Deep Learning / Generative AI Course Project*