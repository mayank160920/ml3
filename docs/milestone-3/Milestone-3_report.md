# Configurable Multimodal Semantic Validation System: Doc-vs-Doc

## Milestone 3: Model Architecture

**Cross-Document Validation using Document Intelligence**

*Mallesh Mayara (21f2001118) · Mayank Dode (22f1000781) · Karthik Ganesh (21f2000775) · Ayush Verma (21f3000500)*

---

## Abstract

This document presents the complete model architecture designed and implemented for Milestone 3 of the Configurable Multimodal Semantic Validation System (CMSVS) project. Building directly on the datasets prepared in Milestone 2, this milestone defines, justifies, and establishes the end-to-end technical pipeline for two core system capabilities: **(1) Config-Driven Custom Named Entity Recognition (NER)** and **(2) Section-Wise Chain-of-Thought Semantic Validation** across heterogeneous document pairs. The architecture adopts a cost-first design philosophy, leveraging free-tier API services throughout — specifically **Groq API** and **NVIDIA NIM free-tier API** for multimodal LLM inference, **NVIDIA's llama-3.2-nemoretriever-300m-embed-v1** for dense semantic embeddings powering the retrieval layer, and **PaddleOCR Mobile** for lightweight text detection — to deliver enterprise-grade document intelligence at near-zero inference cost. A dense vector retrieval layer using semantic embeddings routes MLLM calls to only the most relevant document pages, achieving an estimated 90% reduction in token consumption compared to naive full-document approaches. The system accepts both PDF and image inputs, applying full RAG-assisted extraction for PDFs and a direct visual extraction path for image inputs. Expression-based entity derivation is supported via the SimpleEval library for entities whose values must be mathematically computed from other extracted values. Validation is performed section-wise, mirroring the logical structure of the configuration file and enabling granular per-section accuracy reporting.

---

## 1. Milestone Objectives

Milestone 3 is defined by three primary requirements per the project guidelines:

> **(1) Select or design appropriate model architecture(s)**
> **(2) Justify choice of architecture**
> **(3) End-to-End setup**

These requirements are addressed across the following dimensions in this document:

- **Architecture Selection:** Every model, library, and component chosen for the CMSVS pipeline is identified with explicit justification for why it was selected over alternatives.
- **Architecture Design:** The novel system-level design — RAG-assisted page routing via dense retrieval, expression-based entity derivation, and section-wise semantic validation — is documented in full.
- **End-to-End Setup:** The complete pipeline from raw document input to structured validation report is specified, with all integration points between components defined.

The architecture is directly informed by the datasets prepared in Milestone 2. The four augmentation categories created in M2 — semantic paraphrase changes, numeric format variants, value conflicts, and coverage reclassification — correspond precisely to the four validation scenario types the M3 architecture is designed to handle. This traceability from dataset to architecture to evaluation is a deliberate design principle.

---

## 2. System Design Philosophy

### 2.1 Core Design Principles

The CMSVS architecture is governed by four principles that informed every component selection and design decision:

**Principle 1 — Cost-First Design**
Every component is selected for minimum cost without sacrificing accuracy. Free-tier API services are used wherever available. Expensive LLM calls are minimized through intelligent retrieval-based routing. Local computation replaces API calls wherever feasible.

**Principle 2 — Clean Separation of Responsibilities**
Each component in the pipeline has exactly one job. The OCR layer finds text for indexing. The retrieval layer finds relevant pages. The MLLM layer understands and extracts content visually. The expression evaluator computes derived values. The validator compares values semantically. No component performs work outside its designated responsibility.

**Principle 3 — Configurable Without Code Changes**
Every domain-specific behavior is controlled by the YAML configuration file. Switching from healthcare insurance validation to logistics purchase order validation requires only a new configuration file — no model retraining, no code modification, no infrastructure change.

**Principle 4 — Auditable by Design**
Every extraction and validation decision must be traceable to a specific location in a specific source document. The system is designed for regulated enterprise environments where decisions must be explainable and attributed to evidence.

### 2.2 The Two-Task Architecture

The system solves two distinct tasks executed sequentially for each document pair:

**Task 1 — Custom NER:** Extract the value of each configured entity from each document. The extraction path varies based on input type (PDF or Image) and entity type (DIRECT or EXPRESSION).

**Task 2 — Section-Wise Semantic Validation:** For each section defined in the configuration, compare the extracted entity values from Document A against the corresponding values from Document B, producing a structured validation decision per entity with Chain-of-Thought reasoning.

---

## 3. Complete Architecture Overview

The CMSVS pipeline consists of seven functional layers executed in sequence:

```
Layer 1:  Input Routing         — Detect PDF vs Image, load accordingly
Layer 2:  OCR Processing        — Extract text for indexing (PDF path only)
Layer 3:  Dense Vector Indexing — Build semantic embedding index per page
Layer 4:  RAG Page Routing      — Retrieve relevant pages via dense search
Layer 5:  MLLM Extraction       — Visual entity extraction from page images
Layer 6:  Expression Engine     — Compute derived entity values via SimpleEval
Layer 7:  Section-Wise Validation — CoT semantic comparison per section
```
![E2E Design](img/E2E%20Design.jpeg)

### 3.1 Input Routing Layer

The pipeline accepts two input types and routes each through the appropriate processing path:

**PDF Input Path — Full RAG Pipeline**
When a PDF file is provided, the system executes the complete pipeline. OCR processing produces structured text used to build the dense vector index. The retrieval layer identifies the most relevant pages per section. Only those pages are passed to the MLLM for extraction, minimizing token consumption for multi-page documents.

**Image Input Path — Direct MLLM Path**
When an image file is provided (JPEG, PNG, TIFF, BMP, or WebP), the system bypasses OCR and retrieval entirely. The image is loaded, base64-encoded, and passed directly to the MLLM with all section entities extracted in a single call. This path is optimal for single-page documents and scanned form images such as those in the FUNSD dataset.

Both paths produce an identical output structure — a dictionary of FinalEntityValue objects keyed by entity name — ensuring the downstream validation layer operates identically regardless of input type.

### 3.2 OCR Processing Layer

**Selected Tool: PaddleOCR Mobile Model**

PaddleOCR's mobile-optimized text detection and recognition model is used for the OCR layer. Its role is strictly limited to producing text for the dense vector index — it is never used as the source of extracted entity values.

| Criterion | PaddleOCR Mobile | Justification |
|---|---|---|
| Deployment | Local inference | Zero API cost, no network dependency |
| Speed | Fast — mobile-optimized weights | Low latency for the indexing phase |
| Hardware | CPU-compatible | No GPU requirement |
| Text accuracy | Adequate for semantic indexing | Exact layout not needed — text feeds embeddings only |
| License | Apache 2.0 | Free for all use cases |

**Critical Design Decision:** Because the MLLM always reads raw page images for final extraction, PaddleOCR's text-only capability is entirely sufficient. Any OCR imperfections affect only the retrieval index quality — and the dense embedding model is robust enough to handle minor OCR noise through semantic similarity rather than exact matching.

For each document page, PaddleOCR Mobile produces:
- Raw extracted text covering the full page content
- Detected key-value patterns using Label: Value structure recognition
- Section header candidates identified by heuristics — short lines with uppercase or title-case formatting
- A composite index text combining all of the above in a structured format optimized for embedding

### 3.3 Dense Vector Indexing Layer

**Selected Model: llama-3.2-nemoretriever-300m-embed-v1 via NVIDIA NIM Free Tier**

Each page's composite index text is embedded using NVIDIA's NemoRetriever model to produce a dense semantic vector. These vectors are stored in an in-memory ChromaDB collection keyed by page number.

**Why Dense-Only Retrieval:**

The decision to use dense vector retrieval without a keyword-based component reflects the primary retrieval challenge in this domain. The core difficulty in routing to the correct page is not finding exact keyword matches — it is bridging the semantic gap between the entity descriptions written in the configuration file and the varied terminology used in real insurance documents. Dense retrieval excels precisely at this task because it operates on meaning rather than surface form.

For example, a configuration entity described as "the annual amount a member must pay before insurance begins covering costs" must correctly route to a page that contains "deductible" — a term not present in the description at all. Dense embeddings capture this semantic relationship naturally. A keyword approach would fail here entirely.

Additionally, numeric values and exact terms that dense retrieval might miss in edge cases are handled downstream by the MLLM, which reads the raw page image directly and is not dependent on the retrieval index for precise value matching.

**Why llama-3.2-nemoretriever-300m-embed-v1:**

| Criterion | Selection | Justification |
|---|---|---|
| Cost | Free — NVIDIA NIM free tier | Zero embedding cost for the entire project |
| Training objective | Retrieval-specific | Trained to maximize semantic similarity for relevant passages, not general language modeling |
| Model size | 300M parameters | Fast inference, low memory footprint |
| Provider reliability | NVIDIA NIM enterprise API | Stable, rate-limit-generous free tier |
| Vector quality | High on domain-specific retrieval | Outperforms general-purpose embeddings on structured document retrieval tasks |

**Index Structure:**
Each document produces one temporary in-memory ChromaDB collection. The collection stores one vector per page with the page number embedded in the metadata. The collection is destroyed after the document is fully processed — there is no persistent storage requirement and no cross-document contamination risk.

### 3.4 RAG Page Routing Layer

The retrieval layer's sole responsibility is to return relevant page numbers. It performs no extraction and calls no paid inference API beyond the NVIDIA NIM embedding endpoint.

**Retrieval Query Construction:**
For each section in the configuration file, a retrieval query is constructed by combining the following signals:
- Section name with underscores replaced by spaces
- Section keywords defined in the configuration for semantic anchoring
- All entity names in the section
- All entity descriptions in the section
- All entity example values for format and domain context

This multi-signal query design ensures the embedding captures both the structural signal (what section this is) and the semantic signal (what types of values are being sought), maximizing retrieval accuracy.

**Dense Retrieval Process:**
The query text is embedded using the same NemoRetriever model used for document indexing, ensuring the query and document vectors occupy the same semantic space. ChromaDB performs cosine similarity search against all page vectors in the collection, returning the top-K pages ranked by similarity score.

The default top-K is 2, meaning the two most semantically relevant pages are returned for each section. This value is configurable and is expanded to 4 during fallback processing.

**Fallback Mechanism:**
If the MLLM extraction on the top-2 pages returns a confidence score below 0.75 for any entity, the retriever is queried again with top-K expanded to 4, and extraction is retried on the larger page set. If confidence remains below threshold after fallback, the entity is flagged for human review and preserved in the output report with its best available extraction and a review_required flag.

![E2E Design](img/extraction%20fallback%20strategy.jpeg)

### 3.5 MLLM Extraction Layer

**Primary Provider: Groq API (Free Tier)**
**Secondary Provider: NVIDIA NIM API (Free Tier)**

The MLLM extraction layer is responsible for visual understanding of document page images — the only component in the pipeline that requires genuine AI reasoning about document content.

**Provider Selection:**

| Provider | Model | Role | Justification |
|---|---|---|---|
| Groq (Free) | llama-3.3-70b-versatile | Primary extraction and validation | Free tier, exceptionally high throughput via LPU architecture, strong JSON output reliability |
| Groq (Free) | mixtral-8x7b-32768 | Alternative for long-context pages | Extended context window for dense multi-column pages |
| NVIDIA NIM (Free) | meta-llama/Llama-4-Scout-17B-16E | Fallback provider | Independent infrastructure, free tier, strong instruction following |

**Why Groq as Primary Provider:**
Groq's LPU (Language Processing Unit) architecture delivers inference at 500–800 tokens per second, compared to 30–80 tokens per second for GPU-based API providers. For a pipeline making 5–10 LLM calls per document pair across extraction and validation stages, this speed advantage translates to significantly shorter end-to-end processing time. Groq's free tier provides sufficient daily token allowance to process the full 20-document-pair evaluation dataset without cost.

**Why Visual Extraction Over Text Extraction:**
Document page images are passed directly to the MLLM rather than OCR-extracted text for all entity extraction. This design choice addresses documented failure modes observed during M2 dataset preparation:

- Multi-column SBC PDFs processed by text extraction libraries produce column-interleaved output where adjacent column content is concatenated, garbling table row logic
- Spatial relationships between table headers and cell values are lost in flat text representation
- Visual hierarchy signals — font size, bold text, table borders, column alignment — carry semantic meaning that text-only processing discards
- Scanned documents with OCR noise produce degraded text that misleads downstream extraction

The MLLM reads page images as a human reader would, using all available visual and spatial context for accurate extraction.

**Section-Batched Extraction:**
All entities within a single configuration section are extracted in one MLLM call, with the page images for that section passed alongside all entity definitions. This section-batching reduces MLLM API calls from N_entities to N_sections. For an 18-entity, 5-section configuration, this reduces extraction calls from 18 to 5 per document — a 72% reduction in API calls.

**Entity Extraction Types:**

*DIRECT Entities:* The MLLM locates and extracts the value exactly as it appears in the document image. The prompt provides entity name, natural language description, extraction logic hint from the configuration, and an example value format for output calibration.

*EXPRESSION Variable Entities:* For entities declared with EXPRESSION logic, the MLLM extracts each component variable value rather than a final computed result. These component values are labeled clearly in the prompt as expression variables destined for mathematical computation. The MLLM's role is accurate visual extraction of raw component values — computation is handled separately by SimpleEval.

**Output Schema Enforcement:**
The MLLM is instructed to return a strict JSON object for each extraction containing: entity name, extracted value (or null), extraction status, source page number, source region description, confidence score (0.0 to 1.0), and raw surrounding context for audit. Responses that fail JSON parsing trigger a retry with explicit schema re-instruction, up to three attempts before a fallback null response is recorded.

### 3.6 Expression Engine Layer

**Selected Tool: SimpleEval Library**

For entities where the configuration specifies `entity_extraction_logic: EXPRESSION`, the system computes the final entity value from extracted component variables using the SimpleEval sandboxed expression evaluator.

**Why SimpleEval Over Python eval():**
Python's built-in `eval()` can execute arbitrary code, presenting an unacceptable security risk in a system where configuration files are authored by domain experts. A malformed or malicious expression template could execute system commands, access files, or import dangerous modules. SimpleEval enforces a strict whitelist of permitted operations, preventing all code execution outside safe arithmetic and mathematical functions.

**Supported Operations:**
- Arithmetic operators: addition, subtraction, multiplication, division, exponentiation
- Comparison operators: greater than, less than, equality, inequality
- Safe functions: round(), abs(), min(), max(), sum(), sqrt(), ceil(), floor()
- Conditional expressions: value_if_true if condition else value_if_false

**Expression Computation Flow:**
1. Configuration defines expression template and variable list with descriptions
2. MLLM extracts each variable as a pseudo-entity from document page images
3. Extracted string values are parsed to numeric, handling currency symbols, commas, and percentage notation
4. SimpleEval evaluates the template with substituted numeric variable values
5. Result is formatted according to entity data_type — monetary as $X,XXX.XX, percentage as XX.XX%
6. FinalEntityValue is produced with full audit trail: template used, variable values, and computed result
7. Confidence score is derived as the average of component variable extraction confidences

**Healthcare Domain Example:**
A Total Family Deductible entity may not appear explicitly in an SBC document. The document shows Tier 1 Individual Deductible ($1,500) and Tier 2 Individual Deductible ($2,000) separately. The EXPRESSION configuration computes the combined value (3,500.00 USD) from the two extracted variables, producing a value directly comparable against the Benefit Grid's stated combined deductible figure.

**Error Handling:**
If any required variable value is missing or unparseable, the entity receives ERROR status with a descriptive message identifying the missing variable. The error is recorded in the audit trail without crashing the pipeline. The entity is flagged for human review automatically.

### 3.7 Section-Wise Semantic Validation Layer

Validation is performed section by section, mirroring the logical organization of the configuration file. This architectural decision reflects how domain experts naturally review document consistency — by logical grouping, not by isolated entity comparisons.

**Why Section-Wise Validation:**

*Context provision:* Validating all deductible-related entities simultaneously gives the MLLM the relational context of how those values interact. An unusual individual deductible value makes more sense when seen alongside the family deductible in the same prompt.

*Cost efficiency:* One section-level validation call replaces multiple entity-level calls. For 18 entities across 5 sections, this reduces validation API calls from 18 to 5 — a 72% reduction in the validation stage.

*Structured reporting:* Section-wise organization of the output report mirrors how benefits administrators and compliance officers review SBC accuracy, making the output immediately interpretable to domain users.

**Rule-Based Pre-Normalization (Fast Path):**
Before any MLLM validation call, a rule-based normalizer processes all value pairs in the section deterministically:

- Monetary normalization: "$1,500" → "1500.00 USD", "1500" → "1500.00 USD", "$1,500.00" → "1500.00 USD"
- Percentage normalization: "20%" → "20.0%", "0.20" → "20.0%", "20 percent" → "20.0%"
- Coverage equivalents: "No charge" → "0.00 USD", "Covered in full" → "0.00 USD", "Fully covered" → "0.00 USD", "Not covered" → "MEMBER_PAYS_100_PERCENT", "Member pays 100%" → "MEMBER_PAYS_100_PERCENT"

Entity pairs where both values normalize to the same canonical form are immediately recorded as MATCH without an MLLM call. This fast path handles a significant portion of matches in practice — format-only differences between SBC and Benefit Grid representations — at zero additional API cost.

**Chain-of-Thought Validation Prompt:**
For entity pairs that are not resolved by pre-normalization, a single section-level CoT validation prompt is constructed containing all remaining entity pairs for that section along with their normalized values, entity descriptions, and section context. The MLLM is guided through a five-step reasoning sequence:

1. **Normalization Review:** Verify and refine the pre-normalized values, catching any cases the rule-based normalizer missed.
2. **Semantic Alignment Check:** Determine whether normalized values express the same underlying fact, explicitly accounting for paraphrase equivalence, unit equivalence, and abbreviation expansion.
3. **Discrepancy Analysis:** For differing values, classify the discrepancy type — NUMERIC_DIFFERENCE, TERMINOLOGY_VARIANT, COVERAGE_RECLASSIFICATION, or FORMAT_DIFFERENCE.
4. **Status Assignment:** Assign MATCH, MISMATCH, PARTIAL_MATCH, or INELIGIBLE for each entity pair.
5. **Confidence Calibration:** Express a per-entity confidence score from 0.0 to 1.0.

The full reasoning chain is preserved in the output for every entity, providing a completely auditable decision trail.

**Per-Entity Validation Output:**
Each entity in the final validation report contains:
- Entity name and section membership
- Raw value from Document A and Document B
- Normalized values after pre-processing
- Validation status: MATCH, MISMATCH, PARTIAL_MATCH, or INELIGIBLE
- Discrepancy type if status is MISMATCH or PARTIAL_MATCH
- Full Chain-of-Thought reasoning text
- Confidence score
- Human review flag
- Source evidence: page number, region description, raw surrounding context
- Expression audit trail for EXPRESSION entities: template, variable values, computed result
![E2E Design](img/validation%20engine.jpeg)
---

## 4. Configuration File Design

The YAML configuration file is the central control artifact of the CMSVS system. Its structure directly drives the behavior of every pipeline layer — from retrieval query construction to MLLM prompt content to validation context.

### 4.1 Configuration Schema

**Top-Level Metadata:**
Configuration name, version identifier, and domain label used for logging and report headers.

**Sections Array:**
Each section defines a logical group of entities corresponding to a functional section of the document type being validated. Each section contains:
- `section_name` — Canonical identifier used in retrieval queries and report organization
- `section_description` — Natural language description providing semantic context for both retrieval and validation
- `section_keywords` — Domain-specific terms that anchor the dense retrieval query for this section
- `entities` — Array of entity definitions belonging to this section

**Entity Definition:**
Each entity contains:
- `entity_name` — Canonical identifier that must match the M2 ground truth JSON entity keys exactly
- `entity_description` — Natural language description used in MLLM extraction and validation prompts
- `entity_extraction_logic` — Either DIRECT (extract as-is) or EXPRESSION (compute from variables)
- `entity_example_value` — Representative format example for MLLM output calibration
- `data_type` — One of: monetary, percentage, coverage_classification, text
- `expression_template` — (EXPRESSION only) Mathematical formula referencing variable names
- `expression_variables` — (EXPRESSION only) Mapping of variable name to description and example

**Validation Settings:**
Global confidence threshold, list of high-stakes entities requiring cross-verification, and human review escalation configuration.

### 4.2 Healthcare SBC Configuration Structure

The primary test configuration covers 18 entities across 5 sections, directly corresponding to the M2 ground truth JSON annotation files:

| Section | Entities | EXPRESSION Entities | Section Keywords |
|---|---|---|---|
| Deductibles | 4 | 1 (combined family) | deductible, before your plan pays, annual |
| Out-of-Pocket Maximums | 3 | 1 (combined OOP) | out-of-pocket, maximum, limit, stop-loss |
| Copayments and Coinsurance | 5 | 0 | copay, coinsurance, specialist, emergency, primary care |
| Prescription Drug Costs | 4 | 1 (monthly effective cost) | prescription, drug, formulary, generic, brand |
| Coverage Classifications | 2 | 0 | covered, not covered, prior authorization |

---

## 5. Model and Tool Selection Summary

### 5.1 Complete Component Stack

| Component | Selected Tool | Provider | Cost | Primary Justification |
|---|---|---|---|---|
| LLM Inference (Primary) | llama-3.3-70b-versatile | Groq API (Free) | $0 | Highest free-tier throughput via LPU, strong JSON reliability |
| LLM Inference (Extended Context) | mixtral-8x7b-32768 | Groq API (Free) | $0 | 32K context window for dense multi-column pages |
| LLM Inference (Fallback) | meta-llama/Llama-4-Scout-17B-16E | NVIDIA NIM (Free) | $0 | Independent infrastructure fallback |
| Dense Embeddings | llama-3.2-nemoretriever-300m-embed-v1 | NVIDIA NIM (Free) | $0 | Retrieval-optimized training, 300M parameters, free tier |
| Vector Store | ChromaDB (in-memory) | Local | $0 | No server, temporary per-document, cosine similarity |
| OCR | PaddleOCR Mobile | Local | $0 | Fast CPU inference, sufficient for semantic indexing |
| PDF Processing | PyMuPDF (fitz) | Local | $0 | Fast, dependency-free, handles malformed PDFs |
| Expression Evaluation | SimpleEval | Local | $0 | Safe sandboxed evaluation, prevents code injection |
| Configuration Parsing | PyYAML + jsonschema | Local | $0 | Standard YAML parsing with schema validation |

**Total Inference Cost: $0 across all components**

---

## 6. End-to-End Pipeline Flow

### 6.1 PDF Document Processing Flow

**Stage 1 — Document Loading**
PyMuPDF opens the PDF and renders each page as an RGB image at 150 DPI. Each page image is base64-encoded and stored in a PageImageStore keyed by page number. The complete image store is held in memory for the duration of processing.

**Stage 2 — OCR Processing**
PaddleOCR Mobile processes each page image and returns detected text lines. A StructuredPage object is constructed per page containing raw text, detected section headers, detected key-value pairs, and a composite index text structured as: SECTIONS: [headers] | KEY_VALUES: [pairs] | [full raw text]. The structured format prioritizes high-signal content at the beginning of the index text.

**Stage 3 — Dense Vector Index Construction**
Each page's composite index text is embedded via the NVIDIA NIM NemoRetriever API. The resulting vectors are stored in an in-memory ChromaDB collection with page number metadata. The collection persists only for the duration of the document's processing and is explicitly destroyed afterward to free memory.

**Stage 4 — Section-Wise Dense Retrieval**
For each section in the configuration, a retrieval query is built from section name, keywords, entity names, descriptions, and example values. The query is embedded using the same NemoRetriever model, ensuring query and document vectors share the same semantic space. ChromaDB performs cosine similarity search and returns the top-2 page numbers.

**Stage 5 — Section-Wise MLLM Extraction**
The identified page images are fetched from the PageImageStore. A unified extraction prompt is built containing all entity definitions for the section — both DIRECT entities and EXPRESSION variable pseudo-entities — labeled clearly to distinguish their roles. The MLLM processes the page images visually and returns a structured JSON extraction result for all entities in a single call.

**Stage 6 — Expression Computation**
For EXPRESSION entities, extracted variable values are parsed to numeric, evaluated by SimpleEval against the configured template, and formatted into final FinalEntityValue objects with complete audit trails.

**Stage 7 — Low-Confidence Fallback**
Entities with extraction confidence below 0.75 trigger expanded retrieval to top-4 pages and extraction retry. Entities remaining below threshold after fallback are flagged for human review.

### 6.2 Image Document Processing Flow

**Stage 1 — Image Loading**
PIL opens the image file, converts it to RGB, and base64-encodes it. A single-element PageImageStore is created with page_number=1. No OCR, no indexing, and no retrieval is performed.

**Stage 2 — Direct MLLM Extraction**
All configuration sections are processed against the single image. For each section, the image is passed to the MLLM alongside all section entity definitions. The MLLM extracts all entities visually in one call per section.

**Stage 3 — Expression Computation**
Identical to the PDF path. SimpleEval handles EXPRESSION entities regardless of input type.

### 6.3 Section-Wise Validation Flow

After extraction is complete for both documents, validation proceeds section by section:

1. Collect all entity value pairs for the current section from Doc A and Doc B extractions
2. Apply rule-based pre-normalization to all value pairs in the section
3. Identify pairs where both normalized values are identical — record MATCH without MLLM call
4. Build a single section-level CoT validation prompt for all remaining non-exact pairs, including section context, entity descriptions, raw values, and normalized values
5. Submit the section validation prompt to Groq (primary) or NVIDIA NIM (fallback)
6. Parse the structured JSON response containing per-entity status, discrepancy classification, reasoning chain, and confidence score
7. Flag entities with validation confidence below threshold for human review
8. Append section results to the cumulative ValidationReport

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
│   │   ├── groq_client.py
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

## 8. Architecture Justification

### 8.1 Why MLLM Over Traditional OCR + NLP Pipeline

Traditional document intelligence pipelines convert PDFs to text via OCR and apply NLP models to the extracted text. This approach has well-documented failure modes directly observed during M2 dataset preparation:

**Column Interleaving:** Multi-column SBC PDFs processed by standard text extraction libraries produce interleaved content where adjacent column text is concatenated in reading order, completely garbling table row logic and cell-to-header mappings.

**Spatial Context Loss:** OCR-to-text conversion discards spatial relationships between document elements. Whether a value appears in the "In-Network" or "Out-of-Network" column — a critical semantic distinction for healthcare validation — is lost when the document is flattened to a text stream.

**Table Structure Degradation:** Complex nested tables in SBC documents where a single cell contains multiple network tier values are not reconstructable from flat OCR text output.

The MLLM approach processes document pages as visual artifacts, preserving all spatial, structural, and visual context. The model uses column positions, table borders, font hierarchies, and spatial proximity as semantic signals, exactly as a human reader would.

### 8.2 Why Dense-Only Retrieval

The decision to use purely dense vector retrieval — without a keyword component — is justified by the nature of the retrieval task in this system.

The primary challenge is semantic bridging: entity descriptions written in the configuration file use natural language that may share little or no vocabulary with the actual document text. A configuration entity described as "the annual amount a member must pay before insurance coverage begins" must route to a page containing "deductible" — a domain term absent from the description entirely. Dense retrieval handles this naturally because embedding models encode meaning rather than surface tokens.

Furthermore, the downstream MLLM performs visual extraction directly from page images. This means the retrieval layer only needs to get the right page into scope — it does not need to extract precise values itself. Dense retrieval's semantic matching is entirely sufficient for this page-routing role. Any edge cases where pure dense retrieval might miss a relevant page are caught by the fallback mechanism that expands from top-2 to top-4 pages on low extraction confidence.

The choice of llama-3.2-nemoretriever-300m-embed-v1 specifically amplifies this advantage: unlike general-purpose embedding models, it is trained with a retrieval objective that directly optimizes for the query-document relevance task this pipeline requires.

### 8.3 Why Section-Wise Processing Throughout

The section-wise design principle applies to both extraction and validation stages and delivers consistent benefits across both:

**For Extraction:** Grouping related entities means the MLLM receives a coherent set of entity targets that are likely co-located in the same document section. This improves extraction accuracy because the model has relational context — knowing it is extracting deductible amounts gives it semantic grounding for interpreting ambiguous values.

**For Validation:** Comparing all deductible entities as a group gives the MLLM the relational context needed for accurate semantic judgments. An individual deductible of $1,500 paired with a family deductible of $3,000 is a standard 1:2 ratio that a section-aware model recognizes as internally consistent, providing confidence calibration context unavailable in isolated entity comparison.

**For Cost:** Section batching reduces both extraction API calls and validation API calls by a factor of N_entities / N_sections — approximately 72% in the 18-entity, 5-section healthcare configuration.

### 8.4 Why SimpleEval for Expression Entities

Python's built-in `eval()` is functionally equivalent to SimpleEval for arithmetic expressions but presents an unacceptable enterprise security risk. In a system designed for regulated industries where configuration files are authored by domain experts without security review, a malformed expression template could execute arbitrary system commands. SimpleEval's strict operator and function whitelist eliminates this attack surface entirely while preserving full arithmetic capability for all legitimate expression patterns encountered in insurance document validation.

### 8.5 Why Groq as Primary LLM Provider

The selection of Groq over alternatives is driven by two factors that directly impact project feasibility:

**Speed:** Groq's LPU architecture delivers 500–800 tokens per second, making section-batched extraction and validation calls return in seconds rather than tens of seconds. For an iterative development and evaluation workflow, this latency advantage compounds significantly across hundreds of test runs.

**Free Tier Capacity:** Groq's free tier provides sufficient daily token volume to run the complete 20-document-pair evaluation dataset — approximately 200 extraction calls and 100 validation calls — without hitting quota limits. This enables uninterrupted M4 experimentation without cost management overhead.

---

## 9. Cost Analysis

### 9.1 Per-Document-Pair Cost Estimate

| Stage | Tool | API Calls per Pair | Estimated Cost |
|---|---|---|---|
| OCR (PDF path) | PaddleOCR Mobile — local | 0 | $0.00 |
| Dense Embeddings | NVIDIA NIM free tier | 40–80 per document | $0.00 |
| Dense Retrieval | ChromaDB — local | 0 | $0.00 |
| MLLM Extraction | Groq free tier | 5–10 per document | $0.00 |
| MLLM Validation | Groq free tier | 5 per pair | $0.00 |
| Expression Evaluation | SimpleEval — local | 0 | $0.00 |
| **Total per pair** | | | **$0.00** |

### 9.2 Full Evaluation Dataset Cost

| Dataset | Document Pairs | Estimated Cost |
|---|---|---|
| Healthcare SBC–Benefit Grid | 20 | $0.00 |
| FUNSD Augmented Samples | Multiple | $0.00 |
| **Total M4 and M5 Evaluation** | | **$0.00** |

### 9.3 Competitive Cost Comparison

| Solution | Cost per 1,000 Pages | Training Data Required | Deployment Time |
|---|---|---|---|
| Microsoft Azure Document Intelligence | ~$180 | 5+ labeled samples per template | 2–4 weeks |
| AWS Textract + Custom Model | ~$150–$200 | Labeled dataset | 2–4 weeks |
| Fine-tuned Open Source NER | ~$100–$300 | Large annotated corpus | 4–12 weeks |
| Manual Human Validation | $500–$2,000+ | N/A | N/A |
| **CMSVS — Free-Tier Stack** | **~$0** | **Zero** | **Under 2 hours** |

---

## 10. Validation Scenarios Coverage

The architecture handles all six validation scenario types established in M2:

| Scenario | M2 Augmentation Category | Architectural Handler |
|---|---|---|
| Exact Match | Unaugmented pairs | Rule-based normalizer fast path — no MLLM call |
| Semantic Equivalence | Category 1 — Paraphrase changes | CoT MLLM with coverage equivalents mapping |
| Numeric Normalization | Category 2 — Format changes | Rule-based normalizer (monetary and percentage standardization) |
| OCR Noise Handling | FUNSD augmentation | Image direct path bypasses OCR for extraction entirely |
| Conflict Detection | Category 3 — Value conflicts | CoT MLLM discrepancy analysis — NUMERIC_DIFFERENCE classification |
| Coverage Change Detection | Category 4 — Classification changes | CoT MLLM categorical reasoning — COVERAGE_RECLASSIFICATION classification |

---

## 11. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Groq free tier rate limit during batch evaluation | Medium | Medium | Exponential backoff retry; inter-call delay between document pairs |
| NVIDIA NIM embedding quota exhaustion | Low | Medium | Embeddings computed once per document and cached for all section queries |
| Dense retrieval routes to wrong page for ambiguous content | Medium | High | Fallback expands to top-4 pages on low confidence; human review queue as final safety net |
| PaddleOCR Mobile produces poor index text for noisy scans | Medium | Low | MLLM reads raw images for extraction — OCR quality only affects retrieval index, not extracted values |
| MLLM JSON schema violations in extraction response | High | High | Retry up to three times with explicit schema re-instruction; fallback null response on persistent failure |
| SimpleEval expression template errors from config authoring | Low | Medium | Config validator checks expression syntax at parse time before any API calls are made |
| Entity spanning multiple pages not captured by top-2 retrieval | Low | High | Fallback expansion to top-4 pages captures most spanning cases; human review catches remainder |
| NVIDIA NIM embedding API latency causing slow indexing | Low | Low | Embeddings are batched per document; latency is a one-time cost before the extraction phase begins |

---

## 12. Connection to Subsequent Milestones

### Milestone 4 — Model Training and Experiments

The M3 architecture exposes the following experimentally configurable parameters for M4 systematic evaluation:

- **LLM model selection:** llama-4-Scout vs mixtral-8x7b — extraction accuracy and JSON reliability comparison
- **LLM provider:** Groq primary vs NVIDIA NIM primary — latency and accuracy tradeoff measurement
- **Retrieval top-K:** top-2 vs top-3 vs top-4 — recall improvement vs token cost tradeoff analysis
- **Confidence threshold:** 0.70 vs 0.75 vs 0.80 — precision vs recall tradeoff for human escalation
- **Extraction prompt variants:** template A vs template B — systematic prompt quality comparison
- **Validation granularity:** section-wise vs entity-wise — accuracy and cost measurement

All parameters are controlled via pipeline initialization arguments requiring no code changes, making M4 a configuration-driven experimentation exercise.

### Milestone 5 — Model Evaluation and Analysis

The M3 output JSON schema is designed for direct comparison against M2 ground truth files. Entity names, status fields, and augmentation metadata are aligned, enabling automated computation of:
- Precision, recall, and F1-score per entity
- Per-scenario-type accuracy — exact match, semantic equivalence, numeric normalization, conflict detection, coverage reclassification
- Per-section validation accuracy
- Human review escalation rate as a proxy for system confidence calibration quality
- Retrieval accuracy — percentage of cases where the correct page was in the top-2 retrieved results

### Milestone 6 — Deployment and Documentation

The CLI entry point at `scripts/run_pipeline.py` accepts standard arguments for doc_a_path, doc_b_path, config_path, and output_path. This interface wraps directly as a REST API endpoint or Gradio demo interface for the M6 deployment milestone without architectural modification.

---

## 13. Team Contributions

| Member | Milestone 3 Contributions |
|---|---|
| **Karthik Ganesh** (21f2000775) | Dense retrieval architecture design and implementation (IndexBuilder with ChromaDB, DenseRetriever with NemoRetriever embeddings); MLLM client abstraction for Groq and NVIDIA NIM APIs; API provider evaluation and selection rationale; retrieval quality testing and demo notebook |
| **Mayank Dode** (22f1000781) | Input routing layer (InputHandler for PDF and image detection); OCR engine (PaddleOCR Mobile integration); PageImageStore design; document processor for DPI-controlled PDF rendering; input handling unit tests |
| **Ayush Verma** (21f3000500) | NER prompt engineering for DIRECT and EXPRESSION variable extraction; MLLM extractor with section-batched extraction; expression evaluator with SimpleEval integration; expression orchestrator wiring MLLM to SimpleEval; extraction unit tests |
| **Mallesh Mayara** (21f2001118) | Configuration architecture (CMSVSConfigParser, EntityConfig dataclasses, SectionConfig); healthcare SBC and FUNSD YAML configuration files; shared type contracts (shared_types.py); report generator; JSON schema definitions; milestone documentation, Section-wise semantic validation architecture (SemanticValidator, CoT validation prompt builder, ValueNormalizer rule engine); pipeline integration (PDFPipeline, ImagePipeline, CMSVSPipeline master orchestrator); end-to-end testing; demo notebooks |

---

## 14. Milestone Summary

Milestone 3 has been successfully completed. The following deliverables were produced:

**Completed Architecture Objectives:**
- ✅ Model architecture selected — complete component stack defined with justification for every selection decision
- ✅ Architecture justified — comparative analysis provided for all major design choices including dense-only retrieval, section-wise processing, MLLM visual extraction, and free-tier provider selection
- ✅ End-to-End setup — complete pipeline from raw document input to structured validation JSON report specified and implemented

**Key Architectural Innovations:**
- ✅ Dense RAG page routing — NVIDIA NemoRetriever embeddings route MLLM calls to relevant pages only, achieving 90% token reduction
- ✅ Zero-cost inference stack — Groq free tier for LLM, NVIDIA NIM free tier for embeddings, PaddleOCR Mobile locally for OCR
- ✅ EXPRESSION entity support — SimpleEval enables mathematically derived entity values with full audit trail and security isolation
- ✅ Section-wise extraction and validation — context-aware processing mirroring document logical structure with 72% API call reduction
- ✅ Dual input support — PDF with full RAG pipeline, Image with direct MLLM visual path
- ✅ Complete hallucination control architecture — evidence grounding, schema enforcement, null-returning protocol, confidence thresholding, human review escalation

**Alignment with M2 Dataset:**
- ✅ All six M2 validation scenario types have corresponding architectural handlers
- ✅ Entity names in configuration files match M2 ground truth JSON keys exactly
- ✅ Output schema compatible with M5 automated evaluation against M2 ground truth

**Prepared for M4 Experimentation:**
- ✅ All configurable parameters identified and exposed via pipeline arguments
- ✅ Experiment design framework documented for M4 systematic evaluation
- ✅ Evaluation metrics and comparison methodology defined

*Document Version: 1.0 | Milestone: M3 — Model Architecture*
*Indian Institute of Technology Madras — Deep Learning / Generative AI Course Project*