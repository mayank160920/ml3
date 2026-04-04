# Configurable Multimodal Semantic Validation System: Doc-vs-Doc

## Milestone 2: Dataset Preparation and Preprocessing

**Cross-Document Validation using Document Intelligence**

*Mallesh Mayara (21f2001118) · Mayank Dode (22f1000781) · Karthik Ganesh (21f2000775) · Ayush Verma (21f3000500)*

---

## Abstract

This document reports the dataset preparation methodology undertaken for Milestone 2 of the Configurable Multimodal Semantic Validation System (CMSVS) project. The primary objective of this milestone was to curate, construct, and preprocess evaluation datasets capable of rigorously testing two core system capabilities: **(1) Config-Driven Custom Named Entity Recognition (NER)** and **(2) Chain-of-Thought Semantic Validation** across heterogeneous document pairs. Two complementary datasets were prepared: the **FUNSD (Form Understanding in Noisy Scanned Documents)** benchmark dataset, adapted for entity extraction experiments, and a purpose-built **Healthcare SBC–Benefit Grid Dataset** comprising 20 document pairs derived from real-world Summary of Benefits and Coverage (SBC) documents paired with structured Benefit Grid counterparts. Controlled augmentation strategies were applied across both datasets to simulate the full spectrum of real-world document inconsistencies—including semantic paraphrase variants, numeric format divergences, substantive value conflicts, and coverage reclassification changes. Ground truth JSON annotation files were constructed for all augmented document pairs, enabling quantitative evaluation of both extraction accuracy and validation decision quality. This milestone establishes the empirical foundation upon which all subsequent model architecture, training, and evaluation milestones will build.

---

## 1. Milestone Objectives

Milestone 2 was defined by the following primary objective:

> *Prepare datasets required to evaluate a system capable of performing document entity extraction, cross-document validation, semantic equivalence detection, numeric normalization, and conflict detection.*

This objective directly supports the two major technical tasks of the CMSVS project:

- **Task 1 — Custom NER:** Datasets must contain documents with identifiable named entities of varied types, laid out in realistic and noisy formats, to test the system's ability to extract arbitrary user-defined entities from multimodal documents without fine-tuning.
- **Task 2 — Semantic Validation:** Datasets must contain paired document instances where the same underlying information is expressed in divergent forms—paraphrases, format variants, genuine conflicts—to test the system's ability to make correct semantic equivalence or mismatch judgments.

To satisfy these requirements, two datasets were selected and prepared:

1. **FUNSD Dataset** — for document entity recognition and extraction benchmarking
2. **Healthcare SBC & Benefit Grid Dataset** — for cross-document semantic validation

Together, these datasets simulate the real-world document validation workflows that CMSVS is designed to automate.

---

## 2. Dataset 1: FUNSD — Form Understanding in Noisy Scanned Documents

### 2.1 Dataset Description

The FUNSD (Form Understanding in Noisy Scanned Documents) dataset [Jaume et al., 2019] is a publicly available benchmark for document understanding in realistic, challenging conditions. It consists of fully annotated scanned form documents with rich layout information, making it an ideal evaluation resource for the NER component of CMSVS.

Key characteristics of the FUNSD dataset include:

- **Scanned form documents** representing real administrative and business forms with authentic noise artifacts including scanner distortion, uneven ink density, and background texture.
- **Layout-aware annotations** that encode not only the textual content of each field but also the bounding box coordinates of every annotated region, preserving the spatial relationships between document elements.
- **Key-value entity pairs** structured as linked annotation groups, where a question/label entity is linked to its corresponding answer/value entity—directly analogous to the entity extraction task in CMSVS.
- **Real-world document noise** including OCR-challenging elements such as handwritten annotations, stamps, low-contrast text, and misaligned columns.

The FUNSD dataset is particularly well-suited to CMSVS evaluation because its noisy, scanned format represents the hardest category of documents our system must handle—those that traditional OCR-based pipelines process with greatest degradation and that Multimodal LLMs handle most distinctively by processing the document as a visual artifact.

### 2.2 Data Preparation Steps

The following preparation steps were performed on selected FUNSD samples:

**Step 1 — Document Selection:** A representative subset of FUNSD documents was selected, prioritizing samples with diverse layouts, multiple logical sections, and entity types spanning names, identifiers, dates, addresses, and numeric values.

**Step 2 — Logical Section Identification:** Each selected document was analyzed to identify its logical organizational structure—headers, body sections, summary sections—providing the structural context necessary for config-driven entity targeting in the extraction prompt.

**Step 3 — Entity Group Definition:** Within each document section, entity groups were defined corresponding to the key-value annotations in the FUNSD ground truth. These entity groups served as the basis for constructing CMSVS configuration entries (entity name + natural language description), mapping the academic benchmark annotation format to the CMSVS operational format.

The entity types extracted from FUNSD documents include:
- **Name** — Person names, organization names, product names
- **Address** — Location and address fields
- **Date** — Test dates, report dates, document dates
- **Numeric Values** — Scores, measurements, counts
- **Identifiers** — Project numbers, codes, reference identifiers

### 2.3 FUNSD Data Augmentation

To move beyond static benchmark evaluation and test the robustness of the CMSVS extraction pipeline under realistic variance conditions, a controlled augmentation protocol was applied to selected FUNSD documents. The objective was to simulate the kinds of document variations that arise in production environments—vendor formatting changes, abbreviation conventions, numeric representation standards, and incidental noise—without altering the underlying semantic content of the entities.

Four augmentation categories were applied:

**Synonym Replacement:** Entity labels and surrounding context text were modified to use semantically equivalent but terminologically distinct expressions. For example, the label *"Interest Rate"* was replaced with *"Applicable Interest"*, and *"Date of Test"* was replaced with *"Evaluation Date"*. These modifications test whether the CMSVS NER engine can locate an entity based on its semantic description rather than its literal label.

**Numeric Format Variations:** Numeric values were reformatted across different representation conventions. For example, percentage values expressed as *"11%"* were rewritten as *"0.11"* (decimal fraction), and comma-separated thousands (*"1,000"*) were rewritten in plain integer form (*"1000"*). These modifications test the system's numeric normalization capability—a prerequisite for accurate numeric entity comparison in the validation layer.

**Layout Modifications:** Structural changes were made to simulate format-level variance between document instances of the same type sourced from different vendors or time periods. Column orderings were altered and section positioning was modified to test whether the MLLM-based extraction correctly uses semantic understanding rather than positional heuristics.

**OCR-Style Noise Injection:** Synthetic noise was introduced to simulate the output quality degradation typical of low-resolution scanning or compressed document archival—including character-level substitutions (e.g., *"0"* vs. *"O"*), missing spaces, and punctuation artifacts.

*Purpose: These augmentations collectively test the robustness of entity extraction models against the full spectrum of document variance conditions encountered in enterprise deployments.*

---

## 3. Dataset 2: Healthcare SBC–Benefit Grid Dataset

### 3.1 Domain Context and Motivation

The healthcare insurance domain represents one of the highest-stakes applications of cross-document validation. Insurance plan members, employers, and regulators routinely need to verify that the summary information presented in a **Benefit Grid** (a structured, tabular representation of a health plan's cost-sharing rules) is consistent with the authoritative **Summary of Benefits and Coverage (SBC)** document—a standardized disclosure form mandated by the Affordable Care Act (ACA) for all health insurance plans offered in the United States.

Despite this validation being a routine compliance and consumer protection activity, it is overwhelmingly performed manually by benefits administrators and compliance officers. The SBC–Benefit Grid pair is therefore an ideal real-world test case for CMSVS because:

1. The two documents contain the same underlying factual information expressed in structurally and terminologically different formats—SBCs use narrative-heavy, legally precise language while Benefit Grids use terse, tabular notation.
2. Real-world discrepancies between SBCs and Benefit Grids do occur due to data entry errors, plan year updates applied inconsistently across document versions, and carrier-specific formatting conventions.
3. The validation entities are semantically complex—cost-sharing values involve monetary amounts, percentage figures, network tier qualifications, and service category scoping that require semantic understanding to compare correctly.

### 3.2 SBC Document Collection

**Source:** SBC documents were collected by systematically searching for publicly available Highmark PPO insurance plan disclosures across multiple plan years and benefit tiers.

**Challenges Encountered:**

- *Duplicate Documents:* A significant portion of SBC files available in public repositories were identical across different retrieval paths, necessitating de-duplication to ensure dataset diversity.
- *Formatting Heterogeneity:* While all SBCs must conform to the CMS-mandated SBC template structure, significant variation exists in how carriers render table cells, handle multi-tier cost-sharing entries, and paginate the document—producing distinct formatting styles that stress-test the extraction pipeline.

**Resolution Strategy:** To address these challenges, the collection process was expanded to search across plan tiers—**Gold**, **Silver**, and **Bronze**—and across multiple plan identifiers within each tier, ensuring structural and content diversity across the collected documents. Multiple plan identifier queries were used to avoid retrieval of duplicate files.

**Final Collection:** 20 unique SBC documents were collected, spanning multiple plan tiers and formatting styles. These documents constitute **Doc A** in each validation pair.

### 3.3 Benefit Grid Creation

To create the corresponding **Doc B** counterpart for each SBC document, a standardized Benefit Grid template was designed and populated. The Benefit Grid structure was developed to represent the same cost-sharing information as the SBC in a structured, tabular format typical of employer benefits portals and insurance carrier grid documents.

The Benefit Grid template includes the following columns:

| Column | Description |
|---|---|
| **Service Name** | The healthcare service or procedure category |
| **In-Network Cost** | Cost-sharing amount or percentage for in-network providers |
| **Out-of-Network Cost** | Cost-sharing amount or percentage for out-of-network providers |
| **Notes** | Qualifying conditions, deductible applicability, prior authorization requirements |

Representative service categories included in each Benefit Grid:
- Preventive / Primary Care Doctor Visits
- Specialist Visits
- Emergency Room Care
- Urgent Care
- Hospital Stay (Inpatient)
- Hospital Stay (Outpatient)
- Prescription Drugs (Generic / Brand / Specialty)
- Mental Health and Substance Use Disorder Services
- Imaging (X-Ray / MRI)
- Laboratory Services

Each of the 20 collected SBC documents was manually transcribed into a corresponding Benefit Grid document, producing 20 SBC–Benefit Grid pairs. The Benefit Grids were rendered as PDFs to match the input format expected by the CMSVS pipeline. These documents constitute **Doc B** in each validation pair.

**Document Pair Relationship:**

```
Doc A: SBC Document     →  Detailed, narrative-legal explanation of plan benefits
Doc B: Benefit Grid     →  Structured, tabular summary of the same plan benefits
```

This structural asymmetry—where the same facts are expressed in formally different document types—is precisely the validation challenge that CMSVS is designed to resolve.

---

## 4. Data Augmentation: Benefit Grid Dataset

### 4.1 Augmentation Strategy

To enable rigorous evaluation of the semantic validation component, controlled modifications were applied to a subset of the 20 Benefit Grid documents (Doc B) to introduce known, trackable discrepancies relative to their paired SBC documents (Doc A). This augmentation strategy creates a labeled evaluation dataset where the ground truth validation status (MATCH, MISMATCH, PARTIAL MATCH) is known precisely for each entity pair, enabling quantitative accuracy measurement.

**Augmentation Distribution:**
- **12 Benefit Grid documents** were kept **unchanged** (representing the true-positive MATCH scenario)
- **8 Benefit Grid documents** were **intentionally modified** with controlled perturbations (representing MISMATCH and PARTIAL MATCH scenarios)

All modifications were applied using PDF editing tools, preserving the visual authenticity of the documents (rather than regenerating them from source text) to ensure realistic rendering artifacts.

### 4.2 Augmentation Type Taxonomy

Four categories of modification were applied across the 8 augmented Benefit Grid documents, designed to cover the complete spectrum of discrepancy types encountered in real-world document validation:

**Category 1 — Semantic Changes (Paraphrase Equivalence Testing)**

Values were replaced with semantically equivalent expressions using different terminology, testing whether the validation engine correctly identifies equivalent meaning across linguistic variants.

| Original Value (Doc A / SBC) | Modified Value (Doc B / Benefit Grid) | Validation Status |
|---|---|---|
| *"No charge"* | *"Covered in full"* | MATCH (Semantic Equivalent) |
| *"Not covered"* | *"Member pays 100%"* | MATCH (Semantic Equivalent) |
| *"Prior authorization required"* | *"Pre-approval needed"* | MATCH (Semantic Equivalent) |

These modifications test whether the CMSVS semantic reasoning engine can correctly determine that expressions like *"No charge"* and *"Covered in full"* represent identical cost-sharing obligations—a judgment that string-matching systems cannot make.

**Category 2 — Numeric Format Changes (Normalization Testing)**

Numeric values were reformatted across different representation conventions without changing the underlying value, testing the system's numeric normalization capability.

| Original Value | Modified Value | Validation Status |
|---|---|---|
| *"\$6,550"* | *"6550"* | MATCH (After Normalization) |
| *"20%"* | *"0.20"* | MATCH (After Normalization) |
| *"\$1,500 Individual"* | *"1500"* | MATCH (After Normalization) |

**Category 3 — Value Conflicts (True Mismatch Detection)**

Monetary amounts or percentage values were substantively altered to introduce genuine numerical discrepancies, testing the system's ability to detect real mismatches that represent plan benefit errors.

| Original Value | Modified Value | Validation Status |
|---|---|---|
| *"\$525 copay"* | *"\$400 copay"* | MISMATCH |
| *"\$500 deductible"* | *"\$750 deductible"* | MISMATCH |
| *"30% coinsurance"* | *"20% coinsurance"* | MISMATCH |

These are the highest-stakes discrepancy type in healthcare—a \$125 difference in a copayment amount represents a real financial error that could mislead a plan member's healthcare decisions.

**Category 4 — Coverage Classification Changes (Coverage Reclassification Testing)**

Cost-sharing obligations were reclassified from one coverage category to another, representing changes in coverage policy that alter the fundamental nature of the benefit.

| Original Value | Modified Value | Validation Status |
|---|---|---|
| *"\$30 copay"* | *"Fully covered"* | MISMATCH (Coverage Change) |
| *"Not covered"* | *"\$50 copay after deductible"* | MISMATCH (Coverage Change) |
| *"20% coinsurance"* | *"No charge after deductible"* | MISMATCH (Coverage Change) |

Coverage reclassifications represent a particularly challenging validation scenario because the two values may not be numerically comparable—they require categorical reasoning about what type of cost-sharing obligation each value represents.

---

## 5. Ground Truth Annotation

### 5.1 Ground Truth JSON Schema

Ground truth JSON annotation files were constructed for all 8 augmented document pairs. These files serve as the evaluation oracle against which CMSVS system outputs will be compared in Milestone 5. Each JSON file encodes the complete expected output for a document pair, enabling automated computation of precision, recall, F1-score, and validation accuracy.

The ground truth JSON schema is structured as follows:

```json
{
  "document_pair_id": "sbc_highmark_gold_003",
  "doc_a": "sbc_highmark_gold_003.pdf",
  "doc_b": "benefit_grid_highmark_gold_003_augmented.pdf",
  "entities": [
    {
      "entity_name": "Individual_Deductible_In_Network",
      "value_doc_a": "$500",
      "value_doc_b": "500",
      "normalized_value": "500.00 USD",
      "validation_type": "numeric_normalization",
      "expected_status": "MATCH",
      "augmentation_applied": false
    },
    {
      "entity_name": "Emergency_Room_Copay",
      "value_doc_a": "$525 copay",
      "value_doc_b": "$400 copay",
      "normalized_value": null,
      "validation_type": "value_conflict",
      "expected_status": "MISMATCH",
      "augmentation_applied": true,
      "perturbation_log": {
        "original_value": "$525 copay",
        "modification_type": "value_conflict",
        "modified_value": "$400 copay",
        "modification_date": "2025-03-04"
      }
    },
    {
      "entity_name": "Preventive_Care_Coverage",
      "value_doc_a": "No charge",
      "value_doc_b": "Covered in full",
      "normalized_value": "0.00 USD / 0% coinsurance",
      "validation_type": "semantic_equivalence",
      "expected_status": "MATCH",
      "augmentation_applied": true,
      "perturbation_log": {
        "original_value": "No charge",
        "modification_type": "semantic_change",
        "modified_value": "Covered in full",
        "modification_date": "2025-03-04"
      }
    }
  ],
  "summary": {
    "total_entities": 18,
    "expected_matches": 14,
    "expected_mismatches": 4,
    "augmentations_applied": 4
  }
}
```

Each ground truth record contains:
- **Entity name** (aligned with CMSVS configuration entity names)
- **Value in Doc A** (raw extracted value from SBC)
- **Value in Doc B** (raw value from Benefit Grid, potentially augmented)
- **Normalized value** (canonical form after normalization, where applicable)
- **Validation type** (the category of comparison required)
- **Expected status** (ground truth validation decision)
- **Perturbation log** (for augmented entities: original value, modification type, modified value, date)

### 5.2 Validation Scenarios Coverage

The assembled dataset provides coverage across all six validation scenario types required by the CMSVS evaluation framework:

| Scenario | Description | Dataset Coverage |
|---|---|---|
| **Exact Match** | Identical values in both documents | ✓ Covered (unaugmented pairs) |
| **Semantic Equivalence** | Paraphrase or synonym variants expressing same fact | ✓ Covered (Category 1 augmentation) |
| **Numeric Normalization** | Same numeric value in different format representations | ✓ Covered (Category 2 augmentation) |
| **OCR Noise Handling** | Character-level noise from scanning artifacts | ✓ Covered (FUNSD augmentation) |
| **Conflict Detection** | Substantively different values for the same entity | ✓ Covered (Category 3 augmentation) |
| **Coverage Change Detection** | Reclassification of coverage type or obligation | ✓ Covered (Category 4 augmentation) |

*These scenarios comprehensively represent the real-world document inconsistency spectrum that production validation workflows must handle.*

---

## 6. Final Dataset Summary

The complete dataset produced by Milestone 2 is summarized in Table 1.

| Dataset Component | Count | Purpose |
|---|---|---|
| FUNSD Samples (Original) | Multiple | NER extraction benchmarking |
| FUNSD Samples (Augmented) | Multiple | NER robustness evaluation |
| SBC Documents (Doc A) | 20 | Cross-document validation — Source A |
| Benefit Grid Documents (Doc B, Unaugmented) | 12 | True MATCH scenario evaluation |
| Benefit Grid Documents (Doc B, Augmented) | 8 | MISMATCH / PARTIAL MATCH evaluation |
| Ground Truth JSON Files | 8 | Evaluation oracle for augmented pairs |
| **Total Document Pairs** | **20** | **Full validation dataset** |

*Table 1: Final dataset composition for Milestone 2.*

This dataset jointly supports:
- **Entity extraction evaluation** (Task 1: Custom NER) using both FUNSD and SBC documents
- **Cross-document validation experiments** (Task 2: Semantic Validation) using the SBC–Benefit Grid pairs with annotated ground truth

---

## 7. Challenges Encountered and Resolutions

The dataset preparation process surfaced several practical challenges that are relevant to the broader system design and are documented here for completeness.

**Challenge 1 — Duplicate SBC Documents**

*Description:* A large proportion of SBC files discoverable via public search were functionally identical documents retrieved under different file names or plan identifiers. Simple filename-based de-duplication was insufficient.

*Resolution:* Manual inspection of document content and plan metadata (plan ID, coverage period, carrier name) was performed to identify and remove duplicates. The search strategy was expanded across Gold, Silver, and Bronze plan tiers and across multiple plan identifier queries to ensure genuine diversity.

**Challenge 2 — Complex Multi-Tier Table Structures in SBCs**

*Description:* Many SBC documents present cost-sharing information in tables with multiple network tiers (e.g., Tier 1 / Tier 2 / Out-of-Network) within a single table cell, creating ambiguous cell-to-entity mapping that complicates both extraction and validation.

*Resolution:* Each network tier was treated as a distinct entity in the Benefit Grid template, with explicit entity names encoding the tier level (e.g., `Emergency_Room_Tier1_Copay`, `Emergency_Room_OON_Copay`). This decomposition simplifies the extraction task and produces unambiguous validation pairs.

**Challenge 3 — PDF Formatting and Column Misalignment**

*Description:* Text extraction from multi-column SBC PDFs using standard PDF parsing libraries produced misaligned column readings where content from adjacent columns was interleaved in the extracted text stream, yielding garbled entity values.

*Resolution:* Rather than relying on text extraction, the CMSVS pipeline processes documents as page images passed directly to the Multimodal LLM—precisely the approach that eliminates this class of problem. For ground truth construction, manual extraction was used to ensure annotation quality.

**Challenge 4 — Ambiguous Service Descriptions**

*Description:* Some SBC table rows bundled multiple services within a single entry (e.g., *"Lab work and X-rays"*), making it unclear how to map the associated cost-sharing value to individual service entities in the Benefit Grid.

*Resolution:* Manual decomposition was performed for ambiguous rows, with each component service receiving its own entity entry. Where decomposition was genuinely ambiguous, the combined entity was preserved as a single entry with a composite name and a note in the ground truth JSON flagging the ambiguity.

---

## 8. Relation to CMSVS System Architecture

The datasets prepared in this milestone are designed to align precisely with the two-task pipeline architecture defined in Milestone 1 and further developed in subsequent milestones.

**Alignment with Task 1 (Custom NER):**
- FUNSD documents provide a rigorous test of the MLLM's ability to extract user-defined entity types from noisy, scanned, layout-varied forms without fine-tuning.
- The augmented FUNSD variants test robustness to synonym-level and format-level variance in entity labels and values.
- The SBC documents test extraction from complex, multi-section, multi-table structured insurance documents with domain-specific terminology.

**Alignment with Task 2 (Semantic Validation):**
- The SBC–Benefit Grid pairs provide a realistic, high-stakes cross-document validation scenario where the same information is expressed in structurally and terminologically different document types.
- The four augmentation categories (semantic change, numeric format, value conflict, coverage change) directly correspond to the four comparison difficulty levels the CMSVS semantic reasoning engine must handle.
- The ground truth JSON files provide an evaluation oracle with per-entity, per-scenario labels that enable granular diagnosis of system performance.

**Configuration File Readiness:**
As a byproduct of the entity definition process during Benefit Grid creation, a complete CMSVS configuration file for the Healthcare SBC–Benefit Grid validation workflow was produced. This configuration defines 18 canonical entities spanning deductible amounts, out-of-pocket maximums, copayment values, coinsurance percentages, and coverage availability flags—with natural language descriptions authored in the CMSVS YAML format. This configuration file will serve as the primary test configuration for all pipeline evaluation in Milestones 4 and 5.

---

## 9. Team Contributions

| Member | Contributions |
|---|---|
| **Karthik Ganesh** (21f2000775) | Researched the validation use case and domain context; identified FUNSD and SBC datasets as appropriate evaluation resources; designed the augmentation scenario taxonomy and defined the four augmentation categories |
| **Mayank Dode** (22f1000781) | Implemented FUNSD data augmentation pipeline; created modified document samples with synonym replacement, numeric format variation, layout modification, and OCR noise injection |
| **Ayush Verma** (21f3000500) | Performed SBC and Benefit Grid dataset augmentation using PDF editing tools; generated all 8 ground truth JSON annotation files with perturbation logs |
| **Mallesh Mayara** (21f2001118) | Milestone documentation, dataset summary compilation, configuration file authoring, and cross-team integration review |

---

## 10. Milestone Summary

Milestone 2 has been successfully completed. The following deliverables were produced:

**Completed Objectives:**
- ✅ Dataset research and selection — two complementary datasets identified and justified
- ✅ Data collection from multiple sources — 20 SBC documents collected across plan tiers
- ✅ Document standardization — Benefit Grid template designed and 20 paired documents created
- ✅ Controlled data augmentation — four augmentation categories applied to 8 document pairs
- ✅ Ground truth generation — 8 annotated JSON files with perturbation logs
- ✅ CMSVS configuration file — healthcare domain entity configuration authored in YAML

**Prepared Dataset Enables:**
- ✅ Entity extraction evaluation against FUNSD benchmark and SBC domain
- ✅ Cross-document semantic validation experiments across six validation scenario types
- ✅ Quantitative benchmarking of both NER accuracy and validation decision quality in Milestone 5

**Next Milestone (M3 — Model Architecture):** The dataset prepared in this milestone will directly inform model and prompt architecture decisions in Milestone 3. The diversity of document layouts, entity types, and validation scenarios establishes the evaluation surface against which architectural choices will be justified.

*Document Version: 1.0 | Milestone: M2 — Dataset Preparation 
*Indian Institute of Technology Madras — Deep Learning / Generative AI Course Project*
