"""
streamlit_app/app.py
---------------------
Streamlit frontend for the CMSVS API.

Run with:
    streamlit run streamlit_app/app.py

Expects the FastAPI backend running at:
    http://localhost:8000
"""
from __future__ import annotations

import io, os
import json
import time
from typing import Any

import pandas as pd
import requests
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────

# API_BASE = "http://localhost:8000"
API_BASE = os.environ.get("CMSVS_API_URL", "http://localhost:8000").rstrip("/")
PAGE_TITLE = "CMSVS — Document Validation System"

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# API helpers
# ══════════════════════════════════════════════════════════════════════════════

def api_get(path: str) -> dict | None:
    """GET request to the API. Returns None on failure."""
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def api_post_files(
    path: str,
    files: dict,
    data: dict,
    timeout: int = 300,
) -> dict | None:
    """POST multipart/form-data to the API. Returns None on failure."""
    try:
        r = requests.post(
            f"{API_BASE}{path}",
            files=files,
            data=data,
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        st.error(f"API error {exc.response.status_code}: {detail}")
        return None
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> dict:
    """Render sidebar and return user settings."""
    with st.sidebar:
        st.title("⚙️ Settings")

        # API health check
        health = api_get("/health")
        if health:
            if health.get("nvidia_key_set"):
                st.success("🟢 API Online | NVIDIA Key: Set")
            else:
                st.warning("🟡 API Online | NVIDIA Key: Missing")
        else:
            st.error("🔴 API Offline — start the FastAPI server")

        st.divider()

        # Config selection
        configs = []
        config_data = api_get("/configs")
        if config_data:
            configs = config_data.get("configs", [])

        selected_config = st.selectbox(
            "Configuration",
            options=configs,
            help="Select the entity extraction + validation configuration",
        )

        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.05,
            help="Entities below this threshold are flagged for human review",
        )

        st.divider()

        # Show config details
        if selected_config:
            with st.expander("📋 Config Details", expanded=False):
                detail = api_get(f"/configs/{selected_config}")
                if detail:
                    st.markdown(f"**Domain:** {detail.get('domain', '')}")
                    st.markdown(
                        f"**Sections:** {detail.get('total_sections', 0)} | "
                        f"**Entities:** {detail.get('total_entities', 0)}"
                    )
                    for section in detail.get("sections", []):
                        st.markdown(f"**{section['section_name']}**")
                        for e in section["entities"]:
                            badge = (
                                "🧮" if e["entity_extraction_logic"] == "EXPRESSION"
                                else "🔍"
                            )
                            st.markdown(
                                f"&nbsp;&nbsp;{badge} `{e['entity_name']}`",
                                unsafe_allow_html=True,
                            )

        st.divider()
        st.caption("CMSVS v1.0 | NVIDIA NIM Free Tier")

    return {
        "config_name": selected_config,
        "confidence": confidence,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: Single Document Extraction
# ══════════════════════════════════════════════════════════════════════════════

def render_extraction_tab(settings: dict) -> None:
    """Render the single-document extraction tab."""
    st.header("🔍 Extract Entities")
    st.caption(
        "Upload a single document (PDF or image) to extract all configured entities."
    )

    uploaded = st.file_uploader(
        "Upload Document",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
        help="PDF files use the full RAG pipeline. Images use direct MLLM extraction.",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        doc_label = st.text_input(
            "Document Label",
            value="My Document",
            help="Human-readable name for this document",
        )
    with col2:
        extract_btn = st.button(
            "Extract Entities",
            type="primary",
            disabled=not (uploaded and settings.get("config_name")),
            use_container_width=True,
        )

    if not settings.get("config_name"):
        st.info("Select a configuration in the sidebar to continue.")
        return

    if extract_btn and uploaded:
        with st.spinner("Extracting entities… this may take 30-90 seconds"):
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            data = {
                "config_name": settings["config_name"],
                "confidence_threshold": settings["confidence"],
            }
            result = api_post_files("/extract", files=files, data=data)

        if result:
            _render_extraction_result(result)


def _render_extraction_result(result: dict) -> None:
    """Display extraction results."""
    st.success(
        f"✅ Extraction complete in {result.get('processing_time_s', 0):.1f}s "
        f"| Job: {result.get('job_id', '')}"
    )

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Entities", result.get("total_entities", 0))
    col2.metric("Found", result.get("found_count", 0))
    col3.metric(
        "Not Found",
        result.get("total_entities", 0) - result.get("found_count", 0),
    )
    col4.metric(
        "Review Required",
        result.get("review_count", 0),
        delta=None,
        delta_color="inverse",
    )

    st.divider()

    # Entity table
    entities = result.get("entities", {})
    if not entities:
        st.warning("No entities extracted.")
        return

    rows = []
    for name, ent in entities.items():
        rows.append({
            "Entity": name,
            "Value": ent.get("extracted_value") or "—",
            "Status": ent.get("extraction_status", ""),
            "Type": ent.get("entity_type", ""),
            "Confidence": f"{ent.get('confidence', 0.0):.0%}",
            "Source Page": ent.get("source_page"),
            "Review": "⚠️" if ent.get("review_required") else "✅",
        })

    df = pd.DataFrame(rows)

    def _highlight(row):
        if row["Review"] == "⚠️":
            return ["background-color: #fff3cd"] * len(row)
        if row["Status"] == "NOT_FOUND":
            return ["background-color: #fce8e8"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df.style.apply(_highlight, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # Detailed view
    with st.expander("🔎 Entity Details", expanded=False):
        for name, ent in entities.items():
            with st.container():
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"**{name}**")
                    st.markdown(f"Status: `{ent.get('extraction_status')}`")
                    st.markdown(f"Confidence: `{ent.get('confidence', 0):.0%}`")
                with c2:
                    st.markdown(f"**Value:** {ent.get('extracted_value') or '—'}")
                    if ent.get("source_region"):
                        st.markdown(f"**Region:** {ent.get('source_region')}")
                    if ent.get("expression_audit"):
                        st.json(ent["expression_audit"])
                st.divider()

    # JSON download
    st.download_button(
        "⬇️ Download JSON",
        data=json.dumps(result, indent=2),
        file_name=f"extraction_{result.get('job_id', 'result')}.json",
        mime="application/json",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: Document Pair Validation
# ══════════════════════════════════════════════════════════════════════════════

def render_validation_tab(settings: dict) -> None:
    """Render the document pair validation tab."""
    st.header("⚖️ Validate Document Pair")
    st.caption(
        "Upload two documents to extract entities and run semantic validation."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Document A")
        doc_a_file = st.file_uploader(
            "Upload Document A",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
            key="doc_a_upload",
        )
        doc_a_name = st.text_input(
            "Label for Document A",
            value="Document A (Source)",
            key="doc_a_name",
        )

    with col_b:
        st.subheader("Document B")
        doc_b_file = st.file_uploader(
            "Upload Document B",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
            key="doc_b_upload",
        )
        doc_b_name = st.text_input(
            "Label for Document B",
            value="Document B (Comparison)",
            key="doc_b_name",
        )

    if not settings.get("config_name"):
        st.info("Select a configuration in the sidebar to continue.")
        return

    output_format = st.radio(
        "Output Format",
        options=["Full Validation Report", "M2 Ground Truth Format"],
        horizontal=True,
    )

    validate_btn = st.button(
        "Run Validation",
        type="primary",
        disabled=not (doc_a_file and doc_b_file and settings.get("config_name")),
        use_container_width=False,
    )

    if validate_btn and doc_a_file and doc_b_file:
        endpoint = (
            "/validate/gt"
            if output_format == "M2 Ground Truth Format"
            else "/validate"
        )

        with st.spinner(
            "Validating document pair… this may take 60-180 seconds"
        ):
            files = {
                "doc_a": (doc_a_file.name, doc_a_file.getvalue(), doc_a_file.type),
                "doc_b": (doc_b_file.name, doc_b_file.getvalue(), doc_b_file.type),
            }
            data = {
                "config_name": settings["config_name"],
                "doc_a_name": doc_a_name,
                "doc_b_name": doc_b_name,
                "confidence_threshold": settings["confidence"],
            }
            result = api_post_files(endpoint, files=files, data=data, timeout=600)

        if result:
            if output_format == "M2 Ground Truth Format":
                _render_groundtruth_result(result)
            else:
                _render_validation_result(result)


def _render_validation_result(result: dict) -> None:
    """Display full validation report."""
    summary = result.get("summary", {})

    st.success(
        f"✅ Validation complete in {result.get('processing_time_s', 0):.1f}s "
        f"| Job: {result.get('job_id', '')}"
    )

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    total = summary.get("total_entities", 0)
    matches = summary.get("total_matches", 0)
    mismatches = summary.get("total_mismatches", 0)
    match_rate = summary.get("match_rate", 0.0)

    col1.metric("Total Entities", total)
    col2.metric("✅ Matches", matches)
    col3.metric("❌ Mismatches", mismatches)
    col4.metric("Match Rate", f"{match_rate:.0%}")
    col5.metric("Review Required", summary.get("review_required", 0))

    # Match rate gauge
    st.progress(match_rate, text=f"Overall Match Rate: {match_rate:.1%}")

    st.divider()

    # Per-section results
    sections = result.get("sections", [])
    for section in sections:
        sec_name = section.get("section_name", "")
        sec_match = section.get("match_count", 0)
        sec_mismatch = section.get("mismatch_count", 0)
        sec_total = len(section.get("entities", []))

        with st.expander(
            f"📂 {sec_name}  "
            f"({sec_match}/{sec_total} match)",
            expanded=sec_mismatch > 0,
        ):
            rows = []
            for ent in section.get("entities", []):
                status = ent.get("validation_status", "")
                icon = {
                    "MATCH": "✅",
                    "MISMATCH": "❌",
                    "PARTIAL_MATCH": "⚠️",
                    "INELIGIBLE": "➖",
                }.get(status, "?")

                rows.append({
                    "Entity": ent.get("entity_name", ""),
                    "Doc A Value": ent.get("doc_a_value") or "—",
                    "Doc B Value": ent.get("doc_b_value") or "—",
                    "Status": f"{icon} {status}",
                    "Discrepancy": ent.get("discrepancy_type", ""),
                    "Confidence": f"{ent.get('confidence', 0):.0%}",
                    "Fast Path": "⚡" if ent.get("fast_path_match") else "",
                })

            df = pd.DataFrame(rows)

            def _style(row):
                s = row["Status"]
                if "MISMATCH" in s:
                    return ["background-color: #fce8e8"] * len(row)
                if "MATCH" in s and "PARTIAL" not in s:
                    return ["background-color: #e8f5e9"] * len(row)
                return [""] * len(row)

            st.dataframe(
                df.style.apply(_style, axis=1),
                use_container_width=True,
                hide_index=True,
            )

            # Reasoning details
            for ent in section.get("entities", []):
                if ent.get("validation_status") in ("MISMATCH", "PARTIAL_MATCH"):
                    st.warning(
                        f"**{ent['entity_name']}**: {ent.get('reasoning', '')}"
                    )

    st.divider()

    # Extraction details
    with st.expander("📊 Raw Extraction Results", expanded=False):
        col_a, col_b = st.columns(2)

        def _entity_df(entities_dict: dict) -> pd.DataFrame:
            return pd.DataFrame([
                {
                    "Entity": name,
                    "Value": e.get("extracted_value") or "—",
                    "Status": e.get("extraction_status", ""),
                    "Confidence": f"{e.get('confidence', 0):.0%}",
                    "Page": e.get("source_page"),
                }
                for name, e in entities_dict.items()
            ])

        with col_a:
            st.markdown(f"**{result.get('doc_a_name', 'Doc A')}**")
            st.dataframe(
                _entity_df(result.get("doc_a_entities", {})),
                hide_index=True,
                use_container_width=True,
            )
        with col_b:
            st.markdown(f"**{result.get('doc_b_name', 'Doc B')}**")
            st.dataframe(
                _entity_df(result.get("doc_b_entities", {})),
                hide_index=True,
                use_container_width=True,
            )

    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download Full Report (JSON)",
            data=json.dumps(result, indent=2),
            file_name=f"validation_{result.get('job_id', 'result')}.json",
            mime="application/json",
        )
    with col2:
        # Build GT format from the full result for convenience
        gt_entities = []
        for section in result.get("sections", []):
            for ent in section.get("entities", []):
                gt_entities.append({
                    "entity_name": ent["entity_name"],
                    "doc_a_value": ent.get("doc_a_value"),
                    "doc_b_value": ent.get("doc_b_value"),
                    "normalized_value": ent.get("doc_a_normalized"),
                    "validation_type": "exact_match" if ent.get("fast_path_match") else "semantic_match",
                    "validation_result": ent.get("validation_status", "").lower(),
                })
        gt_payload = {"entities": gt_entities}
        st.download_button(
            "⬇️ Download GT Format (JSON)",
            data=json.dumps(gt_payload, indent=2),
            file_name=f"gt_{result.get('job_id', 'result')}.json",
            mime="application/json",
        )


def _render_groundtruth_result(result: dict) -> None:
    """Display M2 ground truth format result."""
    st.success(
        f"✅ Validation complete in {result.get('processing_time_s', 0):.1f}s "
        f"| Job: {result.get('job_id', '')}"
    )

    entities = result.get("entities", [])
    matches = sum(1 for e in entities if e.get("validation_result") == "match")
    mismatches = len(entities) - matches

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Entities", len(entities))
    col2.metric("Matches", matches)
    col3.metric("Mismatches", mismatches)

    st.divider()

    rows = []
    for ent in entities:
        icon = "✅" if ent.get("validation_result") == "match" else "❌"
        rows.append({
            "Entity": ent.get("entity_name", ""),
            "Doc A Value": ent.get("doc_a_value") or "—",
            "Doc B Value": ent.get("doc_b_value") or "—",
            "Normalized": ent.get("normalized_value") or "—",
            "Type": ent.get("validation_type", ""),
            "Result": f"{icon} {ent.get('validation_result', '')}",
        })

    df = pd.DataFrame(rows)

    def _style(row):
        if "❌" in row["Result"]:
            return ["background-color: #fce8e8"] * len(row)
        return ["background-color: #e8f5e9"] * len(row)

    st.dataframe(
        df.style.apply(_style, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download Ground Truth JSON",
        data=json.dumps({"entities": entities}, indent=2),
        file_name=f"gt_{result.get('job_id', 'result')}.json",
        mime="application/json",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3: API Explorer
# ══════════════════════════════════════════════════════════════════════════════

def render_api_tab() -> None:
    """Render API explorer tab."""
    st.header("🔌 API Explorer")
    st.caption(
        f"FastAPI backend running at `{API_BASE}` | "
        f"[OpenAPI Docs]({API_BASE}/docs) | [ReDoc]({API_BASE}/redoc)"
    )

    # Health check
    st.subheader("Health Check")
    if st.button("GET /health"):
        health = api_get("/health")
        if health:
            st.json(health)

    st.divider()

    # Config list
    st.subheader("List Configs")
    if st.button("GET /configs"):
        configs = api_get("/configs")
        if configs:
            st.json(configs)

    st.divider()

    # Config detail
    st.subheader("Config Detail")
    config_name_input = st.text_input(
        "Config name",
        value="funsd_ner_config",
        key="api_explorer_config",
    )
    if st.button("GET /configs/{name}"):
        detail = api_get(f"/configs/{config_name_input}")
        if detail:
            st.json(detail)

    st.divider()

    # Endpoint reference
    st.subheader("📚 Endpoint Reference")
    endpoints = [
        ("GET",  "/health",       "API health + available configs"),
        ("GET",  "/configs",      "List all configuration names"),
        ("GET",  "/configs/{name}", "Get config sections and entities"),
        ("POST", "/extract",      "Extract entities from one document"),
        ("POST", "/validate",     "Validate a document pair (full report)"),
        ("POST", "/validate/gt",  "Validate a document pair (GT format)"),
    ]
    df = pd.DataFrame(endpoints, columns=["Method", "Endpoint", "Description"])
    st.dataframe(df, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main app
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main Streamlit application entry point."""
    st.title("📄 CMSVS — Document Validation System")
    st.caption(
        "Configurable Multimodal Semantic Validation | "
        "Powered by NVIDIA NIM"
    )

    settings = render_sidebar()

    tab1, tab2, tab3 = st.tabs([
        "🔍 Extract Entities",
        "⚖️ Validate Pair",
        "🔌 API Explorer",
    ])

    with tab1:
        render_extraction_tab(settings)

    with tab2:
        render_validation_tab(settings)

    with tab3:
        render_api_tab()


if __name__ == "__main__":
    main()