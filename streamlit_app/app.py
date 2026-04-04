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


def api_post_json(path: str, payload: dict, timeout: int = 30) -> dict | None:
    """POST JSON to the API. Returns None on failure."""
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
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


def api_delete(path: str) -> dict | None:
    """DELETE request to the API. Returns None on failure."""
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=10)
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
        ("POST", "/configs/create", "Create config from field definitions"),
        ("GET",  "/configs/markdowns", "List saved Markdown configs"),
        ("GET",  "/configs/{name}/markdown", "Get Markdown extraction file"),
        ("DELETE", "/configs/{name}", "Delete a config (YAML + Markdown)"),
    ]
    df = pd.DataFrame(endpoints, columns=["Method", "Endpoint", "Description"])
    st.dataframe(df, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4: Config Builder — CSV fields → YAML + Markdown
# ══════════════════════════════════════════════════════════════════════════════

def _empty_field_row() -> dict:
    """Return a blank field row for the data editor."""
    return {
        "field_name": "",
        "field_description": "",
        "section": "General",
        "data_type": "text",
        "example_value": "",
        "extraction_logic": "DIRECT",
        "expression_template": "",
    }


def render_config_builder_tab() -> None:
    """Render the Config Builder tab for creating extraction configs."""
    st.header("🛠️ Config Builder")
    st.caption(
        "Define extraction fields via form or CSV upload. "
        "The system generates a YAML config + Markdown extraction instruction file."
    )

    # ── Sub-tabs: Create / Manage ──────────────────────────────────────────
    create_tab, manage_tab = st.tabs(["➕ Create Config", "📂 Manage Configs"])

    # ══════════════════════════════════════════════════════════════════════════
    # CREATE CONFIG
    # ══════════════════════════════════════════════════════════════════════════
    with create_tab:
        st.subheader("Define Fields")

        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            config_name_input = st.text_input(
                "Config Name",
                value="",
                placeholder="e.g. invoice_extraction",
                help="Unique name (will be converted to snake_case)",
                key="cb_config_name",
            )
        with col_meta2:
            domain_input = st.text_input(
                "Domain",
                value="general",
                placeholder="e.g. healthcare, finance, legal",
                key="cb_domain",
            )

        st.divider()

        # ── Input method selector ──────────────────────────────────────────
        input_method = st.radio(
            "Input Method",
            options=["📝 Manual Form", "📤 CSV Upload"],
            horizontal=True,
            key="cb_input_method",
        )

        fields_data: list[dict] = []

        if input_method == "📤 CSV Upload":
            st.info(
                "Upload a CSV with columns: "
                "`field_name`, `field_description`, `section`, `data_type`, "
                "`example_value`, `extraction_logic`, `expression_template`  \n"
                "Only `field_name` is required."
            )
            csv_file = st.file_uploader(
                "Upload CSV",
                type=["csv"],
                key="cb_csv_upload",
            )
            if csv_file:
                try:
                    df_csv = pd.read_csv(csv_file)
                    if "field_name" not in df_csv.columns:
                        st.error("CSV must have a `field_name` column.")
                    else:
                        # Fill defaults
                        defaults = {
                            "field_description": "",
                            "section": "General",
                            "data_type": "text",
                            "example_value": "",
                            "extraction_logic": "DIRECT",
                            "expression_template": "",
                        }
                        for col, default in defaults.items():
                            if col not in df_csv.columns:
                                df_csv[col] = default
                            else:
                                df_csv[col] = df_csv[col].fillna(default)

                        st.success(f"Loaded {len(df_csv)} fields from CSV")
                        st.dataframe(df_csv, use_container_width=True, hide_index=True)
                        fields_data = df_csv.to_dict("records")
                except Exception as exc:
                    st.error(f"Failed to parse CSV: {exc}")

            # Download template
            template_csv = (
                "field_name,field_description,section,data_type,"
                "example_value,extraction_logic,expression_template\n"
                "invoice_number,Unique invoice identifier,Header,text,INV-001,DIRECT,\n"
                "total_amount,Total invoice amount,Totals,monetary,$1500.00,DIRECT,\n"
                "tax_amount,Tax computed from subtotal,Totals,monetary,$150.00,EXPRESSION,"
                "subtotal * tax_rate\n"
            )
            st.download_button(
                "📥 Download CSV Template",
                data=template_csv,
                file_name="cmsvs_fields_template.csv",
                mime="text/csv",
            )

        else:
            # ── Manual form using st.data_editor ───────────────────────────
            if "cb_fields" not in st.session_state:
                st.session_state.cb_fields = [_empty_field_row()]

            edited_df = st.data_editor(
                pd.DataFrame(st.session_state.cb_fields),
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "field_name": st.column_config.TextColumn(
                        "Field Name *", help="Required. e.g. invoice_number"
                    ),
                    "field_description": st.column_config.TextColumn(
                        "Description", help="What this field represents"
                    ),
                    "section": st.column_config.TextColumn(
                        "Section", help="Logical grouping", default="General"
                    ),
                    "data_type": st.column_config.SelectboxColumn(
                        "Data Type",
                        options=["text", "monetary", "percentage", "date", "number"],
                        default="text",
                    ),
                    "example_value": st.column_config.TextColumn(
                        "Example", help="Example expected value"
                    ),
                    "extraction_logic": st.column_config.SelectboxColumn(
                        "Logic",
                        options=["DIRECT", "EXPRESSION"],
                        default="DIRECT",
                    ),
                    "expression_template": st.column_config.TextColumn(
                        "Expression", help="Only for EXPRESSION logic"
                    ),
                },
                key="cb_data_editor",
            )

            # Sync edits back
            st.session_state.cb_fields = edited_df.to_dict("records")
            # Filter out empty rows
            fields_data = [
                r for r in edited_df.to_dict("records")
                if r.get("field_name", "").strip()
            ]

        st.divider()

        # ── Generate button ────────────────────────────────────────────────
        can_generate = bool(config_name_input and config_name_input.strip() and fields_data)

        if st.button(
            "🚀 Generate Config & Markdown",
            type="primary",
            disabled=not can_generate,
            use_container_width=True,
            key="cb_generate_btn",
        ):
            payload = {
                "config_name": config_name_input.strip(),
                "domain": domain_input.strip() or "general",
                "fields": [
                    {
                        "field_name": f.get("field_name", "").strip(),
                        "field_description": f.get("field_description", ""),
                        "section": f.get("section", "General"),
                        "data_type": f.get("data_type", "text"),
                        "example_value": str(f.get("example_value", "")),
                        "extraction_logic": f.get("extraction_logic", "DIRECT"),
                        "expression_template": f.get("expression_template") or None,
                    }
                    for f in fields_data
                    if f.get("field_name", "").strip()
                ],
            }

            with st.spinner("Generating config files…"):
                result = api_post_json("/configs/create", payload)

            if result:
                st.success(
                    f"✅ Config **{result['config_name']}** created! "
                    f"({result['total_sections']} sections, {result['total_fields']} fields)"
                )

                # Show Markdown preview
                st.subheader("📄 Generated Markdown (LLM Extraction Instructions)")
                st.markdown(result.get("markdown_preview", ""), unsafe_allow_html=False)

                # Download buttons
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        "⬇️ Download Markdown (.md)",
                        data=result.get("markdown_preview", ""),
                        file_name=f"{result['config_name']}.md",
                        mime="text/markdown",
                        key="cb_dl_md",
                    )
                with col_dl2:
                    st.download_button(
                        "⬇️ Download Config Info (JSON)",
                        data=json.dumps(result, indent=2),
                        file_name=f"{result['config_name']}_info.json",
                        mime="application/json",
                        key="cb_dl_json",
                    )

    # ══════════════════════════════════════════════════════════════════════════
    # MANAGE CONFIGS (dropdown, preview, delete)
    # ══════════════════════════════════════════════════════════════════════════
    with manage_tab:
        st.subheader("Saved Markdown Configs")

        # Fetch list of markdowns
        md_list_data = api_get("/configs/markdowns")
        md_names: list[str] = md_list_data.get("markdowns", []) if md_list_data else []

        if not md_names:
            st.info(
                "No saved Markdown configs yet. Use the **Create Config** tab to build one."
            )
        else:
            selected_md = st.selectbox(
                "Select a Markdown Config",
                options=md_names,
                key="cb_manage_select",
            )

            if selected_md:
                col_actions = st.columns([1, 1, 2])

                with col_actions[0]:
                    view_btn = st.button("👁️ View", key="cb_view_btn", use_container_width=True)
                with col_actions[1]:
                    delete_btn = st.button(
                        "🗑️ Delete", key="cb_delete_btn",
                        type="secondary", use_container_width=True,
                    )

                if view_btn:
                    md_detail = api_get(f"/configs/{selected_md}/markdown")
                    if md_detail:
                        st.markdown(f"**Config:** `{md_detail['config_name']}`")
                        st.markdown(
                            f"**YAML exists:** {'✅' if md_detail['yaml_exists'] else '❌'}"
                        )
                        st.divider()

                        # Render the markdown
                        st.markdown(md_detail["markdown_content"], unsafe_allow_html=False)

                        # Download
                        st.download_button(
                            "⬇️ Download Markdown",
                            data=md_detail["markdown_content"],
                            file_name=f"{selected_md}.md",
                            mime="text/markdown",
                            key="cb_manage_dl",
                        )

                if delete_btn:
                    confirm = st.checkbox(
                        f"Confirm deletion of **{selected_md}** (YAML + Markdown)",
                        key="cb_delete_confirm",
                    )
                    if confirm:
                        del_result = api_delete(f"/configs/{selected_md}")
                        if del_result:
                            st.success(del_result.get("message", "Deleted."))
                            st.rerun()


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

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Extract Entities",
        "⚖️ Validate Pair",
        "🛠️ Config Builder",
        "🔌 API Explorer",
    ])

    with tab1:
        render_extraction_tab(settings)

    with tab2:
        render_validation_tab(settings)

    with tab3:
        render_config_builder_tab()

    with tab4:
        render_api_tab()


if __name__ == "__main__":
    main()