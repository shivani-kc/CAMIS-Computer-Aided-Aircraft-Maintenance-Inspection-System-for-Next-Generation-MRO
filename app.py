from __future__ import annotations

import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import torch
torch.classes.__path__ = []

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path("/teamspace/studios/this_studio/CAMIS_AI")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.dip.pipeline import process_single_image
from src.inference.yolo_infer import run_inference
from src.llm.gemma_client import GemmaClient
from src.storage.csv_store import append_pipeline_results
from src.powerbi.stream_client import PowerBIStreamingClient
from src.utils.dashboard_data import (
    load_history,
    filter_history,
    make_kpis,
    available_filter_values,
    defect_trend_chart,
    severity_stacked_chart,
    defect_type_distribution_chart,
    aircraft_risk_chart,
    preprocessing_chart,
    fly_status_chart,
    confidence_trend_chart,
    recent_inspection_table,
)


def _clean_frontend_table(df):
    import pandas as pd

    if df is None or getattr(df, "empty", True):
        return df

    out = df.copy()

    if "timestamp" in out.columns:
        out["timestamp"] = out["timestamp"].replace(["None", "none", "nan", "NaT", ""], pd.NA)
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        out["timestamp"] = out["timestamp"].fillna("N/A").astype(str)
        out["timestamp"] = out["timestamp"].replace({"NaT": "N/A", "None": "N/A", "nan": "N/A"})

    for col in out.columns:
        out[col] = out[col].astype(str).replace({
            "None": "N/A",
            "nan": "N/A",
            "NaT": "N/A",
            "<NA>": "N/A"
        })

    return out


def _hide_problem_columns(df):
    if df is None:
        return df
    return df.drop(columns=["timestamp"], errors="ignore")

st.set_page_config(
    page_title="CAMIS Fleet Inspection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

INPUT_DIR = PROJECT_ROOT / "data" / "input_images"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_images"
ANNOTATION_DIR = PROJECT_ROOT / "results" / "annotations"
JSON_DIR = PROJECT_ROOT / "results" / "json"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summaries"
MODEL_PATH = PROJECT_ROOT / "models" / "yolo" / "CAMIS_03_aircraft_defect_yolov8l_optimized_weights_best.pt"

for d in [INPUT_DIR, PROCESSED_DIR, ANNOTATION_DIR, JSON_DIR, SUMMARY_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = uploaded_file.name.replace(" ", "_")
    out_path = INPUT_DIR / f"{timestamp}_{safe_name}"
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path


def safe_open_image(path_str: str):
    try:
        path = Path(path_str)
        if path.exists():
            return Image.open(path)
    except Exception:
        return None
    return None


def reset_app():
    for k in list(st.session_state.keys()):
        del st.session_state[k]


st.title("CAMIS Fleet Inspection Dashboard")
st.caption("Upload image -> image processing -> damage annotation -> readable AI summary -> CSV-backed fleet analytics")

history_df_all = load_history()
aircraft_options, severity_options, defect_options = available_filter_values(history_df_all)

with st.sidebar:
    st.header("Inspection control")
    uploaded_file = st.file_uploader(
        "Attach inspection image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=False,
    )
    run_clicked = st.button("Run inspection pipeline", use_container_width=True)
    reset_clicked = st.button("Restart session", use_container_width=True)

    st.divider()
    st.header("Fleet filters")
    selected_aircraft = st.multiselect("Aircraft", aircraft_options)
    selected_severity = st.multiselect("Severity", severity_options)
    selected_defects = st.multiselect("Defect type", defect_options)

if reset_clicked:
    reset_app()
    st.rerun()

if uploaded_file is not None:
    st.markdown("## Uploaded image")
    st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

if run_clicked:
    if uploaded_file is None:
        st.error("Attach an image first.")
        st.stop()

    if not MODEL_PATH.exists():
        st.error(f"YOLO model not found at: {MODEL_PATH}")
        st.stop()

    input_path = save_uploaded_file(uploaded_file)
    processed_path = PROCESSED_DIR / f"{input_path.stem}_dip{input_path.suffix}"
    annotated_path = ANNOTATION_DIR / f"{input_path.stem}_annotated{input_path.suffix}"
    result_json_path = JSON_DIR / f"{input_path.stem}_result.json"
    summary_json_path = SUMMARY_DIR / f"{input_path.stem}_summary.json"

    progress = st.progress(0, text="Starting CAMIS pipeline...")
    progress.progress(20, text="Running digital image processing...")
    dip_result = process_single_image(str(input_path), str(processed_path))

    progress.progress(50, text="Running YOLO damage detection...")
    inference_result = run_inference(
        model_path=str(MODEL_PATH),
        input_path=str(processed_path),
        annotated_output_path=str(annotated_path),
        json_output_path=str(result_json_path),
    )

    progress.progress(75, text="Generating readable Gemma summary...")
    gemma = GemmaClient()
    gemma_result = gemma.analyze_json_file(str(result_json_path), str(summary_json_path))

    progress.progress(90, text="Updating inspection history...")
    csv_result = append_pipeline_results(
        image_name=input_path.name,
        inference_result=inference_result,
        dip_result=dip_result,
        gemma_result=gemma_result,
        zones_affected=[],
    )

    powerbi_result = PowerBIStreamingClient().post_single_row(csv_result["row"])
    st.session_state["powerbi_result"] = powerbi_result

    st.session_state["uploaded_path"] = str(input_path)
    st.session_state["processed_path"] = str(processed_path)
    st.session_state["annotated_path"] = str(annotated_path)
    st.session_state["dip_result"] = dip_result
    st.session_state["gemma_result"] = gemma_result
    st.session_state["csv_result"] = csv_result

    progress.progress(100, text="Inspection completed.")
    st.success("Inspection saved successfully. Fleet dashboard updated.")
    st.rerun()

if "uploaded_path" in st.session_state:
    st.markdown("## Inspection workflow")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original upload")
        img = safe_open_image(st.session_state["uploaded_path"])
        if img is not None:
            st.image(img, use_container_width=True)
        else:
            st.warning("Uploaded image preview unavailable.")

    with col2:
        st.subheader("Annotated result")
        ann = safe_open_image(st.session_state["annotated_path"])
        if ann is not None:
            st.image(ann, use_container_width=True)
        else:
            st.warning("Annotated image was not found or could not be opened.")

    st.markdown("### Digital image processing")
    dip_result = st.session_state["dip_result"]
    d1, d2, d3 = st.columns(3)
    d1.metric("Blur score", f"{dip_result.get('blur_score', 0):.2f}")
    d2.metric("Brightness", f"{dip_result.get('brightness_score', 0):.2f}")
    d3.metric("Contrast", f"{dip_result.get('contrast_score', 0):.2f}")
    st.write("Applied DIP operations:", ", ".join(dip_result.get("applied_operations", [])) or "No preprocessing applied")

    st.markdown("### Gemma summary")
    st.write(st.session_state["gemma_result"].get("summary", "No summary available"))

    row = st.session_state["csv_result"]["row"]
    st.markdown("### Inspection record")
    if "powerbi_result" in st.session_state:
        pbi = st.session_state["powerbi_result"]
        if pbi.get("success"):
            st.success("Power BI streaming updated successfully.")
        else:
            st.warning(f"Power BI streaming not updated: {pbi.get('message', 'Unknown error')}")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Inspection ID", row["inspection_id"])
    r2.metric("Flight ID", row["flight_id"])
    r3.metric("Aircraft ID", row["aircraft_id"])
    r4.metric("Fly status", row["fly_status"])
    r5.metric("Est. Repair Cost", f'${row.get("estimated_cost_usd", 0):,.0f}')

history_df_all = load_history()
history_df = filter_history(
    history_df_all,
    aircraft_ids=selected_aircraft,
    severities=selected_severity,
    defect_types=selected_defects,
)

st.markdown("## Fleet management overview")
kpis = make_kpis(history_df)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Inspections", kpis["total_inspections"])
k2.metric("Defects", kpis["total_defects"])
k3.metric("Critical defects", kpis["critical_defects"])
k4.metric("Avg confidence", kpis["avg_confidence"])
k5.metric("No-fly aircraft", kpis["no_fly_aircraft"])

r1c1, r1c2 = st.columns(2)
with r1c1:
    st.plotly_chart(defect_trend_chart(history_df), use_container_width=True)
with r1c2:
    st.plotly_chart(severity_stacked_chart(history_df), use_container_width=True)

r2c1, r2c2 = st.columns(2)
with r2c1:
    st.plotly_chart(defect_type_distribution_chart(history_df), use_container_width=True)
with r2c2:
    st.plotly_chart(aircraft_risk_chart(history_df), use_container_width=True)

r3c1, r3c2 = st.columns(2)
with r3c1:
    st.plotly_chart(preprocessing_chart(history_df), use_container_width=True)
with r3c2:
    st.plotly_chart(fly_status_chart(history_df), use_container_width=True)

st.markdown("## Model confidence")
st.plotly_chart(confidence_trend_chart(history_df), use_container_width=True)

st.markdown("## Recent inspections")
st.dataframe(_hide_problem_columns(recent_inspection_table(history_df)), use_container_width=True, hide_index=True)
