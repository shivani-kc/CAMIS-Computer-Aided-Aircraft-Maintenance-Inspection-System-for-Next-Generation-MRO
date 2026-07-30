import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import datetime
from ultralytics import YOLO

# Set page configuration to wide layout for an engineering dashboard appearance
st.set_page_config(page_title="AeroGuard: Aircraft Defect Inspection", layout="wide")

# Initialize session state for persistent tracking across image uploads
if "history_df" not in st.session_state:
    st.session_state.history_df = pd.DataFrame(columns=["Timestamp", "Filename", "Defect Type", "Confidence", "DIP Applied"])

# ---------------------------------------------------------
# CONSTANTS & MODEL LOADING
# ---------------------------------------------------------
WEIGHTS_PATH = "/teamspace/studios/this_studio/CAMIS_03_aircraft_defect_yolov8l_optimized_weights_best.pt"

@st.cache_resource
def load_yolo_model():
    if not os.path.exists(WEIGHTS_PATH):
        # Fallback check
        alternative_path = "/teamspace/studios/this_studio/CAMIS_03/aircraft_defect_yolov8l_optimized/weights/best.pt"
        if os.path.exists(alternative_path):
            return YOLO(alternative_path)
        raise FileNotFoundError(f"Weights file missing at {WEIGHTS_PATH}")
    return YOLO(WEIGHTS_PATH)

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"Failed to load model weights: {e}")
    st.stop()

# ---------------------------------------------------------
# HEADER SECTION
# ---------------------------------------------------------
st.title("✈️ AeroGuard AI: Structural Integrity & Defect Inspection Pipeline")
st.markdown("---")

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("Pipeline Configuration")
conf_threshold = st.sidebar.slider("YOLOv8l Confidence Threshold", min_value=0.05, max_value=1.00, value=0.25, step=0.05)
inference_size = st.sidebar.selectbox("Inference Canvas Resolution (imgsz)", options=[640, 1280], index=1)

st.sidebar.markdown("---")
st.sidebar.header("Digital Image Processing (DIP)")
enable_clahe = st.sidebar.checkbox("Enable Adaptive Histogram Equalization (CLAHE)", value=True)
enable_sharpen = st.sidebar.checkbox("Enable Laplacian Unsharp Masking", value=False)

# ---------------------------------------------------------
# MAIN INTERFACE LAYOUT
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload Airframe Inspection Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read file directly into OpenCV numpy matrix
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=uint8)
    raw_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    processed_image = raw_image.copy()
    
    dip_log = "None"
    
    # Execute the conditional DIP Layer
    if enable_clahe:
        dip_log = "CLAHE"
        lab = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        processed_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
    if enable_sharpen:
        dip_log = "CLAHE + Sharpen" if enable_clahe else "Sharpen"
        gaussian_blur = cv2.GaussianBlur(processed_image, (9, 9), 10.0)
        processed_image = cv2.addWeighted(processed_image, 1.5, gaussian_blur, -0.5, 0)

    # Display processing stage inputs sequentially
    st.subheader("Analysis Viewport")
    
    # Run YOLOv8l Inference on CPU using parameters configured in UI
    results = model.predict(source=processed_image, conf=conf_threshold, imgsz=inference_size, device='cpu', verbose=False)
    result = results[0]
    annotated_image = result.plot()
    
    # Convert back to RGB for crisp web browser rendering
    annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, caption=f"Inference Pipeline Output (Resolution: {inference_size}px)", use_column_width=True)
    
    # Parse Detection Metrics and Generate Mockup JSON Payload for Gemma
    detections_list = []
    new_rows = []
    
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        coords = [round(x, 1) for x in box.xyxy[0].tolist()]
        
        detections_list.append({
            "class": label,
            "confidence": round(conf, 3),
            "coordinates": coords
        })
        
        new_rows.append({
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Filename": uploaded_file.name,
            "Defect Type": label,
            "Confidence": conf,
            "DIP Applied": dip_log
        })

    # Update session metrics log table
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        st.session_state.history_df = pd.concat([st.session_state.history_df, new_df], ignore_index=True)

    # UI Information Layout Blocks
    st.markdown("---")
    st.subheader("Pipeline Telemetry & Diagnostics")
    
    # Structural JSON Payload Block
    st.markdown("#### Structured JSON Dispatched to LLM (Gemma)")
    st.json({
        "filename": uploaded_file.name,
        "total_anomalies": len(detections_list),
        "detections": detections_list
    })

    # Gemma Automated Narrative Summary Block
    st.markdown("#### Gemma Automated Technical Narrative")
    if len(detections_list) > 0:
        # Static engineering report simulation mapping to Gemma's typical output format
        summary_text = (
            f"Analysis of airframe file {uploaded_file.name} resolved {len(detections_list)} primary structural "
            f"anomalies using an inspection resolution baseline of {inference_size}px. The predominant defect "
            f"signature classified is '{detections_list[0]['class']}' manifesting a localized confidence envelope "
            f"of {detections_list[0]['confidence']*100:.1f}%. Immediate inspection review is requested regarding target "
            f"boundary limits {detections_list[0]['coordinates']} to avoid operational stress cracks."
        )
        st.info(summary_text)
    else:
        st.success("No structural defect anomalies exceeded the specified detection bounds. Component clears inspection parameters.")

# ---------------------------------------------------------
# HISTORICAL ANALYTICS GRAPHICS SECTION (Mimics PowerBI)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("Analytics Log & Cumulative Metrics Dashboard")

if not st.session_state.history_df.empty:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Historical Defect Distribution**")
        defect_counts = st.session_state.history_df["Defect Type"].value_counts()
        st.bar_chart(defect_counts)
        
        # Calculate summary metrics cards
        avg_conf = st.session_state.history_df["Confidence"].mean()
        st.metric(label="Average Inspection Confidence", value=f"{avg_conf:.1%}")
        
    with col2:
        st.markdown("**Unified System Telemetry Log**")
        st.dataframe(st.session_state.history_df, use_container_width=True)
else:
    st.write("System analytics logger is idle. Upload airframe target image frames to populate baseline matrix charts.")
