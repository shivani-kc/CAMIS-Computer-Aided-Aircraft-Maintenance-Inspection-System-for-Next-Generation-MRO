# 🚀 Milestone 1: Problem Definition, Research Gaps, and Architecture

## 1. Problem Statement
Manual walkaround inspections at MRO facilities are highly time-consuming, inspector-dependent, and sensitive to environmental conditions (such as harsh hangar lighting and shadowing). Subtle defects including micro-cracks, hairline dents, and missing fasteners frequently remain undetected. Furthermore, existing inspection workflows lack a unified digital history linking aircraft details, past findings, and spatial damage coordinates, preventing effective fleet-level analytics.

## 2. Identified Research Gaps
During our literature and market review, we identified the following critical gaps in existing Computer Vision (CV) solutions for aircraft inspection:
* **The Multi-Scale Failure:** Standard CNNs and early YOLO variants struggle to detect micro-cracks (2-pixel width) while simultaneously tracking massive structural dents due to rigid grid sizing.
* **Cascading Pipeline Errors:** Many current systems use redundant multi-model cascades (e.g., YOLO for bounding boxes -> EfficientNet for classification), creating severe latency bottlenecks.
* **False Positive Bias:** Existing models frequently misidentify healthy panel gaps, rivets, and hangar lighting reflections as structural damage.
* **Data Disconnection:** Current AI inspection tools output raw JSON coordinate data without translating it into actionable, historically aware maintenance records.

## 3. Proposed Solution
**CAMIS** resolves these gaps by replacing multi-stage pipelines with an Integrated Multi-Task Pipeline:
1. **Digital Image Processing (DIP):** An offline OpenCV suite to normalize lighting, enhance shadow contrast, and sharpen crack boundaries prior to inference.
2. **End-to-End Transformer Detection:** Utilizing RT-DETR-L paired with SAHI (Slicing Aided Hyper Inference) to identify damage coordinates and severity simultaneously across both micro and macro scales.
3. **Automated LLM Reporting:** Routing detection outputs through a Gemma 3 AI engine to generate contextual maintenance reports.

## 4. Architectural Pipeline Draft
* **Input:** Inspector uploads a high-resolution image ($1280 \times 1280$).
* **Enhancement:** Image passes through a 7-stage DIP pipeline (Laplacian Variance, Gray World, Gamma, CLAHE, Inpainting, Bilateral Filter, Unsharp Masking).
* **Inference:** $640 \times 640$ resized copy is passed to RT-DETR-L. SAHI slices the image into $320 \times 320$ overlapping tiles for high-precision micro-defect scanning.
* **Consolidation:** Weighted Box Fusion (WBF) merges overlapping detections.
* **Output:** Coordinated data is pushed to a Fleet Management Frontend (Power BI) and a PDF report is generated via LLM.
