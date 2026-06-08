# 📊 Milestone 2: Dataset Acquisition, Annotation, and Algorithm Research

## 1. Dataset Acquisition
To ensure the model generalizes across diverse MRO environments and aircraft types, we compiled a massive unified dataset of approximately **14,000 to 16,000 images** from the following sources:
* **UTS Aircraft Defect Detection:** 9,352 images (Dents, Fastener Damage).
* **Youssef Donia Aircraft Damage:** 3,174 images (Cracks, Scratches).
* **Lemi Debele Aircraft Surface Damage:** 1,200 images (Corrosion).
* **Dibya Dillip Dataset:** 1,115 images (Mixed defects).
* **Custom Verification Set:** 50-100 manually captured/selected images.

## 2. Image Annotation (Roboflow Workflow)
To maintain strict domain accuracy and validate the open-source labels, a controlled subset was manually annotated.
* **Platform:** Roboflow (Object Detection).
* **Taxonomy Alignment:** Annotations were standardized into specific fault categories: `Structural_Crack`, `Mild_Dent`, `Deep_Dent`, `Missing_Fastener`, `Light_Corrosion`, etc.
* **Bounding Box Strategy:** Strict adherence to "Tight Bounding Boxes" to ensure the self-attention mechanisms of our transformer model do not incorrectly correlate healthy metal with damage features.

## 3. Algorithm Research & Selection
Following Milestone 1, we finalized the specific mathematical algorithms required to execute the pipeline:

### A. Digital Image Processing (DIP) Algorithms
Research confirmed that transformers require heavily conditioned inputs to prevent false positives in harsh hangar lighting.
* **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Selected for localized contrast enhancement in deep wing-root shadows.
* **Bilateral Filtering:** Selected for edge-preserving noise reduction, ensuring high-frequency camera noise isn't sharpened into false cracks.
* **Unsharp Masking:** Selected to aggressively sharpen the edges of hairline fractures.

### B. Detection Architecture Algorithms
* **Primary Engine: RT-DETR-L (Real-Time Detection Transformer).** Selected over YOLOv8 due to the transformer's multi-scale self-attention, which natively understands global context better than convolutional grids.
* **Small Object Strategy: SAHI (Slicing Aided Hyper Inference).** Selected to preserve the pixel density of micro-defects during inference. 
* **Box Consolidation: WBF (Weighted Box Fusion).** Selected over standard NMS. WBF averages overlapping boxes based on confidence scores rather than simply deleting them, resulting in vastly superior localization precision.
