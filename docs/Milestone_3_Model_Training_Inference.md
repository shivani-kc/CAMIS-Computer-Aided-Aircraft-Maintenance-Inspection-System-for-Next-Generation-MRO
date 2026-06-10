# 🚀 Milestone 3: Model Training, Simulation Evaluation, and DIP Integration

## 1. Technical Execution & Training Simulation
To develop a highly accurate detection engine, we executed the training phase utilizing optimized hardware and a strict dataset split to prevent data leakage.
* **Architecture:** YOLOv8 Large (YOLOv8l) utilizing pre-trained weights for foundational feature extraction.
* **Compute Environment:** NVIDIA A100 GPU optimized with Automatic Mixed Precision (`amp=True`) and a batch size of `64` for maximum computational throughput.
* **Dataset Split:** Rigorous **78% (Train) / 11% (Valid) / 11% (Test)** configuration to ensure an untainted evaluation on strictly unseen data.
* **Optimization Strategy:** Implemented an early stopping mechanism (`patience=15`), halting the training loop at epoch 22 when validation loss plateaued. This successfully prevented overfitting and extracted the optimum `best.pt` weights.

## 2. Simulation Results & Performance Metrics
The model was evaluated exclusively on its simulated performance against the reserved validation set, yielding highly reliable detection capabilities across structural fault categories.
* **Precision (All Classes):** 96.77% (Demonstrating minimal false positive rates).
* **Recall (All Classes):** 84.10% (Validating strong capability in identifying existing defects).
* **mAP50 (All Classes):** 89.79% (Primary performance metric).
* **mAP50-95 (All Classes):** 68.39% (Strict localization metric).
* **Simulated Inference Speed:** ~4.1ms per image (Tested on CPU architecture, proving viability for real-time edge deployment).

## 3. Digital Image Processing (DIP) Pipeline
The Digital Image Processing (DIP) pipeline serves as the foundational data conditioning layer of the CAMIS architecture. Upon image ingestion by the user, the raw payload sequentially routes through the 6-stage DIP pipeline to standardize visual features, passes into the trained AI detection model for automated defect extraction, and concludes with a structured data stream upload to Power BI for terminal telemetry visualization.
* **Stage 1: Blur Detection (Laplacian Variance):** Computes the variance of the Laplacian operator on a grayscale conversion of the input frame. Images returning a variance score below 50 are flagged as severely degraded, serving as an automated data quality gate prior to inference.
* **Stage 2: White Balance (Gray World Assumption):** Normalizes the RGB channels based on the assumption that the average reflectance of a well-illuminated scene is gray. This removes artificial color casts caused by variant ambient lighting conditions.
* **Stage 3: Dynamic Gamma Correction:** Measures global luminance and applies an adaptive non-linear Look-Up Table (LUT) lookup. It dynamically scales intensity ($\gamma = 1.8$ for underexposed frames; $\gamma = 0.7$ for overexposed frames) to recover structural details lost in deep shadows or highlights.
* **Stage 4: Contrast Limited Adaptive Histogram Equalization (CLAHE):** Converts the matrix to the LAB color space and applies localized contrast enhancement to the Lightness ($L$) channel across $8 \times 8$ tile grids. A strict clipping limit prevents the amplification of background noise while distinctively isolating hairline cracks and surface anomalies.
* **Stage 5: Edge-Preserving Denoising (Bilateral Filter):** Applies a bilateral smoothing filter ($d=9, \sigma=75$) to reduce sensor grain and surface texture noise. Unlike standard Gaussian blurs, it smooths flat regions while strictly preserving high-frequency structural boundaries and defect edges.
* **Stage 6: Structural Sharpening (Unsharp Masking):** Isolates high-frequency components by subtracting a Gaussian-blurred matrix from the denoised image, then scales these edge components back into the original frame. This heightens micro-contrast along damage boundaries to improve bounding box regression accuracy.

## 4. Testing Pipeline & Inference
Following the simulated validation, we established the inference pipeline to test the model's bounding box logic on entirely unseen MRO imagery.
* **Blind Inference:** Executed predictions on raw, unannotated test images.
* **Threshold Tuning:** Dynamically adjusted confidence (`conf=0.05` to `0.25`) and Intersection over Union (`iou=0.45`) thresholds to successfully uncover faint or complex structural damage.
* **Pipeline Integration:** The visual outputs validate the effectiveness of passing raw inspection imagery through our initial Digital Image Processing (DIP) algorithms prior to deep learning-based detection for next-generation MRO systems.

