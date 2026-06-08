# ✈️ CAMIS: Computer-Aided Aircraft Maintenance & Inspection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch Lightning](https://img.shields.io/badge/PyTorch-Lightning-EE4C2C)
![OpenCV](https://img.shields.io/badge/OpenCV-DIP-green)
![LLM](https://img.shields.io/badge/LLM-Gemma_3-purple)

**CAMIS** is an AI-driven, end-to-end computer vision platform designed for Next-Generation MRO (Maintenance, Repair, and Overhaul) facilities. It automates the detection, classification, and reporting of exterior aircraft structural damage, replacing fragmented manual inspections with a unified, fleet-wide digital tracking system.

## 📖 Table of Contents
- [The Problem & Our Solution](#the-problem--our-solution)
- [System Architecture](#system-architecture)
- [Key Engineering Features](#key-engineering-features)
- [Dataset & Classes](#dataset--classes)
- [Tech Stack](#tech-stack)
- [Team Gabbar](#team-gabbar)

---

## 🛑 The Problem & 💡 Our Solution
**The Status Quo:** Manual walkaround inspections are slow, highly subjective, and prone to human error caused by inspector fatigue and poor hangar lighting. Critical hairline cracks or early-stage corrosion often go unnoticed. Furthermore, isolated paper-based reports make fleet-wide structural tracking impossible.

**The CAMIS Solution:** We replace redundant multi-stage cascading models with a **2-Stage Integrated Multi-Task Pipeline**. CAMIS ingests high-resolution inspection photos, physically enhances them using a 7-stage Digital Image Processing (DIP) suite, and feeds them into a state-of-the-art **RT-DETR-L** transformer. The system concurrently extracts spatial coordinates and damage severity, passing the data to an **LLM** for automated compliance reporting and Power BI for fleet-level heatmapping.

---

## ⚙️ System Architecture

1. **Digital Image Processing (DIP) Pipeline:** Raw images undergo Laplacian Variance (blur detection), Gray World Assumption (white balance), Gamma Correction, CLAHE (contrast enhancement), and Unsharp Masking to prepare metallic surfaces for the AI.
2. **SAHI (Slicing Aided Hyper Inference):** High-resolution images are mathematically sliced into overlapping tiles to prevent the compression/destruction of micro-crack pixels.
3. **RT-DETR-L:** The Real-Time Detection Transformer utilizes global self-attention to identify both macroscopic dents and microscopic splits simultaneously.
4. **Weighted Box Fusion (WBF):** Consolidates overlapping bounding boxes into highly accurate, tight localizations.
5. **Gemma 3 LLM Reporting:** Contextualizes the detected bounding boxes against the aircraft's historical logs to write human-readable maintenance narratives.

---

## 🚀 Key Engineering Features
- **Micro & Macro Detection:** Eliminates the "small object problem" via SAHI integration.
- **Robustness to Hangar Environments:** The OpenCV DIP pipeline actively corrects harsh hangar glare and deep wing-root shadows.
- **False Positive Elimination:** Model trained with 10-15% "Hard Negative" (clean metal) datasets to prevent misidentification of healthy rivets or reflections.
- **Automated Business Intelligence:** Connects detection coordinates directly to Power BI, allowing MRO managers to visualize repeat-failure heatmaps across entire aircraft fleets.

---

## 📊 Dataset & Classes
The model is trained on a unified, high-quality dataset of **~25,000 images** aggregated from UTS Aircraft Defect, Youssef Donia, Lemi Debele, and custom Roboflow annotations.

**Master Taxonomies:**
1. `Splits & Breaks` (Micro-cracks, structural fractures)
2. `Burnouts` (Lightning strikes, severe corrosion)
3. `Wear & Tear` (Dents, paint degradation, minor scratches)

---

## 💻 Tech Stack
* **Deep Learning Framework:** PyTorch, PyTorch Lightning
* **Vision Models:** RT-DETR-L (Hugging Face / TIMM)
* **Inference Optimization:** SAHI, WBF
* **Image Processing:** OpenCV, NumPy
* **Automated Reporting:** Gemma 3 (via FastAPI backend)
* **Analytics:** Microsoft Power BI
* **Compute:** Optimized for NVIDIA H200 (141GB VRAM) using BFloat16 Mixed Precision.

---

## 👨‍💻 Team Gabbar
This project was developed as an engineering initiative by Team-Gabbar from **Amrita Vishwa Vidyapeetham**.

* **Gopika Gokul** (CB.EN.U4EEE23009)
* **Rithesh N** (CB.EN.U4EEE23130)
* **Shivani K C** (CB.EN.U4EEE23139)
* **Vishnu K Mahesh** (CB.EN.U4EEE23145)

---
*Disclaimer: This repository contains the architecture and methodology for the CAMIS project. Proprietary aircraft data and specific L&T MRO integration scripts may be omitted for confidentiality.*
