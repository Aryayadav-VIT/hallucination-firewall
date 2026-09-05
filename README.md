# Multimodal Hallucination Firewall

### Multimodal Hallucination Detection using CLIP and Machine Learning

A multimodal machine learning system designed to detect potential hallucinations in image-question-answering systems by combining visual and textual representations with supervised machine learning.

---

## 📌 Project Overview

Vision-Language Models (VLMs) can sometimes generate answers that are not supported by the visual content of an image.

This project develops a **Multimodal Hallucination Firewall** that analyzes image-question-answer relationships and predicts whether an answer is:

- ✅ Supported
- ⚠️ Hallucinated

The system uses **CLIP (ViT-B/32)** to obtain multimodal representations and evaluates multiple machine learning classifiers to identify the best-performing hallucination detector.

---

## 🎯 Objectives

- Detect hallucinated answers in multimodal question answering.
- Combine image and textual information for hallucination detection.
- Extract semantic representations using CLIP.
- Compare multiple machine learning models.
- Build a final prediction layer that acts as a hallucination firewall.
- Evaluate the system using standard classification metrics.

---

## 🧠 Methodology

The overall pipeline is:

```text
POPE Dataset
     │
     ▼
Data Cleaning & Validation
     │
     ▼
Image Verification / Linking
     │
     ▼
CLIP ViT-B/32
     │
     ├── Image Features
     └── Text Features
            │
            ▼
     Multimodal Features
            │
            ▼
 Leakage-Safe Train / Validation / Test Split
            │
            ▼
 ┌─────────────────────────────┐
 │ Machine Learning Models     │
 │                             │
 │ • Logistic Regression       │
 │ • SVM                       │
 │ • Random Forest             │
 │ • Extra Trees               │
 │ • Gradient Boosting         │
 └─────────────────────────────┘
            │
            ▼
       Model Comparison
            │
            ▼
       Best Model: SVM
            │
            ▼
   Hallucination Firewall
            │
            ▼
 Supported / Hallucinated
