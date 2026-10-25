# Multimodal Hallucination Firewall: An Evidence-Aware System for Detecting Unsupported Claims in Image-Based AI Answers

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 16 Passed](https://img.shields.io/badge/Tests-16%20Passed-brightgreen.svg)](tests/)

An evidence-aware post-hoc guardrail system designed to audit, detect, and explain unsupported or fabricated claims in Vision-Language Model (VLM) outputs. Developed as an MSc Artificial Intelligence and Machine Learning research prototype.

---

## 📌 Executive Summary & Research Motivation

Large Vision-Language Models (such as GPT-4V, LLaVA, Gemini, and Claude 3.5 Sonnet) have demonstrated impressive multimodal comprehension. However, they frequently suffer from **object hallucination**—generating textual assertions that describe objects, attributes, actions, or quantities absent from the underlying visual input.

In high-stakes domains—such as medical diagnostics, autonomous surveillance, and automated inspection—unsupported claims introduce critical vulnerabilities. 

The **Multimodal Hallucination Firewall** functions as an external, post-hoc verification barrier between the vision-language model and end users:
1. **Parses** question-answer pairs to isolate concrete claims, entity targets, and assertion polarity.
2. **Classifies** the claim into one of eight distinct multimodal hallucination categories.
3. **Extracts** multimodal representations via OpenAI's CLIP (ViT-B/32) vision-language backbone.
4. **Evaluates** claims using trained statistical pipelines (8-feature SVM with calibrated Platt scaling) or contrastive zero-shot verification heads.
5. **Enforces Scientific Honesty**: Explicitly flags model blind spots (e.g., counting, spatial compositionality) rather than generating ungrounded pseudo-certainty.

---

## 🏗️ System Architecture

The project is structured into modular, decoupled Python components:

```text
hallucination-firewall-main/
├── app.py                      # Production Streamlit UI (5-tab interactive dashboard)
├── requirements.txt            # Project dependencies with version bounds
├── LICENSE                     # MIT Open Source License
├── models/
│   ├── hallucination_firewall_corrected_bundle (3).joblib   # 8-feature SVM pipeline
│   └── hallucination_firewall_svm.pkl                       # Legacy 3-feature model
├── src/
│   ├── config.py               # Constants, thresholds (0.27), schemas, Enums
│   ├── model_loader.py         # Dynamic path resolution & bundle validation
│   ├── question_parser.py      # Category classifier & entity extraction engine
│   ├── vision_features.py      # CLIP ViT-B/32 cache & 8-feature vector extractor
│   ├── hallucination_checks.py # Evidence-aware claim verification dispatcher
│   ├── demo_cases.py           # 12-case benchmark lab metadata & observations
│   └── generate_demo_assets.py # Reproducible synthetic test asset generator
├── assets/
│   └── demo_images/            # 12 clean synthetic test images (cases 1-12)
└── tests/
    ├── test_model_loader.py    # Bundle integrity & schema tests
    ├── test_question_parser.py # Regex, polarity, category routing tests
    ├── test_vision_features.py # CLIP ViT-B/32 feature dimension tests
    ├── test_hallucination_checks.py # End-to-end verification pipeline tests
    └── test_demo_cases.py      # Asset presence & integrity tests
```

---

## 🚀 Quick Start (Local Execution)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run Automated Tests
```bash
python -m unittest discover -s hallucination-firewall-main/tests -p "test_*.py" -v
```

See [hallucination-firewall-main/README.md](hallucination-firewall-main/README.md) for full scientific documentation, calibration formulas, and category breakdown.
