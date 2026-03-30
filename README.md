# Text Summarize Intelligence System

AI-powered dialogue distillation system built using T5 Transformers with hardware acceleration, asynchronous processing, and a high-performance web interface.

---

## Overview

This project is an end-to-end Natural Language Processing (NLP) system designed to transform long, unorganized dialogues into concise, context-aware summaries. Unlike simple extractive tools, this system utilizes a fine-tuned **T5 (Text-to-Text Transfer Transformer)** model to understand conversational intent and rephrase it into human-like summaries.

The system is deployed using a **FastAPI** backend and provides an interactive, modern interface for real-time AI inference.

---

## Key Features

* **Abstractive Dialogue Summarization:** Generates intelligent, coherent summaries rather than simple sentence extraction.
* **Fine-Tuned Neural Engine:**
  * Model Architecture: **T5-Small**
  * Training Data: Fine-tuned on the **SAMSum dataset** (14,500+ dialogues)
  * Performance: Optimized for high ROUGE scores and low validation loss (~0.34)
* **Hardware-Aware Inference:**
  * Integrated **Apple Metal Performance Shaders (MPS)** for accelerated inference on Mac
  * Automatic fallback to **NVIDIA CUDA** or CPU based on environment
* **Modern API Architecture:**
  * Built with **FastAPI** for high-concurrency and low-latency handling
  * Asynchronous endpoints for non-blocking UI updates
* **Advanced Text Preprocessing:**
  * Regex-based cleaning engine to handle raw chat logs and noise
  * Optimized tokenization and padding management for model stability
* **Interactive Web Dashboard:**
  * Real-time "Summarize" trigger with loading animations
  * Clean, responsive UI optimized for professional readability

---

## Tech Stack

* Python
* PyTorch
* HuggingFace Transformers
* FastAPI
* Uvicorn
* HTML5 / CSS3
* JavaScript (Fetch API)

---

## System Workflow

1. Data preprocessing (cleaning raw dialogue, regex-based noise removal)
2. Tokenization and encoding (converting text into tensors for the T5 model)
3. Model inference (generating summaries using Beam Search decoding)
4. Hardware acceleration (utilizing MPS/CUDA for neural processing)
5. Post-processing (decoding tensors back to human-readable text)
6. Asynchronous delivery of results to the frontend dashboard

---

## Run Locally

```bash
git clone [https://github.com/Rajneel-Chavan/text-summarize-intelligence-system.git](https://github.com/Rajneel-Chavan/text-summarize-intelligence-system.git)
cd text-summarize-intelligence-system
pip install torch transformers fastapi uvicorn pydantic
python app.py
