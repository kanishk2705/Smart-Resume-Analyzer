# 🚀 Smart Resume Analyzer (AI-Powered ATS)

# 🚀 Phase 1 (NLP Keyword Matching) ✅
  
> *A resume analysis tool that uses Natural Language Processing (NLTK) to clean, lemmatize, and match resumes against Job Descriptions.*

![Smart Resume Analyzer Demo](assets/phase_1_demo.png)

# 🤖 Phase 2 (Semantic Search & Explainability) ✅

> *A Next-Gen ATS Simulator that uses Deep Learning (SBERT Transformers) to understand the **meaning** of a resume, not just keywords.*

![Phase 2 Analysis Demo](assets/phase_2_demo.png)

# 🐳 Phase 3 (Containerized Microservice) ✅
  
> *A Production-Ready ATS Simulator. Containerized with Docker for "Run Anywhere" deployment, featuring Semantic Search and AI-Powered Explainability.*

![Phase 3](assets/phase_3_demo.png)

#  Phase 4 (LLM Integration) ✅  

> *An End-to-End MLOps project that analyzes resumes using Deep Learning (SBERT) and generates personalized upskilling plans using Google Gemini 2.0.*

![Project Demo](assets/phase_4_demo.png)

## 📌 The Problem
Traditional Applicant Tracking Systems (ATS) are broken. They reject qualified candidates because of simple keyword mismatches (e.g., rejecting "ML Ops" when looking for "DevOps"). 

## 📌 Project Overview

Most ATS (Applicant Tracking Systems) reject qualified candidates because they lack specific keywords.

Unlike basic keyword matchers that fail on simple word variations (e.g., "Analyze" vs. "Analysis"), this **Phase 1** implementation integrates an **NLP Pipeline** to normalize text before matching. It identifies the gap between a candidate's resume and the job requirements using mathematical vectorization.

In **Phase 2**, we upgraded from simple keyword matching to **Semantic Understanding**. By integrating **BERT-based Transformers**, this tool now understands that *"Building Dashboards"* is semantically similar to *"Data Visualization"*, identifying qualified candidates even if they don't use the exact phrasing of the Job Description (JD).

The **Phase 3**, project is an **Engineering-Grade ATS Simulator** that evolves beyond simple scripts. It uses **Deep Learning (Transformers)** to understand the semantic meaning of a resume and is deployed as a **Dockerized Microservice**, ensuring it runs consistently on any machine (Local, AWS, Azure, etc.).

**Phase 4**: Introduced a GenAI-powered feature that identifies candidates’ top three skill gaps and provides relevant online learning resources, going beyond keyword matching and hero sentence suggestions.

## 🛠️ Tech Stack
* **Core:** Python 3.10, PyTorch (CPU-Optimized).
* **AI Models:** `sentence-transformers` (SBERT), Google Gemini 2.0 Flash.
* **Infrastructure:** Docker & Docker Compose (Microservice).
* **Frontend:** Streamlit (Optimized with Session State & WebSocket fixes).
* **Resilience:** Custom "No-Library" API Client to handle model versioning and quota management.

## ✨ Key Features

I built an **AI-First ATS** that focuses on *context*, not just keywords.

1.  **Semantic Matching:** Uses **SBERT Transformers** to understand that "Dashboarding" == "Data Visualization".
2.  **Explainable AI:** Highlights "Hero Sentences" so users know exactly what scored points.
3.  **🐳 Portable & Scalable:** Fully containerized. No "it works on my machine" issues—just pull the image and run.
4.  **⚡ Instant Startup:** Uses **Docker Layer Caching** to pre-bake heavy AI models (80MB+) into the image, so the app launches instantly.
5.  **GenAI Career Coach:** Instead of just saying "You lack SQL", it uses **Gemini 2.0 Flash** to generate a specific **Google Search Link** for the best course and a practical **Project Idea** to fill that gap.

## 📂 Project Structure
```text
resume-analyzer/
│
├── src/
│   ├── parser.py       # Handles PDF/DOCX extraction
│   ├── cleaner.py      # NLTK Pipeline (Lemmatization & Cleaning)
│   └── analyzer.py     # Core logic (Cosine Similarity)
|   └──recommendations.py # Gemini API Integration
|── Dockerfile          # Container instructions (Multi-layer build)
├── .dockerignore       # Build context optimization
├── app.py              # Main Streamlit Interface
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```
## 🚀 How to Run (Docker)
The app is fully containerized. You don't need to install Python or PyTorch.

1.  **Build the Image**
    ```bash
    docker build -t resume-analyzer .
    ```

2.  **Run with API Key**
    *Replace `YOUR_KEY` with your Google Gemini API Key.*
    ```bash
    # Windows PowerShell
    $key = "YOUR_KEY"
    docker run -p 8501:8501 -e GEMINI_API_KEY="$key" resume-analyzer
    ```

3. Access App Open your browser to ``` http://localhost:8501 ```

🚀 Option 2: Run Locally

1. Clone the Repository
```bash
git clone https://github.com/kanishk2705/Smart-Resume-Analyzer.git
cd resume-analyzer
```
2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Run the Application
```bash
streamlit run app.py
```
4. Usage
```bash
    . Upload your Resume (PDF or DOCX).
    . Paste the Job Description.
    . Click "Analyze" to see your score and missing keywords.
```
🗺️ Engineering Roadmap

This project follows an iterative engineering path, moving from basic scripts to a production-grade AI application.

[x] Phase 1: NLP-Enhanced Matcher - Implemented NLTK pipeline for robust keyword matching.

[x] Phase 2: The Intelligence Layer - Integrating BERT/SBERT Transformers for Semantic Context.

[x] Phase 3: DevOps & Scalability - Dockerizing the application for portable deployment.

[x] Phase 4: GenAI Mentor - Using LLMs (Gemini) to generate custom learning paths.

``` Built by A C KANISHK as part of an end-to-end Machine Learning Engineering study. ```