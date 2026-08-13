# Candidate Recommendation System

---

## Project Overview

This project focuses on building an automated **Candidate Recommendation Engine** that matches job applicants (CVs) with job descriptions. The system processes unstructured candidate resumes and job requirements, converts them into numerical vectors, and computes similarity scores to rank the top candidates automatically.

The main objective is to assist recruiters in identifying top-fit talent efficiently without manually sifting through hundreds of CVs.

---

## Project Objectives

- Preprocess and clean unstructured CV and Job Description text (tokenization, stop-word removal, lemmatization).
- Build vector representations for candidates and jobs using **TF-IDF** (with options for Word2Vec/BERT).
- Calculate candidate-to-job match scores using **Cosine Similarity**.
- Generate ranked candidate recommendations (Top-3 / Top-5) for given job openings.
- Evaluate system performance using Precision, Recall, and F1-Score against a ground-truth dataset.

---

## Deliverables

- **Recommendation Engine:** Python pipeline converting CVs and JDs into vector match scores.
- **Project Documentation Report:** Comprehensive technical report covering methodology, system architecture, evaluation results, and future improvements.
- **Presentation & Demo:** Slide deck and video demonstration showing end-to-end matching.

---

## Folder Structure

````text
candidate-recommendation-engine/
│
├── data/       # Raw CV text files and job descriptions
├── models/     # Saved vectorizer and scoring models
├── src/        # Python source code (cleaning, vectorization, similarity)
├── results/    # Generated candidate rankings and evaluation metrics
└── docs/       # Project report, presentation slides, and README

## Setup and Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd candidate-recommendation-engine
````

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
```

### 3. Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Required Libraries

Install the required Python libraries:

```powershell
python -m pip install numpy pandas scikit-learn nltk spacy
```

### 5. Verify the Installation

Run the following command:

```powershell
python -c "import numpy, pandas, sklearn, nltk, spacy; print('Core packages imported successfully')"
```

If the installation is successful, you should see:

```text
Core packages imported successfully
```

### 6. Run the Project

Open the preprocessing notebook:

```text
src/preprocessing.ipynb
```

Run the notebook cells in order to preprocess the CVs and job descriptions and continue with the candidate recommendation pipeline.

## Tech Stack & Tools

Language: Python

Libraries: scikit-learn, nltk / spacy, pandas, numpy

Version Control: Git & GitHub
