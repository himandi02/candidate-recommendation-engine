# Candidate Recommendation System

---

## Project Overview
This project focuses on building an automated **Candidate Recommendation Engine** that matches job applicants (CVs) with job descriptions. The system processes unstructured candidate resumes and job requirements, converts them into numerical vectors, and computes similarity scores to rank the top candidates automatically.

The main objective is to assist recruiters in identifying top-fit talent efficiently without manually sifting through hundreds of CVs.

---

## Project Objectives
* Preprocess and clean unstructured CV and Job Description text (tokenization, stop-word removal, lemmatization).
* Build vector representations for candidates and jobs using **TF-IDF** (with options for Word2Vec/BERT).
* Calculate candidate-to-job match scores using **Cosine Similarity**.
* Generate ranked candidate recommendations (Top-3 / Top-5) for given job openings.
* Evaluate system performance using Precision, Recall, and F1-Score against a ground-truth dataset.

---

## Deliverables
* **Recommendation Engine:** Python pipeline converting CVs and JDs into vector match scores.
* **Project Documentation Report:** Comprehensive technical report covering methodology, system architecture, evaluation results, and future improvements.
* **Presentation & Demo:** Slide deck and video demonstration showing end-to-end matching.

---

## Folder Structure
```text
candidate-recommendation-engine/
│
├── data/       # Raw CV text files and job descriptions
├── models/     # Saved vectorizer and scoring models
├── src/        # Python source code (cleaning, vectorization, similarity)
├── results/    # Generated candidate rankings and evaluation metrics
└── docs/       # Project report, presentation slides, and README

## Tech Stack & Tools

Language: Python

Libraries: scikit-learn, nltk / spacy, pandas, numpy

Version Control: Git & GitHub