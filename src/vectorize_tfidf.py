"""
Vectorization & Feature Engineering - Method 1: TF-IDF

Reads the cleaned candidate and job text files produced in preprocessing,
converts them into TF-IDF vectors in a SHARED feature space, and saves:

- the fitted vectorizer      -> models/tfidf_vectorizer.pkl
- the candidate vectors      -> models/tfidf_candidate_vectors.pkl
- the job vectors            -> models/tfidf_job_vectors.pkl
- a human-readable summary   -> results/tfidf_summary.txt

Run from the project root:

python src/vectorize_tfidf.py
"""

import os
import glob
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# Paths

CLEANED_CV_DIR = "data/cleaned_cvs"
CLEANED_JOB_DIR = "data/cleaned_jobs"
MODELS_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_text_files(directory):
    """
    Read every .txt file in a directory into a dictionary:

    {document_id: text}

    Files are sorted to ensure reproducible ordering.
    """
    docs = {}

    for filepath in sorted(glob.glob(os.path.join(directory, "*.txt"))):
        doc_id = os.path.splitext(os.path.basename(filepath))[0]

        with open(filepath, "r", encoding="utf-8") as f:
            docs[doc_id] = f.read().strip()

    return docs


def run():
    # 1. Load candidates and jobs
    candidate_docs = load_text_files(CLEANED_CV_DIR)
    job_docs = load_text_files(CLEANED_JOB_DIR)

    print(
        f"[TF-IDF] Loaded "
        f"{len(candidate_docs)} candidates, "
        f"{len(job_docs)} jobs"
    )

    if not candidate_docs or not job_docs:
        raise RuntimeError(
            "No documents found in "
            "data/cleaned_cvs or data/cleaned_jobs."
        )

    candidate_ids = list(candidate_docs.keys())
    job_ids = list(job_docs.keys())

    candidate_texts = [
        candidate_docs[candidate_id]
        for candidate_id in candidate_ids
    ]

    job_texts = [
        job_docs[job_id]
        for job_id in job_ids
    ]

    # 2. Combine candidates and jobs.
    # One vectorizer is fitted on both so they share
    # the same vocabulary and feature columns.
    all_texts = candidate_texts + job_texts

    # 3. Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        min_df=1,
        max_df=0.95,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    # 4. Fit TF-IDF and transform all documents
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    n_candidates = len(candidate_texts)

    # Separate candidates and jobs
    candidate_vectors = tfidf_matrix[:n_candidates]
    job_vectors = tfidf_matrix[n_candidates:]

    # 5. Display vector characteristics
    vocab_size = len(vectorizer.get_feature_names_out())

    print(f"[TF-IDF] Vocabulary size: {vocab_size}")
    print(f"[TF-IDF] Candidate matrix: {candidate_vectors.shape}")
    print(f"[TF-IDF] Job matrix: {job_vectors.shape}")

    # 6. Save fitted vectorizer
    vectorizer_path = os.path.join(
        MODELS_DIR,
        "tfidf_vectorizer.pkl"
    )

    joblib.dump(vectorizer, vectorizer_path)

    # 7. Save candidate vectors
    candidate_vectors_path = os.path.join(
        MODELS_DIR,
        "tfidf_candidate_vectors.pkl"
    )

    joblib.dump(
        {
            "ids": candidate_ids,
            "vectors": candidate_vectors
        },
        candidate_vectors_path
    )

    # 8. Save job vectors
    job_vectors_path = os.path.join(
        MODELS_DIR,
        "tfidf_job_vectors.pkl"
    )

    joblib.dump(
        {
            "ids": job_ids,
            "vectors": job_vectors
        },
        job_vectors_path
    )

    # 9. Create human-readable summary
    feature_names = np.array(
        vectorizer.get_feature_names_out()
    )

    lines = [
        "TF-IDF Vectorization Summary",
        "=" * 40,
        f"Candidates: {n_candidates}",
        f"Jobs: {len(job_ids)}",
        f"Vocabulary: {vocab_size}",
        "",
        "Vectorization settings:",
        "  min_df = 1",
        "  max_df = 0.95",
        "  ngram_range = (1, 2)",
        "  sublinear_tf = True",
        "",
        f"Candidate matrix shape: {candidate_vectors.shape}",
        f"Job matrix shape: {job_vectors.shape}",
        "",
    ]

    # 10. Show top TF-IDF terms for each job
    for i, job_id in enumerate(job_ids):
        row = job_vectors[i].toarray().ravel()

        top_idx = row.argsort()[::-1][:10]

        lines.append(f"{job_id} - top terms:")

        for j in top_idx:
            if row[j] > 0:
                lines.append(
                    f"    {feature_names[j]:<25} "
                    f"{round(row[j], 3)}"
                )

        lines.append("")

    summary_path = os.path.join(
        RESULTS_DIR,
        "tfidf_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write("\n".join(lines))

    # 11. Completion message
    print(
        "[TF-IDF] Saved vectorizer, vectors, and summary to "
        f"'{MODELS_DIR}/' and '{RESULTS_DIR}/'"
    )

    return (
        vectorizer,
        candidate_ids,
        candidate_vectors,
        job_ids,
        job_vectors,
    )


if __name__ == "__main__":
    run()