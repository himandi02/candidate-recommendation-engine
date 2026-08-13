"""
Vectorization & Feature Engineering - Method Comparison

Compares the CHARACTERISTICS of each vectorization method:
- dimensionality
- sparsity/density
- representation type
- example vector values

This script does NOT calculate cosine similarity or rank candidates.
That is handled by the Similarity Computation & Recommendation part
of the project.

Run from the project root, after running:
    python src/vectorize_tfidf.py
    python src/vectorize_word2vec.py

Then run:
    python src/compare_vectorizations.py
"""

import os
import joblib
import numpy as np
import scipy.sparse as sp


MODELS_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_pair(prefix):
    """
    Load the candidate and job vector files for a method.

    Returns:
        (candidate_data, job_data) if both files exist.
        None if either file is missing.
    """

    candidate_path = os.path.join(
        MODELS_DIR,
        f"{prefix}_candidate_vectors.pkl"
    )

    job_path = os.path.join(
        MODELS_DIR,
        f"{prefix}_job_vectors.pkl"
    )

    if not (
        os.path.exists(candidate_path)
        and os.path.exists(job_path)
    ):
        return None

    candidate_data = joblib.load(candidate_path)
    job_data = joblib.load(job_path)

    return candidate_data, job_data


def describe_method(name, candidate_data, job_data):
    """
    Summarise the characteristics of one vectorization method.

    No similarity scoring is performed here.
    """

    candidate_vectors = candidate_data["vectors"]
    job_vectors = job_data["vectors"]

    n_candidates, n_dimensions = candidate_vectors.shape
    n_jobs = job_vectors.shape[0]

    # Check whether the vectors are sparse or dense
    is_sparse = sp.issparse(candidate_vectors)

    if is_sparse:
        total_values = (
            candidate_vectors.shape[0]
            * candidate_vectors.shape[1]
        )

        density = candidate_vectors.nnz / total_values

        sample_vector = (
            candidate_vectors[0]
            .toarray()
            .ravel()
        )

    else:
        density = (
            np.count_nonzero(candidate_vectors)
            / candidate_vectors.size
        )

        sample_vector = candidate_vectors[0]

    lines = [
        f"Method: {name}",
        f"  Candidates: {n_candidates}",
        f"  Jobs: {n_jobs}",
        f"  Vector dimensions: {n_dimensions}",
        f"  Representation: "
        f"{'sparse' if is_sparse else 'dense'}",
        f"  Density (non-zero fraction): {density:.3f}",
        "  Example - first 8 values of the first candidate vector:",
        f"    {np.round(sample_vector[:8], 3).tolist()}",
        "",
    ]

    return lines


def run():

    all_lines = [
        "Vectorization Method Comparison",
        "=" * 40,
        "",
        "This compares vector CHARACTERISTICS only:",
        "- dimensionality",
        "- representation type",
        "- sparsity/density",
        "- example vector values",
        "",
        "Similarity scoring and candidate ranking are NOT performed",
        "by this script. Those operations are handled by the",
        "Similarity Computation & Recommendation component.",
        "",
    ]

    found_any = False

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    tfidf = load_pair("tfidf")

    if tfidf:
        all_lines += describe_method(
            "TF-IDF",
            *tfidf
        )

        found_any = True

        print(
            "[Compare] TF-IDF characteristics recorded"
        )

    else:
        print(
            "[Compare] Skipped TF-IDF - "
            "run vectorize_tfidf.py first"
        )

    # --------------------------------------------------------
    # Word2Vec
    # --------------------------------------------------------

    word2vec = load_pair("word2vec")

    if word2vec:
        all_lines += describe_method(
            "Word2Vec",
            *word2vec
        )

        found_any = True

        print(
            "[Compare] Word2Vec characteristics recorded"
        )

    else:
        print(
            "[Compare] Skipped Word2Vec - "
            "run vectorize_word2vec.py first (optional)"
        )

    # --------------------------------------------------------
    # Sentence-BERT
    # --------------------------------------------------------

    bert = load_pair("bert")

    if bert:
        all_lines += describe_method(
            "Sentence-BERT",
            *bert
        )

        found_any = True

        print(
            "[Compare] Sentence-BERT characteristics recorded"
        )

    else:
        print(
            "[Compare] Skipped BERT - "
            "run vectorize_bert.py first (optional)"
        )

    # --------------------------------------------------------
    # Make sure at least one method exists
    # --------------------------------------------------------

    if not found_any:
        raise SystemExit(
            "No vectorization outputs found. "
            "Run vectorize_tfidf.py first."
        )

    # --------------------------------------------------------
    # Discussion for the report
    # --------------------------------------------------------

    all_lines += [
        "Discussion (for the report):",
        "",
        "- TF-IDF represents each document as a sparse vector "
        "over the shared vocabulary, weighted by term frequency "
        "and inverse document frequency.",
        "",
        "- TF-IDF is simple and interpretable because the features "
        "correspond directly to words and n-grams in the corpus.",
        "",
        "- TF-IDF can work well on a small corpus because it does "
        "not need to learn word relationships from a large training "
        "dataset.",
        "",
        "- Word2Vec represents each document as a dense vector by "
        "averaging the learned word embeddings.",
        "",
        "- Word2Vec can capture semantic relationships between words, "
        "but the current project has only 12 CVs and 4 job "
        "descriptions. Therefore, the Word2Vec model is trained on "
        "a very small corpus.",
        "",
        "- Because of the small corpus, Word2Vec should be treated "
        "as a comparison method rather than necessarily the primary "
        "representation for the recommendation system.",
        "",
        "- Sentence-BERT, if included, uses a pretrained language "
        "model and therefore benefits from knowledge learned from "
        "large external datasets. However, it requires additional "
        "model setup and resources.",
        "",
    ]

    # --------------------------------------------------------
    # Save comparison report
    # --------------------------------------------------------

    output_path = os.path.join(
        RESULTS_DIR,
        "vectorization_comparison.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write("\n".join(all_lines))

    print(
        f"[Compare] Saved -> {output_path}"
    )


# ------------------------------------------------------------
# Run script
# ------------------------------------------------------------

if __name__ == "__main__":
    run()