"""
Vectorization & Feature Engineering - Method 2: Word2Vec

Trains a Word2Vec model on the combined candidate + job corpus.

Each document's vector is the average of its word vectors
(mean pooling).

Saves:
- models/word2vec_model.bin
- models/word2vec_candidate_vectors.pkl
- models/word2vec_job_vectors.pkl

Run from the project root:

python src/vectorize_word2vec.py
"""

import os
import glob
import joblib
import numpy as np
from gensim.models import Word2Vec


CLEANED_CV_DIR = "data/cleaned_cvs"
CLEANED_JOB_DIR = "data/cleaned_jobs"
MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)


VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 1
EPOCHS = 100


def load_text_files(directory):
    docs = {}

    for filepath in sorted(
        glob.glob(os.path.join(directory, "*.txt"))
    ):
        doc_id = os.path.splitext(
            os.path.basename(filepath)
        )[0]

        with open(filepath, "r", encoding="utf-8") as f:
            docs[doc_id] = f.read().strip()

    return docs


def document_vector(model, tokens):
    """Average the Word2Vec vectors of all known words."""
    vectors = [
        model.wv[word]
        for word in tokens
        if word in model.wv
    ]

    if not vectors:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


def run():

    # 1. Load candidates and jobs
    candidate_docs = load_text_files(CLEANED_CV_DIR)
    job_docs = load_text_files(CLEANED_JOB_DIR)

    print(
        f"[Word2Vec] Loaded "
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

    # 2. Tokenize documents
    candidate_tokens = [
        candidate_docs[cid].split()
        for cid in candidate_ids
    ]

    job_tokens = [
        job_docs[jid].split()
        for jid in job_ids
    ]

    # 3. Combine candidate and job corpus
    all_tokenized = candidate_tokens + job_tokens

    # 4. Train Word2Vec
    model = Word2Vec(
        sentences=all_tokenized,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        epochs=EPOCHS,
        sg=1,
        seed=42,
    )

    print(
        f"[Word2Vec] Trained on vocabulary "
        f"of {len(model.wv)} words"
    )

    # 5. Create document vectors using mean pooling
    candidate_vectors = np.array([
        document_vector(model, tokens)
        for tokens in candidate_tokens
    ])

    job_vectors = np.array([
        document_vector(model, tokens)
        for tokens in job_tokens
    ])

    print(
        f"[Word2Vec] Candidate matrix: "
        f"{candidate_vectors.shape}"
    )

    print(
        f"[Word2Vec] Job matrix: "
        f"{job_vectors.shape}"
    )

    # 6. Save Word2Vec model
    model.save(
        os.path.join(
            MODELS_DIR,
            "word2vec_model.bin"
        )
    )

    # 7. Save candidate vectors
    joblib.dump(
        {
            "ids": candidate_ids,
            "vectors": candidate_vectors
        },
        os.path.join(
            MODELS_DIR,
            "word2vec_candidate_vectors.pkl"
        )
    )

    # 8. Save job vectors
    joblib.dump(
        {
            "ids": job_ids,
            "vectors": job_vectors
        },
        os.path.join(
            MODELS_DIR,
            "word2vec_job_vectors.pkl"
        )
    )

    print(
        f"[Word2Vec] Saved model and vectors "
        f"to '{MODELS_DIR}/'"
    )

    return (
        model,
        candidate_ids,
        candidate_vectors,
        job_ids,
        job_vectors,
    )


if __name__ == "__main__":
    run()