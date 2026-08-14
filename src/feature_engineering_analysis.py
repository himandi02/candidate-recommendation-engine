"""
Vectorization & Feature Engineering - Feature Engineering Analysis

Documents and justifies the feature engineering decisions behind
vectorize_tfidf.py.

This script compares different TF-IDF configurations on the same corpus
and reports how each choice changes the resulting feature space.

It does NOT compute similarity or rank candidates.

Run from the project root:

python src/feature_engineering_analysis.py
"""

import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer


CLEANED_CV_DIR = "data/cleaned_cvs"
CLEANED_JOB_DIR = "data/cleaned_jobs"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_all_texts():
    """Load all cleaned candidate and job text files."""
    texts = []

    for directory in (CLEANED_CV_DIR, CLEANED_JOB_DIR):
        for filepath in sorted(
            glob.glob(os.path.join(directory, "*.txt"))
        ):
            with open(filepath, "r", encoding="utf-8") as f:
                texts.append(f.read().strip())

    return texts


# Each entry contains:
# (description, settings passed to TfidfVectorizer)

CONFIGS = [
    (
        "unigrams only, no limits",
        {
            "ngram_range": (1, 1),
            "min_df": 1,
            "max_df": 1.0,
        },
    ),

    (
        "unigrams + bigrams (chosen final config)",
        {
            "ngram_range": (1, 2),
            "min_df": 1,
            "max_df": 0.95,
            "sublinear_tf": True,
        },
    ),

    (
        "unigrams + bigrams + trigrams",
        {
            "ngram_range": (1, 3),
            "min_df": 1,
            "max_df": 0.95,
            "sublinear_tf": True,
        },
    ),

    (
        "stricter min_df=2 (drop words seen in only 1 document)",
        {
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True,
        },
    ),

    (
        "capped vocabulary, max_features=200",
        {
            "ngram_range": (1, 2),
            "min_df": 1,
            "max_df": 0.95,
            "max_features": 200,
            "sublinear_tf": True,
        },
    ),
]


def run():
    texts = load_all_texts()

    print(
        f"[FeatureEng] Loaded {len(texts)} documents "
        "(candidates + jobs combined)"
    )

    if not texts:
        raise RuntimeError(
            "No cleaned documents found in "
            "data/cleaned_cvs or data/cleaned_jobs."
        )

    lines = [
        "Feature Engineering Analysis",
        "=" * 40,
        f"Corpus size: {len(texts)} documents",
        "",
        "Comparing TF-IDF configuration choices and their effect on",
        "the resulting feature space.",
        "",
    ]

    for label, kwargs in CONFIGS:

        vectorizer = TfidfVectorizer(**kwargs)

        matrix = vectorizer.fit_transform(texts)

        vocab_size = len(
            vectorizer.get_feature_names_out()
        )

        density = (
            matrix.nnz
            / (matrix.shape[0] * matrix.shape[1])
        )

        feature_names = vectorizer.get_feature_names_out()

        # Show example bigrams if the configuration produces them.
        example_bigrams = [
            feature
            for feature in feature_names
            if " " in feature
        ][:5]

        lines.append(f"Config: {label}")
        lines.append(f"  Settings: {kwargs}")
        lines.append(
            f"  Vocabulary size (feature dimensions): "
            f"{vocab_size}"
        )
        lines.append(
            f"  Matrix density: {density:.3f}"
        )

        if example_bigrams:
            lines.append(
                f"  Example multi-word features: "
                f"{example_bigrams}"
            )

        lines.append("")

        print(
            f"[FeatureEng] {label}: "
            f"{vocab_size} features"
        )

    lines += [
        "Decision (used in vectorize_tfidf.py):",
        "",
        "- ngram_range=(1, 2): unigrams alone can lose useful "
        "domain phrases such as 'machine learning'. Bigrams "
        "provide additional context without creating as many "
        "features as trigrams.",
        "",
        "- min_df=1: with only 16 documents, requiring a word "
        "to appear in 2 or more documents could remove "
        "candidate-specific skills that are useful for "
        "distinguishing candidates.",
        "",
        "- max_df=0.95: removes terms appearing in almost every "
        "document because these terms provide less "
        "discriminative information.",
        "",
        "- sublinear_tf=True: reduces the influence of very "
        "frequent repeated terms by using logarithmic term "
        "frequency scaling.",
        "",
        "- No max_features cap: the corpus is small, so limiting "
        "the vocabulary is unnecessary. The final TF-IDF "
        "configuration produces a manageable feature space.",
        "",
    ]

    output_path = os.path.join(
        RESULTS_DIR,
        "feature_engineering_analysis.txt",
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(lines))

    print(
        f"[FeatureEng] Saved -> {output_path}"
    )


if __name__ == "__main__":
    run()