# Vectorization & Feature Engineering - Methodology

## Scope of this module

This covers exactly these tasks from the project's **Vectorization & Feature Engineering** section:

- Select a vectorization model (TF-IDF required; Word2Vec / BERT optional)

- Convert cleaned CVs into numerical vectors

- Convert cleaned job descriptions into numerical vectors in the same feature space as the CVs

- Compare vectorization methods (optional)

**Out of scope:** cosine similarity, candidate ranking, and top-N recommendation. These are handled by the Similarity Computation & Recommendation component. This module produces the vector files required by the next stage.

## Method 1: TF-IDF

`src/vectorize_tfidf.py` fits a single `TfidfVectorizer` across the combined candidate + job corpus, then splits the resulting matrix back into candidate vectors and job vectors.

### Why use one shared vectorizer?

The candidate CVs and job descriptions must use the same feature space so their vectors can be compared later.

If separate vectorizers were fitted to CVs and jobs, the same word could be assigned to different column positions. Therefore, the resulting vectors would not be directly comparable.

Fitting one vectorizer on the combined corpus guarantees that both candidate and job vectors use the same vocabulary and column positions.

### TF-IDF feature engineering choices

The final vectorizer uses:

```python
TfidfVectorizer(
    min_df=1,
    max_df=0.95,
    ngram_range=(1, 2),
    sublinear_tf=True,
)
```

- `min_df=1` keeps terms that appear in at least one document. This is appropriate for the small dataset because candidate-specific skills may occur only once.

- `max_df=0.95` removes terms that occur in more than 95% of documents because they provide little distinguishing information.

- `ngram_range=(1, 2)` includes both individual words and two-word phrases. This helps capture phrases such as `machine learning`, `data analysis`, and `human resources`.

- `sublinear_tf=True` applies logarithmic term-frequency scaling, reducing the influence of terms that are repeated many times within a document.

The final TF-IDF vectorization produced **955 features** from the project's 16 documents (12 candidate CVs and 4 job descriptions).

### TF-IDF outputs

Running:

```text
python src/vectorize_tfidf.py
```

produces:

- `models/tfidf_vectorizer.pkl`

- `models/tfidf_candidate_vectors.pkl`

- `models/tfidf_job_vectors.pkl`

- `results/tfidf_summary.txt`

The candidate matrix has shape:

```text
(12, 955)
```

The job matrix has shape:

```text
(4, 955)
```

The 955 columns are shared between both matrices, which allows the next stage to calculate candidate-to-job similarity.

## Method 2: Word2Vec

Word2Vec was implemented as an optional comparison method using:

`src/vectorize_word2vec.py`

The model is trained on the combined candidate and job corpus using the Skip-gram architecture.

The configuration is:

```python
Word2Vec(
    sentences=all_tokenized,
    vector_size=100,
    window=5,
    min_count=1,
    epochs=100,
    sg=1,
    seed=42,
)
```

Each document is represented by the mean of the Word2Vec vectors of its words.

### Why include Word2Vec?

Word2Vec can represent words as dense numerical vectors and can potentially capture relationships between semantically related words.

However, this project contains only **16 documents**, so there is very little training data available for learning reliable word relationships.

Therefore, Word2Vec is included as a comparison method rather than being selected as the primary representation.

The Word2Vec results were:

```text
Candidate matrix: (12, 100)
Job matrix: (4, 100)
Vocabulary: 374 words
```

### Word2Vec outputs

Running:

```text
python src/vectorize_word2vec.py
```

produces:

- `models/word2vec_model.bin`

- `models/word2vec_candidate_vectors.pkl`

- `models/word2vec_job_vectors.pkl`

## Method 3: Sentence-BERT

Sentence-BERT was **not implemented** because it is optional in the project brief.

The project brief allows the use of:

- TF-IDF

- Word2Vec

- BERT / Sentence Transformers

The required vectorization method is satisfied by TF-IDF. Word2Vec was additionally implemented to provide a comparison between a traditional sparse representation and a learned dense representation.

Therefore, Sentence-BERT is not required for this project.

## Vectorization Method Comparison

The script:

`src/compare_vectorizations.py`

compares the characteristics of the implemented vectorization methods. It does not calculate similarity scores or rank candidates.

The comparison includes:

- Vector dimensionality

- Sparse or dense representation

- Vector density

- Example vector values

The current results are:

| Method   | Dimensions | Representation | Dataset limitation                        |
| -------- | ---------- | -------------- | ----------------------------------------- |
| TF-IDF   | 955        | Sparse         | Works well with a small corpus            |
| Word2Vec | 100        | Dense          | Limited by the very small training corpus |

### TF-IDF comparison result

```text
Candidates: 12
Jobs: 4
Vector dimensions: 955
Representation: sparse
Density: 0.080
```

TF-IDF is highly interpretable because each feature corresponds to a word or phrase from the corpus.

### Word2Vec comparison result

```text
Candidates: 12
Jobs: 4
Vector dimensions: 100
Representation: dense
Density: 1.000
```

Word2Vec produces compact dense vectors, but its learned word representations are less reliable because the model was trained using only 16 short documents.

## Feature Engineering Analysis

The script:

`src/feature_engineering_analysis.py`

tests different TF-IDF configurations using the same dataset.

The tested configurations include:

- Unigrams only

- Unigrams + bigrams

- Unigrams + bigrams + trigrams

- `min_df=2`

- `max_features=200`

The results demonstrate why the final configuration was selected.

### Unigrams only

```text
Vocabulary size: 371
```

Using only individual words produces a smaller feature space but does not explicitly represent useful multi-word phrases.

### Unigrams + bigrams

```text
Vocabulary size: 955
```

This configuration provides additional contextual features such as:

```text
account management
account payable
computer science
```

It provides more information than unigrams without increasing the feature space as much as adding trigrams.

### Unigrams + bigrams + trigrams

```text
Vocabulary size: 1571
```

Trigrams increase the feature space considerably. Given the small dataset, the additional features are not necessary for the current system.

### `min_df=2`

```text
Vocabulary size: 117
```

This removes terms that appear in only one document. For a candidate recommendation system, this could remove important candidate-specific skills, so `min_df=1` is preferable for this dataset.

### `max_features=200`

```text
Vocabulary size: 200
```

Limiting the vocabulary reduces dimensionality, but the current dataset is already small and the full 955-feature representation is manageable.

### Final feature engineering decision

The final TF-IDF configuration was selected as:

```python
TfidfVectorizer(
    min_df=1,
    max_df=0.95,
    ngram_range=(1, 2),
    sublinear_tf=True,
)
```

This configuration provides a reasonable balance between:

- preserving useful candidate-specific terms

- capturing meaningful two-word phrases

- reducing the influence of extremely common terms

- reducing the effect of repeated terms

- keeping the feature space manageable

The analysis is saved to:

```text
results/feature_engineering_analysis.txt
```

## Handoff to Similarity Computation & Recommendation

The vectorization module provides the TF-IDF vectors required by the next project component.

The next teammate can load the saved vectors using:

```python
import joblib

candidate_data = joblib.load(
    "models/tfidf_candidate_vectors.pkl"
)

job_data = joblib.load(
    "models/tfidf_job_vectors.pkl"
)

candidate_vectors = candidate_data["vectors"]
job_vectors = job_data["vectors"]

candidate_ids = candidate_data["ids"]
job_ids = job_data["ids"]
```

The resulting matrices are:

```text
Candidate vectors: (12, 955)
Job vectors:       (4, 955)
```

Because both matrices were created using the same fitted TF-IDF vectorizer, they share the same 955-dimensional feature space.

The Similarity Computation & Recommendation component can therefore use these vectors to calculate cosine similarity, rank candidates for each job, and produce the required top-3 or top-5 recommendations.

## Files Created by This Module

```text
src/
├── vectorize_tfidf.py
├── vectorize_word2vec.py
├── compare_vectorizations.py
└── feature_engineering_analysis.py

models/
├── tfidf_vectorizer.pkl
├── tfidf_candidate_vectors.pkl
├── tfidf_job_vectors.pkl
├── word2vec_model.bin
├── word2vec_candidate_vectors.pkl
└── word2vec_job_vectors.pkl

results/
├── tfidf_summary.txt
├── vectorization_comparison.txt
└── feature_engineering_analysis.txt

docs/
└── vectorization_methodology.md
```

## Contribution

This documentation covers the Vectorization & Feature Engineering component of the project.

## Conclusion

The required vectorization work has been completed using **TF-IDF**. Candidate CVs and job descriptions are converted into numerical vectors within a shared 955-dimensional feature space.

**Word2Vec** was additionally implemented as an optional comparison method. **Sentence-BERT was not implemented because it is optional and is not required to satisfy the project brief.**

The vectorization module is therefore ready to hand off to the **Similarity Computation & Recommendation** component.
