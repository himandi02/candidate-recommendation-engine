# Similarity Recommandation 

## Overview

This project implements a **Candidate Recommendation System** using **TF-IDF vectorization** and **Cosine Similarity**.

The system compares job descriptions with candidate profiles and calculates similarity scores to identify the most suitable candidates for each job.

Candidates are then ranked based on their similarity scores, and the **Top 5 candidate recommendations** are generated for every job.

---

## Objective

The main objective of this system is to automate the candidate matching process by:

* Loading preprocessed TF-IDF vectors for candidates and jobs.
* Calculating similarity between job requirements and candidate profiles.
* Ranking candidates based on similarity scores.
* Generating the Top 5 recommended candidates for each job.
* Saving all results for further analysis.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

---

## Project Structure

```text
project/
│
├── models/
│   ├── tfidf_candidate_vectors.pkl
│   └── tfidf_job_vectors.pkl
│
├── results/
│   ├── similarity_scores.csv
│   ├── candidate_rankings.csv
│   ├── top5_recommendations.csv
│   └── recommendations.txt
│
├── candidate_recommendation.py
│
├── requirements.txt
│
└── README.md
```

---

# How the System Works

## 1. Load TF-IDF Vectors

The system loads previously generated TF-IDF vectors for both candidates and job descriptions.

```python
candidate_data = joblib.load(
    "models/tfidf_candidate_vectors.pkl"
)

job_data = joblib.load(
    "models/tfidf_job_vectors.pkl"
)
```

The loaded data contains:

* TF-IDF vectors
* Candidate IDs
* Job IDs

---

## 2. Calculate Cosine Similarity

Cosine Similarity is used to measure how similar each candidate profile is to a job description.

```python
similarity_matrix = cosine_similarity(
    job_vectors,
    candidate_vectors
)
```

The resulting matrix contains similarity scores between each job and every candidate.

A higher similarity score indicates that the candidate is more relevant to the job requirements.

---

## 3. Create Similarity Matrix

The similarity scores are stored in a Pandas DataFrame.

```python
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=job_ids,
    columns=candidate_ids
)
```

Example:

| Job   | Candidate 1 | Candidate 2 | Candidate 3 |
| ----- | ----------: | ----------: | ----------: |
| Job 1 |        0.82 |        0.45 |        0.67 |
| Job 2 |        0.54 |        0.91 |        0.38 |

The similarity matrix is saved as:

```text
results/similarity_scores.csv
```

---

## 4. Rank Candidates

For each job, candidate similarity scores are sorted in descending order.

```python
ranking_indices = np.argsort(
    scores
)[::-1]
```

The candidate with the highest similarity score receives **Rank 1**.

Example:

| Job   | Rank | Candidate   | Similarity Score |
| ----- | ---: | ----------- | ---------------: |
| Job 1 |    1 | Candidate A |           0.9234 |
| Job 1 |    2 | Candidate B |           0.8756 |
| Job 1 |    3 | Candidate C |           0.8123 |

The complete ranking is saved as:

```text
results/candidate_rankings.csv
```

---

## 5. Generate Top-5 Recommendations

The system filters the top five candidates for each job.

```python
top_recommendations = ranking_df[
    ranking_df["Rank"] <= 5
]
```

The recommendations are saved as:

```text
results/top5_recommendations.csv
```

---

## 6. Generate Text Recommendation Report

A readable recommendation report is also generated.

Example output:

```text
============================================================
JOB: Job_001
============================================================

Rank 1: Candidate_015 - Score: 0.9234
Rank 2: Candidate_008 - Score: 0.8756
Rank 3: Candidate_021 - Score: 0.8123
Rank 4: Candidate_004 - Score: 0.7564
Rank 5: Candidate_012 - Score: 0.7012
```

The report is saved as:

```text
results/recommendations.txt
```

---

# Output Files

After successfully running the recommendation process, the following files are generated:

| File                       | Description                                              |
| -------------------------- | -------------------------------------------------------- |
| `similarity_scores.csv`    | Contains similarity scores between jobs and candidates   |
| `candidate_rankings.csv`   | Contains the complete ranking of candidates for each job |
| `top5_recommendations.csv` | Contains the Top 5 recommended candidates for each job   |
| `recommendations.txt`      | Contains a readable text-based recommendation report     |

---

# Installation

Clone the repository:

```bash
git clone <your-repository-url>
```

Navigate to the project directory:

```bash
cd <project-folder>
```

Install the required libraries:

```bash
pip install pandas numpy scikit-learn joblib
```

---

# How to Run

Make sure the following files are available:

```text
models/tfidf_candidate_vectors.pkl
models/tfidf_job_vectors.pkl
```

Then run:

```bash
python candidate_recommendation.py
```

After execution, the generated results will be available in the `results/` folder.

---

# Recommendation Process

```text
Candidate Profiles
        │
        ▼
TF-IDF Vectorization
        │
        ▼
Candidate TF-IDF Vectors
        │
        │
        ├───────────────┐
        │               │
        ▼               ▼
Job TF-IDF Vectors   Candidate Vectors
        │               │
        └───────┬───────┘
                │
                ▼
        Cosine Similarity
                │
                ▼
        Similarity Scores
                │
                ▼
        Candidate Ranking
                │
                ▼
     Top-5 Recommendations
```

---

# Conclusion

This Candidate Recommendation System uses **TF-IDF** to represent candidate profiles and job descriptions as numerical vectors. **Cosine Similarity** is then used to measure the similarity between jobs and candidates.

Based on the calculated similarity scores, candidates are ranked from the most suitable to the least suitable candidate. The system finally generates the **Top 5 candidate recommendations for each job**, helping automate and improve the recruitment and candidate selection process.

