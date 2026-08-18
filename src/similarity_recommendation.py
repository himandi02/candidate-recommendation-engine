import joblib
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity



# Load TF-IDF vectors

candidate_data = joblib.load(
    "models/tfidf_candidate_vectors.pkl"
)

job_data = joblib.load(
    "models/tfidf_job_vectors.pkl"
)


candidate_vectors = candidate_data["vectors"]
candidate_ids = candidate_data["ids"]

job_vectors = job_data["vectors"]
job_ids = job_data["ids"]


# Display vector information


print("Candidate vectors shape:",
      candidate_vectors.shape)

print("Job vectors shape:",
      job_vectors.shape)

print("\nCandidates:")
print(candidate_ids)

print("\nJobs:")
print(job_ids)


# Calculate Cosine Similarity


similarity_matrix = cosine_similarity(
    job_vectors,
    candidate_vectors
)


# Create Similarity DataFrame


similarity_df = pd.DataFrame(
    similarity_matrix,
    index=job_ids,
    columns=candidate_ids
)

print("\nSimilarity Matrix:")
print(similarity_df)


# Save similarity scores


similarity_df.to_csv(
    "results/similarity_scores.csv"
)

# Rank Candidates

all_rankings = []

for i, job_id in enumerate(job_ids):

    scores = similarity_matrix[i]

    ranking_indices = np.argsort(
        scores
    )[::-1]

    for rank, candidate_index in enumerate(
        ranking_indices,
        start=1
    ):

        all_rankings.append({
            "Job": job_id,
            "Rank": rank,
            "Candidate": candidate_ids[
                candidate_index
            ],
            "Similarity Score": round(
                scores[candidate_index],
                4
            )
        })


# Create Ranking DataFrame


ranking_df = pd.DataFrame(
    all_rankings
)

print("\nCandidate Rankings:")
print(ranking_df)


# Save Complete Rankings

ranking_df.to_csv(
    "results/candidate_rankings.csv",
    index=False
)


# Top-5 Recommendations

top_recommendations = ranking_df[
    ranking_df["Rank"] <= 5
]

top_recommendations.to_csv(
    "results/top5_recommendations.csv",
    index=False
)


# Generate Recommendation Report


with open(
    "results/recommendations.txt",
    "w",
    encoding="utf-8"
) as file:

    for job_id in job_ids:

        file.write("\n")
        file.write("=" * 60 + "\n")
        file.write(
            f"JOB: {job_id}\n"
        )
        file.write("=" * 60 + "\n")

        job_results = ranking_df[
            (ranking_df["Job"] == job_id) &
            (ranking_df["Rank"] <= 5)
        ]

        for _, row in job_results.iterrows():

            file.write(
                f"Rank {row['Rank']}: "
                f"{row['Candidate']} - "
                f"Score: "
                f"{row['Similarity Score']}\n"
            )


print(
    "\nRecommendation process completed successfully!"
)

print(
    "Results saved in the results/ folder."
)