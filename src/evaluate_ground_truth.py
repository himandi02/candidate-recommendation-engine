"""Evaluate recommendations using the dataset-provided ground-truth scores.

The ground-truth file contains one observed job assignment and matched score
for each candidate. For Top-K classification, observed pairs at or above the
chosen threshold are relevant and unlisted cross-job pairs are assumed not
relevant. Score agreement is calculated only for the 12 observed pairs.

Run from any directory:

    python src/evaluate_ground_truth.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GROUND_TRUTH = PROJECT_ROOT / "data/ground_truth.csv"
DEFAULT_RANKINGS = PROJECT_ROOT / "results/candidate_rankings.csv"
DEFAULT_SIMILARITIES = PROJECT_ROOT / "results/similarity_scores.csv"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

GROUND_TRUTH_COLUMNS = {
    "candidate_id",
    "job_id",
    "job_position_name",
    "matched_score",
}
RANKING_COLUMNS = {"Job", "Rank", "Candidate", "Similarity Score"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate rankings using dataset ground-truth scores."
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH,
        help="CSV containing observed candidate-job matched scores.",
    )
    parser.add_argument(
        "--rankings",
        type=Path,
        default=DEFAULT_RANKINGS,
        help="CSV containing complete candidate rankings for every job.",
    )
    parser.add_argument(
        "--similarities",
        type=Path,
        default=DEFAULT_SIMILARITIES,
        help="CSV matrix containing full-precision cosine similarities.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for generated evaluation artifacts.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Minimum matched_score considered relevant (default: 0.50).",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[3, 5],
        help="Recommendation cutoffs to evaluate (default: 3 5).",
    )
    return parser.parse_args()


def require_columns(
    dataframe: pd.DataFrame,
    required: set[str],
    source_name: str,
) -> None:
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {sorted(missing)}"
        )


def load_inputs(
    ground_truth_path: Path,
    rankings_path: Path,
    similarities_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ground_truth = pd.read_csv(ground_truth_path)
    rankings = pd.read_csv(rankings_path)
    similarity_matrix = pd.read_csv(similarities_path, index_col=0)

    require_columns(ground_truth, GROUND_TRUTH_COLUMNS, "Ground truth")
    require_columns(rankings, RANKING_COLUMNS, "Rankings")

    ground_truth = ground_truth.copy()
    rankings = rankings.copy()
    ground_truth["matched_score"] = pd.to_numeric(
        ground_truth["matched_score"], errors="raise"
    )
    rankings["Rank"] = pd.to_numeric(
        rankings["Rank"], errors="raise"
    ).astype(int)
    rankings["Similarity Score"] = pd.to_numeric(
        rankings["Similarity Score"], errors="raise"
    )
    similarity_matrix = similarity_matrix.apply(
        pd.to_numeric, errors="raise"
    )

    return ground_truth, rankings, similarity_matrix


def validate_inputs(
    ground_truth: pd.DataFrame,
    rankings: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    threshold: float,
    cutoffs: list[int],
) -> None:
    if not 0 <= threshold <= 1:
        raise ValueError("The relevance threshold must be between 0 and 1.")
    if not cutoffs or any(cutoff <= 0 for cutoff in cutoffs):
        raise ValueError("Every evaluation cutoff must be positive.")
    if ground_truth.empty or rankings.empty or similarity_matrix.empty:
        raise ValueError("Evaluation input files must not be empty.")
    if ground_truth[list(GROUND_TRUTH_COLUMNS)].isna().any().any():
        raise ValueError("Ground truth contains missing required values.")
    if rankings[list(RANKING_COLUMNS)].isna().any().any():
        raise ValueError("Rankings contain missing required values.")
    if not ground_truth["matched_score"].between(0, 1).all():
        raise ValueError("Ground-truth matched scores must be between 0 and 1.")

    ground_pairs = ground_truth[["job_id", "candidate_id"]]
    ranking_pairs = rankings[["Job", "Candidate"]].rename(
        columns={"Job": "job_id", "Candidate": "candidate_id"}
    )
    if ground_pairs.duplicated().any():
        raise ValueError("Ground truth contains duplicate job-candidate pairs.")
    if ranking_pairs.duplicated().any():
        raise ValueError("Rankings contain duplicate job-candidate pairs.")

    ranking_pair_set = set(
        ranking_pairs.itertuples(index=False, name=None)
    )
    missing_ground_pairs = [
        pair
        for pair in ground_pairs.itertuples(index=False, name=None)
        if pair not in ranking_pair_set
    ]
    if missing_ground_pairs:
        raise ValueError(
            "Ground-truth pairs missing from rankings: "
            f"{missing_ground_pairs[:5]}"
        )

    title_counts = ground_truth.groupby("job_id")[
        "job_position_name"
    ].nunique()
    if (title_counts != 1).any():
        raise ValueError("Each job ID must map to exactly one job title.")

    candidate_sets = rankings.groupby("Job")["Candidate"].apply(set)
    first_candidate_set = candidate_sets.iloc[0]
    if not all(
        candidate_set == first_candidate_set
        for candidate_set in candidate_sets.iloc[1:]
    ):
        raise ValueError("Every job must rank the same candidate set.")

    for job_id, job_rankings in rankings.groupby("Job"):
        ordered = job_rankings.sort_values("Rank")
        if ordered["Rank"].tolist() != list(range(1, len(ordered) + 1)):
            raise ValueError(
                f"Ranks for {job_id} must be consecutive and start at 1."
            )
        if not ordered["Similarity Score"].is_monotonic_decreasing:
            raise ValueError(
                f"Similarity scores for {job_id} are not descending."
            )
        if max(cutoffs) > len(ordered):
            raise ValueError(
                f"Cutoff {max(cutoffs)} exceeds candidates for {job_id}."
            )

    if set(rankings["Job"]) != set(similarity_matrix.index):
        raise ValueError("Similarity-matrix job IDs do not match rankings.")
    if first_candidate_set != set(similarity_matrix.columns):
        raise ValueError(
            "Similarity-matrix candidate IDs do not match rankings."
        )


def similarity_long_form(similarity_matrix: pd.DataFrame) -> pd.DataFrame:
    return (
        similarity_matrix.rename_axis("job_id")
        .reset_index()
        .melt(
            id_vars="job_id",
            var_name="candidate_id",
            value_name="cosine_similarity",
        )
    )


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_agreement(
    ground_truth: pd.DataFrame,
    rankings: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_scores = similarity_long_form(similarity_matrix)
    comparison = ground_truth.merge(
        long_scores,
        on=["job_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        rankings[["Job", "Candidate", "Rank"]],
        left_on=["job_id", "candidate_id"],
        right_on=["Job", "Candidate"],
        how="left",
        validate="one_to_one",
    )
    comparison["relevant_at_threshold"] = (
        comparison["matched_score"] >= threshold
    ).astype(int)
    comparison["score_error"] = (
        comparison["cosine_similarity"] - comparison["matched_score"]
    )
    comparison["absolute_error"] = comparison["score_error"].abs()
    comparison["squared_error"] = comparison["score_error"] ** 2
    comparison = comparison.drop(columns=["Job", "Candidate"]).sort_values(
        ["job_id", "candidate_id"]
    )

    actual = comparison["matched_score"]
    predicted = comparison["cosine_similarity"]
    errors = predicted - actual
    mse = float(np.mean(errors**2))
    actual_variation = float(np.sum((actual - actual.mean()) ** 2))
    r_squared = (
        1 - float(np.sum(errors**2)) / actual_variation
        if actual_variation > 0
        else float("nan")
    )
    pearson = float(actual.corr(predicted, method="pearson"))
    spearman = float(
        actual.rank(method="average").corr(
            predicted.rank(method="average"), method="pearson"
        )
    )

    score_metrics = pd.DataFrame(
        [
            {
                "Observed Pairs": len(comparison),
                "Relevance Threshold": threshold,
                "Relevant Observed Pairs": int(
                    comparison["relevant_at_threshold"].sum()
                ),
                "Ground Truth Mean": float(actual.mean()),
                "Cosine Similarity Mean": float(predicted.mean()),
                "Mean Error": float(errors.mean()),
                "MAE": float(np.mean(np.abs(errors))),
                "MSE": mse,
                "RMSE": float(np.sqrt(mse)),
                "Pearson Correlation": pearson,
                "Spearman Correlation": spearman,
                "R Squared": r_squared,
            }
        ]
    )
    return comparison, score_metrics


def build_label_universe(
    ground_truth: pd.DataFrame,
    rankings: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    job_titles = ground_truth[["job_id", "job_position_name"]].drop_duplicates()
    universe = rankings.rename(
        columns={"Job": "job_id", "Candidate": "candidate_id"}
    ).merge(
        ground_truth,
        on=["job_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    ).merge(job_titles, on="job_id", how="left", suffixes=("", "_job"))

    universe["ground_truth_available"] = universe["matched_score"].notna().astype(
        int
    )
    universe["relevant"] = (
        universe["matched_score"].fillna(-1) >= threshold
    ).astype(int)
    universe["job_position_name"] = universe["job_position_name"].fillna(
        universe["job_position_name_job"]
    )
    universe = universe.drop(columns=["job_position_name_job"])
    return universe.sort_values(["job_id", "Rank"])


def calculate_topk_metrics(
    label_universe: pd.DataFrame,
    cutoffs: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []

    for job_id, job_rows in label_universe.groupby("job_id"):
        job_rows = job_rows.sort_values("Rank")
        relevant_ids = set(
            job_rows.loc[job_rows["relevant"] == 1, "candidate_id"]
        )
        all_candidates = set(job_rows["candidate_id"])
        job_title = str(job_rows["job_position_name"].iloc[0])

        for cutoff in cutoffs:
            recommended_ids = set(job_rows.head(cutoff)["candidate_id"])
            true_positive_ids = relevant_ids.intersection(recommended_ids)
            false_positive_ids = recommended_ids.difference(relevant_ids)
            false_negative_ids = relevant_ids.difference(recommended_ids)
            true_negative_ids = all_candidates.difference(
                recommended_ids.union(relevant_ids)
            )

            true_positives = len(true_positive_ids)
            false_positives = len(false_positive_ids)
            false_negatives = len(false_negative_ids)
            true_negatives = len(true_negative_ids)
            precision = true_positives / cutoff
            recall = true_positives / len(relevant_ids)
            accuracy = (true_positives + true_negatives) / len(all_candidates)

            rows.append(
                {
                    "Job ID": job_id,
                    "Job Title": job_title,
                    "K": cutoff,
                    "Candidates": len(all_candidates),
                    "Relevant Candidates": len(relevant_ids),
                    "True Positives": true_positives,
                    "False Positives": false_positives,
                    "False Negatives": false_negatives,
                    "True Negatives": true_negatives,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1_score(precision, recall),
                    "Relevant Recommended IDs": ";".join(
                        sorted(true_positive_ids)
                    ),
                    "Missed Relevant IDs": ";".join(
                        sorted(false_negative_ids)
                    ),
                }
            )

    metrics = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []

    for cutoff in cutoffs:
        selected = metrics[metrics["K"] == cutoff]
        tp = int(selected["True Positives"].sum())
        fp = int(selected["False Positives"].sum())
        fn = int(selected["False Negatives"].sum())
        tn = int(selected["True Negatives"].sum())
        micro_precision = tp / (tp + fp) if tp + fp else 0.0
        micro_recall = tp / (tp + fn) if tp + fn else 0.0

        summary_rows.append(
            {
                "K": cutoff,
                "Jobs": len(selected),
                "Total Pairs": tp + fp + fn + tn,
                "Total Relevant": tp + fn,
                "Total True Positives": tp,
                "Total False Positives": fp,
                "Total False Negatives": fn,
                "Total True Negatives": tn,
                "Macro Accuracy": selected["Accuracy"].mean(),
                "Macro Precision": selected["Precision"].mean(),
                "Macro Recall": selected["Recall"].mean(),
                "Macro F1": selected["F1 Score"].mean(),
                "Micro Accuracy": (tp + tn) / (tp + fp + fn + tn),
                "Micro Precision": micro_precision,
                "Micro Recall": micro_recall,
                "Micro F1": f1_score(micro_precision, micro_recall),
            }
        )

    return metrics, pd.DataFrame(summary_rows)


def format_ids(value: object) -> str:
    text = str(value).strip()
    return text if text else "None"


def create_report(
    ground_truth: pd.DataFrame,
    label_universe: pd.DataFrame,
    score_metrics: pd.DataFrame,
    topk_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    threshold: float,
    cutoffs: list[int],
) -> str:
    scores = score_metrics.iloc[0]
    unobserved = int((label_universe["ground_truth_available"] == 0).sum())
    lines = [
        "Dataset Ground-Truth Evaluation",
        "=" * 35,
        "Status: FINAL",
        "",
        "Evaluation setup",
        "----------------",
        "Recommendation method: TF-IDF cosine similarity",
        f"Observed ground-truth pairs: {len(ground_truth)}",
        f"All ranked pairs: {len(label_universe)}",
        f"Unlisted pairs assumed non-relevant: {unobserved}",
        f"Relevance threshold: matched_score >= {threshold:.2f}",
        f"Relevant pairs at threshold: {int(label_universe['relevant'].sum())}",
        f"Cutoffs: {', '.join(f'Top-{cutoff}' for cutoff in cutoffs)}",
        "",
        "Score agreement on the 12 observed pairs",
        "----------------------------------------",
        f"Ground-truth mean: {scores['Ground Truth Mean']:.4f}",
        f"Cosine-similarity mean: {scores['Cosine Similarity Mean']:.4f}",
        f"MAE: {scores['MAE']:.4f}",
        f"RMSE: {scores['RMSE']:.4f}",
        f"Pearson correlation: {scores['Pearson Correlation']:.4f}",
        f"Spearman correlation: {scores['Spearman Correlation']:.4f}",
        f"R squared: {scores['R Squared']:.4f}",
    ]

    for cutoff in cutoffs:
        lines.extend(["", f"Top-{cutoff} classification results", "-" * 36])
        selected = topk_metrics[topk_metrics["K"] == cutoff]
        for _, row in selected.iterrows():
            lines.append(
                f"{row['Job ID']} ({row['Job Title']}): "
                f"Accuracy={row['Accuracy']:.4f}, "
                f"Precision={row['Precision']:.4f}, "
                f"Recall={row['Recall']:.4f}, "
                f"F1={row['F1 Score']:.4f}"
            )
            lines.append(
                "  Relevant recommended: "
                f"{format_ids(row['Relevant Recommended IDs'])}"
            )
            lines.append(
                f"  Missed relevant: {format_ids(row['Missed Relevant IDs'])}"
            )

        overall = summary[summary["K"] == cutoff].iloc[0]
        lines.extend(
            [
                "",
                f"Macro: Accuracy={overall['Macro Accuracy']:.4f}, "
                f"Precision={overall['Macro Precision']:.4f}, "
                f"Recall={overall['Macro Recall']:.4f}, "
                f"F1={overall['Macro F1']:.4f}",
                f"Micro: Accuracy={overall['Micro Accuracy']:.4f}, "
                f"Precision={overall['Micro Precision']:.4f}, "
                f"Recall={overall['Micro Recall']:.4f}, "
                f"F1={overall['Micro F1']:.4f}",
            ]
        )

    lines.extend(
        [
            "",
            "Interpretation",
            "--------------",
            "Accuracy is influenced by the many non-relevant pairs and should "
            "not be interpreted without Precision Recall and F1.",
            "The cosine similarities are much lower than the dataset matched "
            "scores and have weak agreement with their ordering.",
            "The low agreement is consistent with sparse exact-term TF-IDF "
            "matching and limited overlap between CV and job text.",
            "",
            "Required assumptions and limitations",
            "------------------------------------",
            "- ground_truth.csv provides scores for only 12 of the 48 pairs.",
            "- The other 36 cross-job pairs are assumed non-relevant for the "
            "Top-K classification metrics because no matched score is given.",
            f"- A {threshold:.2f} relevance threshold is used because the CSV "
            "contains continuous scores rather than binary labels.",
            "- MAE RMSE and correlation use only the 12 pairs with observed "
            "matched scores.",
            "- Cosine similarity is not calibrated to reproduce the dataset's "
            "matched-score scale so score error measures should be interpreted "
            "as agreement diagnostics rather than prediction error from a "
            "trained regression model.",
        ]
    )
    return "\n".join(lines) + "\n"


def rounded_copy(dataframe: pd.DataFrame) -> pd.DataFrame:
    rounded = dataframe.copy()
    float_columns = rounded.select_dtypes(include="float").columns
    rounded[float_columns] = rounded[float_columns].round(4)
    return rounded


def run(
    ground_truth_path: Path,
    rankings_path: Path,
    similarities_path: Path,
    results_dir: Path,
    threshold: float,
    cutoffs: list[int],
) -> None:
    cutoffs = sorted(set(cutoffs))
    ground_truth, rankings, similarity_matrix = load_inputs(
        ground_truth_path, rankings_path, similarities_path
    )
    validate_inputs(
        ground_truth,
        rankings,
        similarity_matrix,
        threshold,
        cutoffs,
    )
    pair_comparison, score_metrics = score_agreement(
        ground_truth, rankings, similarity_matrix, threshold
    )
    label_universe = build_label_universe(
        ground_truth, rankings, threshold
    )
    topk_metrics, summary = calculate_topk_metrics(label_universe, cutoffs)

    results_dir.mkdir(parents=True, exist_ok=True)
    rounded_copy(pair_comparison).to_csv(
        results_dir / "ground_truth_pair_comparison.csv", index=False
    )
    rounded_copy(label_universe).to_csv(
        results_dir / "ground_truth_labeled_rankings.csv", index=False
    )
    rounded_copy(score_metrics).to_csv(
        results_dir / "ground_truth_score_metrics.csv", index=False
    )
    rounded_copy(topk_metrics).to_csv(
        results_dir / "ground_truth_topk_metrics.csv", index=False
    )
    rounded_copy(summary).to_csv(
        results_dir / "ground_truth_evaluation_summary.csv", index=False
    )
    report = create_report(
        ground_truth,
        label_universe,
        score_metrics,
        topk_metrics,
        summary,
        threshold,
        cutoffs,
    )
    (results_dir / "ground_truth_evaluation_analysis.txt").write_text(
        report, encoding="utf-8"
    )

    print("Ground-truth evaluation completed successfully.")
    print(f"Observed score pairs: {len(pair_comparison)}")
    print(f"Ranked classification pairs: {len(label_universe)}")
    print(f"Relevance threshold: {threshold:.2f}")
    print("\nScore agreement:")
    print(rounded_copy(score_metrics).to_string(index=False))
    print("\nTop-K metrics:")
    print(rounded_copy(summary).to_string(index=False))


if __name__ == "__main__":
    arguments = parse_args()
    run(
        ground_truth_path=arguments.ground_truth,
        rankings_path=arguments.rankings,
        similarities_path=arguments.similarities,
        results_dir=arguments.results_dir,
        threshold=arguments.threshold,
        cutoffs=arguments.ks,
    )
