"""Evaluate ranked candidate recommendations against the approved answer key.

The script calculates Precision@K, Recall@K, and F1@K for each job and
produces macro- and micro-averaged summaries. Run it from any directory:

    python src/evaluate_recommendations.py

By default, K is evaluated at 3 and 5.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANSWER_KEY = PROJECT_ROOT / "data/evaluation_answer_key.csv"
DEFAULT_RANKINGS = PROJECT_ROOT / "results/candidate_rankings.csv"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

ANSWER_KEY_COLUMNS = {
    "Job ID",
    "Job Title",
    "Candidate ID",
    "Relevant",
    "Skills Match",
    "Experience Match",
    "Education Match",
    "Reason",
}

RANKING_COLUMNS = {
    "Job",
    "Rank",
    "Candidate",
    "Similarity Score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate candidate rankings against a binary answer key."
    )
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=DEFAULT_ANSWER_KEY,
        help="CSV containing one relevance label per job-candidate pair.",
    )
    parser.add_argument(
        "--rankings",
        type=Path,
        default=DEFAULT_RANKINGS,
        help="CSV containing complete ranked candidates for each job.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for generated evaluation artifacts.",
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
    answer_key_path: Path,
    rankings_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    answer_key = pd.read_csv(answer_key_path)
    rankings = pd.read_csv(rankings_path)

    require_columns(answer_key, ANSWER_KEY_COLUMNS, "Answer key")
    require_columns(rankings, RANKING_COLUMNS, "Rankings")

    answer_key = answer_key.copy()
    rankings = rankings.copy()

    answer_key["Relevant"] = pd.to_numeric(
        answer_key["Relevant"], errors="raise"
    ).astype(int)
    rankings["Rank"] = pd.to_numeric(
        rankings["Rank"], errors="raise"
    ).astype(int)
    rankings["Similarity Score"] = pd.to_numeric(
        rankings["Similarity Score"], errors="raise"
    )

    return answer_key, rankings


def validate_inputs(
    answer_key: pd.DataFrame,
    rankings: pd.DataFrame,
    cutoffs: list[int],
) -> None:
    if answer_key.empty or rankings.empty:
        raise ValueError("The answer key and rankings must not be empty.")

    if any(cutoff <= 0 for cutoff in cutoffs):
        raise ValueError("Every evaluation cutoff must be a positive integer.")

    null_answer_columns = answer_key[list(ANSWER_KEY_COLUMNS)].columns[
        answer_key[list(ANSWER_KEY_COLUMNS)].isna().any()
    ].tolist()
    if null_answer_columns:
        raise ValueError(
            "Answer key contains missing values in: "
            f"{null_answer_columns}"
        )

    null_ranking_columns = rankings[list(RANKING_COLUMNS)].columns[
        rankings[list(RANKING_COLUMNS)].isna().any()
    ].tolist()
    if null_ranking_columns:
        raise ValueError(
            "Rankings contain missing values in: "
            f"{null_ranking_columns}"
        )

    if not set(answer_key["Relevant"]).issubset({0, 1}):
        raise ValueError("Relevant must contain binary values only: 0 or 1.")

    allowed_match_levels = {"Low", "Medium", "High"}
    for column in ("Skills Match", "Experience Match"):
        invalid = set(answer_key[column]).difference(allowed_match_levels)
        if invalid:
            raise ValueError(
                f"{column} contains invalid values: {sorted(invalid)}"
            )

    invalid_education = set(answer_key["Education Match"]).difference(
        {"No", "Partial", "Yes"}
    )
    if invalid_education:
        raise ValueError(
            "Education Match contains invalid values: "
            f"{sorted(invalid_education)}"
        )

    answer_pairs = answer_key[["Job ID", "Candidate ID"]]
    ranking_pairs = rankings[["Job", "Candidate"]].rename(
        columns={"Job": "Job ID", "Candidate": "Candidate ID"}
    )

    if answer_pairs.duplicated().any():
        raise ValueError("Answer key contains duplicate job-candidate pairs.")
    if ranking_pairs.duplicated().any():
        raise ValueError("Rankings contain duplicate job-candidate pairs.")

    answer_pair_set = set(answer_pairs.itertuples(index=False, name=None))
    ranking_pair_set = set(ranking_pairs.itertuples(index=False, name=None))
    if answer_pair_set != ranking_pair_set:
        missing = sorted(answer_pair_set.difference(ranking_pair_set))
        unexpected = sorted(ranking_pair_set.difference(answer_pair_set))
        raise ValueError(
            "Answer key and rankings do not cover identical pairs. "
            f"Missing from rankings: {missing[:5]}; "
            f"unexpected in rankings: {unexpected[:5]}"
        )

    title_counts = answer_key.groupby("Job ID")["Job Title"].nunique()
    if (title_counts != 1).any():
        raise ValueError("Each Job ID must map to exactly one Job Title.")

    candidates_by_job = answer_key.groupby("Job ID")["Candidate ID"].apply(
        set
    )
    first_candidate_set = candidates_by_job.iloc[0]
    if not all(
        candidate_set == first_candidate_set
        for candidate_set in candidates_by_job.iloc[1:]
    ):
        raise ValueError(
            "Every job must be evaluated against the same candidate set."
        )

    relevant_counts = answer_key.groupby("Job ID")["Relevant"].sum()
    jobs_without_relevant = relevant_counts[relevant_counts == 0].index.tolist()
    if jobs_without_relevant:
        raise ValueError(
            "Recall is undefined for jobs with no relevant candidates: "
            f"{jobs_without_relevant}"
        )

    for job_id, job_rankings in rankings.groupby("Job"):
        ordered = job_rankings.sort_values("Rank")
        expected_ranks = list(range(1, len(ordered) + 1))
        if ordered["Rank"].tolist() != expected_ranks:
            raise ValueError(
                f"Ranks for {job_id} must be consecutive and start at 1."
            )
        if not ordered["Similarity Score"].is_monotonic_decreasing:
            raise ValueError(
                f"Similarity scores for {job_id} are not in descending order."
            )
        if max(cutoffs) > len(ordered):
            raise ValueError(
                f"Cutoff {max(cutoffs)} exceeds the {len(ordered)} "
                f"ranked candidates for {job_id}."
            )


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(
    answer_key: pd.DataFrame,
    rankings: pd.DataFrame,
    cutoffs: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    renamed_rankings = rankings.rename(
        columns={"Job": "Job ID", "Candidate": "Candidate ID"}
    )
    evaluated_rankings = renamed_rankings.merge(
        answer_key,
        on=["Job ID", "Candidate ID"],
        how="left",
        validate="one_to_one",
    )

    for cutoff in cutoffs:
        evaluated_rankings[f"In Top {cutoff}"] = (
            evaluated_rankings["Rank"] <= cutoff
        ).astype(int)

    evaluated_rankings = evaluated_rankings[
        [
            "Job ID",
            "Job Title",
            "Rank",
            "Candidate ID",
            "Similarity Score",
            "Relevant",
            "Skills Match",
            "Experience Match",
            "Education Match",
            "Reason",
            *[f"In Top {cutoff}" for cutoff in cutoffs],
        ]
    ].sort_values(["Job ID", "Rank"])

    metric_rows: list[dict[str, object]] = []

    for job_id in sorted(answer_key["Job ID"].unique()):
        job_key = answer_key[answer_key["Job ID"] == job_id]
        job_rankings = evaluated_rankings[
            evaluated_rankings["Job ID"] == job_id
        ].sort_values("Rank")
        job_title = str(job_key["Job Title"].iloc[0])
        relevant_ids = set(
            job_key.loc[job_key["Relevant"] == 1, "Candidate ID"]
        )

        for cutoff in cutoffs:
            recommended_ids = set(
                job_rankings.head(cutoff)["Candidate ID"]
            )
            true_positive_ids = relevant_ids.intersection(recommended_ids)
            missed_ids = relevant_ids.difference(recommended_ids)

            true_positives = len(true_positive_ids)
            false_positives = cutoff - true_positives
            false_negatives = len(missed_ids)
            precision = true_positives / cutoff
            recall = true_positives / len(relevant_ids)

            metric_rows.append(
                {
                    "Job ID": job_id,
                    "Job Title": job_title,
                    "K": cutoff,
                    "Relevant Candidates": len(relevant_ids),
                    "Recommended Candidates": cutoff,
                    "True Positives": true_positives,
                    "False Positives": false_positives,
                    "False Negatives": false_negatives,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1_score(precision, recall),
                    "Relevant Recommended IDs": ";".join(
                        sorted(true_positive_ids)
                    ),
                    "Missed Relevant IDs": ";".join(sorted(missed_ids)),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    summary_rows: list[dict[str, object]] = []

    for cutoff in cutoffs:
        cutoff_metrics = metrics[metrics["K"] == cutoff]
        total_true_positives = int(cutoff_metrics["True Positives"].sum())
        total_false_positives = int(cutoff_metrics["False Positives"].sum())
        total_false_negatives = int(cutoff_metrics["False Negatives"].sum())
        total_relevant = int(cutoff_metrics["Relevant Candidates"].sum())
        total_recommended = int(
            cutoff_metrics["Recommended Candidates"].sum()
        )
        micro_precision = total_true_positives / total_recommended
        micro_recall = total_true_positives / total_relevant

        summary_rows.append(
            {
                "K": cutoff,
                "Jobs": len(cutoff_metrics),
                "Total Relevant": total_relevant,
                "Total Recommended": total_recommended,
                "Total True Positives": total_true_positives,
                "Total False Positives": total_false_positives,
                "Total False Negatives": total_false_negatives,
                "Macro Precision": cutoff_metrics["Precision"].mean(),
                "Macro Recall": cutoff_metrics["Recall"].mean(),
                "Macro F1": cutoff_metrics["F1 Score"].mean(),
                "Micro Precision": micro_precision,
                "Micro Recall": micro_recall,
                "Micro F1": f1_score(micro_precision, micro_recall),
            }
        )

    summary = pd.DataFrame(summary_rows)
    return evaluated_rankings, metrics, summary


def format_ids(value: object) -> str:
    text = str(value).strip()
    return text if text else "None"


def create_text_report(
    answer_key: pd.DataFrame,
    evaluated_rankings: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    cutoffs: list[int],
) -> str:
    relevant_by_job = answer_key.groupby("Job ID")["Relevant"].sum()
    score_min = evaluated_rankings["Similarity Score"].min()
    score_mean = evaluated_rankings["Similarity Score"].mean()
    score_max = evaluated_rankings["Similarity Score"].max()

    lines = [
        "Candidate Recommendation Evaluation & Analysis",
        "=" * 52,
        "Status: FINAL - answer key approved by the project team",
        "",
        "Evaluation setup",
        "----------------",
        "Recommendation method: TF-IDF cosine similarity",
        f"Jobs: {answer_key['Job ID'].nunique()}",
        f"Candidates per job: {answer_key.groupby('Job ID').size().iloc[0]}",
        f"Answer-key pairs: {len(answer_key)}",
        f"Relevant pairs: {int(answer_key['Relevant'].sum())}",
        f"Cutoffs: {', '.join(f'Top-{cutoff}' for cutoff in cutoffs)}",
        "",
        "Relevant candidates by job:",
    ]

    for job_id, relevant_count in relevant_by_job.items():
        job_title = answer_key.loc[
            answer_key["Job ID"] == job_id, "Job Title"
        ].iloc[0]
        lines.append(f"  {job_id} ({job_title}): {int(relevant_count)}")

    lines.extend(
        [
            "",
            "Metric definitions",
            "------------------",
            "Precision@K = relevant candidates in Top-K / K",
            "Recall@K = relevant candidates in Top-K / all relevant candidates",
            "F1@K = harmonic mean of Precision@K and Recall@K",
            "Macro average = equal weight for each job",
            "Micro average = aggregate TP/FP/FN across all jobs",
        ]
    )

    for cutoff in cutoffs:
        lines.extend(
            [
                "",
                f"Top-{cutoff} results",
                "-" * (14 + len(str(cutoff))),
            ]
        )
        cutoff_metrics = metrics[metrics["K"] == cutoff]
        for _, row in cutoff_metrics.iterrows():
            lines.append(
                f"{row['Job ID']} ({row['Job Title']}): "
                f"TP={int(row['True Positives'])}, "
                f"FP={int(row['False Positives'])}, "
                f"FN={int(row['False Negatives'])}, "
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

        summary_row = summary[summary["K"] == cutoff].iloc[0]
        lines.extend(
            [
                "",
                f"Macro: Precision={summary_row['Macro Precision']:.4f}, "
                f"Recall={summary_row['Macro Recall']:.4f}, "
                f"F1={summary_row['Macro F1']:.4f}",
                f"Micro: Precision={summary_row['Micro Precision']:.4f}, "
                f"Recall={summary_row['Micro Recall']:.4f}, "
                f"F1={summary_row['Micro F1']:.4f}",
            ]
        )

    lines.extend(
        [
            "",
            "Relevant candidate ranks",
            "------------------------",
        ]
    )
    for job_id in sorted(answer_key["Job ID"].unique()):
        relevant_ranks = evaluated_rankings[
            (evaluated_rankings["Job ID"] == job_id)
            & (evaluated_rankings["Relevant"] == 1)
        ].sort_values("Rank")
        rank_text = ", ".join(
            f"{row['Candidate ID']} (rank {int(row['Rank'])})"
            for _, row in relevant_ranks.iterrows()
        )
        lines.append(f"  {job_id}: {rank_text}")

    top3_summary = summary[summary["K"] == min(cutoffs)].iloc[0]
    largest_cutoff_summary = summary[summary["K"] == max(cutoffs)].iloc[0]
    additional_hits = int(
        largest_cutoff_summary["Total True Positives"]
        - top3_summary["Total True Positives"]
    )

    lines.extend(
        [
            "",
            "Analysis",
            "--------",
            f"The Top-{min(cutoffs)} recommendations retrieved "
            f"{int(top3_summary['Total True Positives'])} of "
            f"{int(top3_summary['Total Relevant'])} relevant pairs overall.",
            f"Expanding from Top-{min(cutoffs)} to Top-{max(cutoffs)} "
            f"added {additional_hits} additional relevant candidates.",
            f"Similarity scores range from {score_min:.4f} to {score_max:.4f} "
            f"with a mean of {score_mean:.4f}.",
            "Jobs with only one relevant candidate have a maximum possible "
            "Precision@3 of 0.3333 and Precision@5 of 0.2000 even when that "
            "candidate is ranked first.",
            "Absolute similarity scores are low because TF-IDF rewards exact "
            "term overlap and the job and candidate documents share few terms.",
            "",
            "Limitations",
            "-----------",
            "- The answer key uses team-approved manual relevance judgements; "
            "the reported metrics depend on those labels.",
            "- The candidate pool is very small and has uneven numbers of "
            "relevant candidates across jobs.",
            "- Some relevance decisions use transferable skills because the "
            "sample contains few direct HR and Civil Engineering matches.",
            "- Precision@K is sensitive to having fewer than K relevant "
            "candidates in the available pool.",
            "- TF-IDF cannot recognize semantic matches that use different "
            "wording and does not separately weight skills experience and "
            "education.",
        ]
    )

    return "\n".join(lines) + "\n"


def rounded_copy(dataframe: pd.DataFrame) -> pd.DataFrame:
    rounded = dataframe.copy()
    float_columns = rounded.select_dtypes(include="float").columns
    rounded[float_columns] = rounded[float_columns].round(4)
    return rounded


def run(
    answer_key_path: Path,
    rankings_path: Path,
    results_dir: Path,
    cutoffs: list[int],
) -> None:
    cutoffs = sorted(set(cutoffs))
    answer_key, rankings = load_inputs(answer_key_path, rankings_path)
    validate_inputs(answer_key, rankings, cutoffs)
    evaluated_rankings, metrics, summary = evaluate(
        answer_key, rankings, cutoffs
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    rounded_copy(evaluated_rankings).to_csv(
        results_dir / "evaluated_rankings.csv", index=False
    )
    rounded_copy(metrics).to_csv(
        results_dir / "evaluation_metrics.csv", index=False
    )
    rounded_copy(summary).to_csv(
        results_dir / "evaluation_summary.csv", index=False
    )
    report = create_text_report(
        answer_key,
        evaluated_rankings,
        metrics,
        summary,
        cutoffs,
    )
    (results_dir / "evaluation_analysis.txt").write_text(
        report, encoding="utf-8"
    )

    print("Evaluation completed successfully.")
    print(f"Answer-key pairs: {len(answer_key)}")
    print(f"Ranked pairs: {len(rankings)}")
    print(f"Results saved to: {results_dir}")
    print("\nOverall metrics:")
    print(rounded_copy(summary).to_string(index=False))


if __name__ == "__main__":
    arguments = parse_args()
    run(
        answer_key_path=arguments.answer_key,
        rankings_path=arguments.rankings,
        results_dir=arguments.results_dir,
        cutoffs=arguments.ks,
    )
