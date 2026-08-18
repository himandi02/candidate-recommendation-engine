# Evaluation & Analysis Methodology

## Scope

This component evaluates the ranked recommendations produced by
`src/similarity_recommendation.py`. It does not modify the similarity or
ranking algorithm.

The evaluation uses:

- `data/ground_truth.csv` as the dataset-provided matched-score reference.
- `data/evaluation_answer_key.csv` as the team-approved relevance
  answer key.
- `results/candidate_rankings.csv` as the complete recommendation ranking.
- `results/similarity_scores.csv` for full-precision cosine scores.
- Top-3 and Top-5 cutoffs, matching the project objectives.

## Dataset ground truth

The task-provided `ground_truth.csv` contains 12 observed candidate-job pairs
and a continuous `matched_score` for each pair. It does not provide scores for
the other 36 cross-job combinations and does not contain binary relevance
labels.

The ground-truth evaluation therefore performs two separate analyses:

1. It compares cosine similarity with `matched_score` on the 12 observed pairs
   using MAE, MSE, RMSE, Pearson correlation, Spearman correlation, and R
   squared.
2. It calculates Top-3 and Top-5 classification metrics by treating an
   observed pair with `matched_score >= 0.50` as relevant and treating the 36
   unlisted cross-job pairs as non-relevant.

The `0.50` threshold is explicit and configurable because the dataset does not
provide its own binary cutoff:

```text
python src/evaluate_ground_truth.py --threshold 0.50
```

## Team-approved 48-pair answer key

The answer key contains all 48 job-candidate combinations: 4 jobs multiplied
by 12 candidates. Each pair has a binary `Relevant` label and supporting
judgements for skills, experience, and education.

The relevance rules are:

- `1` means the candidate is considered suitable or reasonably transferable
  for the job.
- `0` means the candidate does not have sufficient overall alignment.
- `High`, `Medium`, and `Low` describe skill and experience alignment.
- `Yes`, `Partial`, and `No` describe education alignment.

The project team reviewed and approved these manual relevance labels before
the final metrics were calculated. Any later label change should be agreed by
the team and followed by a complete evaluation rerun.

Current relevant-pair distribution:

| Job | Relevant candidates |
|---|---:|
| Senior Software Engineer | 3 |
| HR Officer | 1 |
| Civil Engineer | 1 |
| Business Development Executive | 3 |

## Metrics

For each job and cutoff `K`:

```text
Precision@K = relevant candidates in Top-K / K
Recall@K = relevant candidates in Top-K / all relevant candidates
F1@K = 2 * Precision@K * Recall@K / (Precision@K + Recall@K)
Accuracy@K = (true positives + true negatives) / all candidate-job pairs
```

Accuracy is included in the dataset ground-truth evaluation. It is reported
alongside Precision, Recall, and F1 because the large number of non-relevant
pairs can make accuracy appear high even when few relevant candidates are
recommended.

The output includes two overall views:

- **Macro average:** calculates the metric for each job and gives every job
  equal weight.
- **Micro average:** aggregates true positives, false positives, and false
  negatives across all jobs before calculating the metric.

When a job has fewer than `K` relevant candidates, its maximum possible
Precision@K is lower than 1. For example, a job with one relevant candidate
has a maximum Precision@3 of `0.3333`, even if that candidate is ranked first.

## Running the evaluation

From the project root:

```text
python src/evaluate_ground_truth.py
python src/evaluate_recommendations.py
```

`evaluate_ground_truth.py` performs the task dataset evaluation.
`evaluate_recommendations.py` provides the additional evaluation against the
team-approved 48-pair answer key.

Optional paths and cutoffs can be supplied:

```text
python src/evaluate_recommendations.py \
  --answer-key data/evaluation_answer_key.csv \
  --rankings results/candidate_rankings.csv \
  --ks 3 5
```

## Generated outputs

Dataset ground-truth outputs:

- `results/ground_truth_pair_comparison.csv`: observed matched scores compared
  with cosine similarities.
- `results/ground_truth_score_metrics.csv`: score errors and correlations.
- `results/ground_truth_labeled_rankings.csv`: all 48 ranking pairs with the
  threshold-derived labels.
- `results/ground_truth_topk_metrics.csv`: per-job Accuracy, Precision, Recall,
  and F1 at each cutoff.
- `results/ground_truth_evaluation_summary.csv`: overall macro and micro
  metrics.
- `results/ground_truth_evaluation_analysis.txt`: readable findings and
  required assumptions.

Team-approved answer-key outputs:

- `results/evaluated_rankings.csv`: all rankings joined with answer-key labels.
- `results/evaluation_metrics.csv`: per-job Precision, Recall, and F1 results.
- `results/evaluation_summary.csv`: macro and micro overall metrics.
- `results/evaluation_analysis.txt`: readable results, analysis, and
  limitations for the report.

## Interpretation limitations

- The dataset ground truth covers only 12 of the 48 possible pairs.
- Its other 36 cross-job pairs must be assumed non-relevant for Top-K metrics.
- The `0.50` relevance threshold is an evaluation decision because the source
  provides continuous scores rather than binary labels.
- Cosine similarity is not calibrated to reproduce the scale of the provided
  `matched_score` values.
- The answer key uses team-approved manual judgements and the metrics depend on
  those relevance decisions.
- Only 12 candidates and 4 jobs are evaluated.
- Relevant candidates are unevenly distributed across jobs.
- HR Officer and Civil Engineer have only one relevant candidate each.
- Some labels accept transferable experience because the sample has few
  direct matches for those jobs.
- TF-IDF uses exact term overlap and cannot reliably identify synonyms or
  broader semantic matches.
- Skills, experience, and education are combined in one document instead of
  being scored as separately weighted factors.
