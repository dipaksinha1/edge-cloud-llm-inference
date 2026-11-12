# 0004 · Hybrid router: logistic-regression difficulty score + 3-sample agreement

**Context.** The proposal report specifies a two-stage router after Ding et al. (ICLR 2024), kept simple for the course timeline, with a random router at the same offload ratio as the baseline.

**Decision.** Stage 1: scikit-learn `LogisticRegression(class_weight="balanced")` on cheap features (word count, number count, math operators, dataset, MMLU subject group, longest number), trained on a calibration set labelled "edge model wrong". Score > threshold (0.6) → cloud. Stage 2: 3 edge samples at temperature 0.7, and at least 2 equal extracted answers keep the answer on the edge, otherwise → cloud. The random router uses the hybrid's measured offload ratio.

**Alternatives.** A BERT-style learned router like Ding et al. (needs GPU training, out of scope). A rule-only router (doesn't match the proposal). Stage 2 only (more edge calls on obviously hard questions).

**Consequences.** Hybrid edge cost is up to 3 edge calls per question, which shows up in latency, and that's an honest result. Calibration and test sets must stay disjoint.
