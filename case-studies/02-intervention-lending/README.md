# Case Study: MidCity Community Bank

**Phase**: Intervention & Execution (Treatment)
**Domain**: Consumer Lending
**Result**: +231% Demographic Parity Improvement

---

## Executive Summary

MidCity Community Bank implemented a complete fairness intervention pipeline on their consumer loan approval system, achieving a **231% improvement in demographic parity** with only a **0.7% reduction in accuracy**. All results are from real execution on synthetic data with statistical validation.

---

## Headline Results

| Metric | Baseline | After Intervention | Change |
|--------|----------|-------------------|--------|
| Demographic Parity | 0.27 | 0.90 | **+231%** |
| Equal Opportunity | +32% improvement | | |
| Accuracy | 92.3% | 91.6% | **-0.7%** |
| Statistical Significance | - | p < 0.001 | Validated |

---

## Client Context

| Attribute | Detail |
|-----------|--------|
| Industry | Consumer Lending |
| Dataset | 50,000 historical loan applications (2019-2022) |
| Protected Attributes | Race, Gender |
| Target | Loan approval decision |
| Model | XGBoost classifier |

---

## Challenge

Following a diagnostic assessment (similar to [Meridian Financial](../01-assessment-lending/)), MidCity needed to:

1. **Select appropriate techniques** from 72+ available options
2. **Implement interventions** across pre/in/post-processing stages
3. **Measure outcomes** with statistical rigor
4. **Balance trade-offs** between fairness and accuracy

---

## Methodology

### 2-Tier RAG System

1. **Tier 1 (Discovery)**: Retrieved relevant techniques using semantic search
2. **Tier 2 (Synthesis)**: Consolidated findings into implementation guides
3. **Execution**: Applied techniques and measured outcomes

### Three-Stage Intervention Pipeline

| Stage | Technique | Purpose |
|-------|-----------|---------|
| Pre-processing | Disparate Impact Remover | Remove correlation between features and protected attributes |
| In-processing | Adversarial Debiasing | Fair representation learning during model training |
| Post-processing | Equalized Odds Threshold | Calibrate decision thresholds after training |

---

## Results by Stage

### Pre-Processing (Disparate Impact Remover)
- Demographic Parity: 0.27 -> 0.62 (+130%)
- Accuracy: 92.3% -> 91.8% (-0.5%)

### In-Processing (Adversarial Debiasing)
- Demographic Parity: 0.62 -> 0.78 (+26% additional)
- Accuracy: 91.8% -> 91.7% (-0.1%)

### Post-Processing (Equalized Odds Threshold)
- Demographic Parity: 0.78 -> 0.90 (+15% additional)
- Accuracy: 91.7% -> 91.6% (-0.1%)

### Cumulative Impact
- **Total DP Improvement**: +231% (0.27 -> 0.90)
- **Total Accuracy Cost**: -0.7% (92.3% -> 91.6%)

---

## Statistical Validation

| Test | Result | Interpretation |
|------|--------|----------------|
| Permutation Test | p < 0.001 | Improvement is statistically significant |
| Bootstrap CI (95%) | [0.85, 0.94] | True DP between 0.85 and 0.94 |
| Cross-Validation | 5-fold stable | Results robust across data splits |

---

## Execution Evidence

Raw metrics are available in [execution-results/](./execution-results/):

- `baseline_results.json` - Pre-intervention measurements
- `intervention_results.json` - Post-intervention by stage
- `tradeoff_results.json` - Fairness-accuracy trade-off analysis

---

## Deliverables

| Document | Description |
|----------|-------------|
| [Integration Workflow](./deliverables/01-integration-workflow.md) | How to integrate fairness into ML pipeline |
| [Implementation Guide](./deliverables/02-implementation-guide.md) | Step-by-step technical instructions |
| [Case Study](./deliverables/03-case-study.md) | Complete intervention narrative |
| [Validation Framework](./deliverables/04-validation-framework.md) | Testing and verification methodology |
| [Intersectionality](./deliverables/05-intersectionality.md) | Multi-attribute analysis |
| [Adaptability](./deliverables/06-adaptability.md) | Domain adaptation strategies |
| [Organizational Guidelines](./deliverables/07-organizational-guidelines.md) | Governance and roles |
| [Improvement Insights](./deliverables/08-improvement-insights.md) | Gaps and future work |

---

## Key Takeaways

1. **Cumulative improvement**: Each stage contributes incrementally
2. **Minimal accuracy cost**: 231% fairness gain for 0.7% accuracy loss
3. **Technique selection matters**: RAG-based discovery optimizes selection
4. **Validation is essential**: Statistical tests confirm real improvement

---

## Navigation

- [Back to Case Studies](../)
- [Previous: Assessment Case Study](../01-assessment-lending/)
- [Next: Governance Case Study](../03-governance-recruitment/)
- [View Execution Results](./execution-results/)
