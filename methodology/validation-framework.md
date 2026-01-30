# Validation Framework

## Overview

Statistical validation ensures that observed fairness improvements are real, not artifacts of sampling variation or overfitting.

---

## Validation Methods

### 1. Permutation Testing

**Purpose**: Determine if improvement is statistically significant.

**Process**:
1. Calculate observed improvement (e.g., ΔDP = 0.63)
2. Randomly shuffle intervention labels 1000+ times
3. Calculate improvement under each permutation
4. Compute p-value: proportion of permutations ≥ observed

**Interpretation**:
- p < 0.05: Statistically significant
- p < 0.01: Highly significant
- p < 0.001: Very highly significant

**Example from EquiHire**:
```
Observed DP improvement: +109.5%
Permutation p-value: 0.0002
Interpretation: Improvement is real (p < 0.001)
```

---

### 2. Bootstrap Confidence Intervals

**Purpose**: Estimate uncertainty around point estimates.

**Process**:
1. Resample data with replacement 1000+ times
2. Calculate metric for each bootstrap sample
3. Compute 2.5th and 97.5th percentiles

**Interpretation**:
- Narrow CI: High precision
- CI excludes baseline: Significant improvement
- CI includes 1.0: Parity achieved

**Example from MidCity**:
```
Final DP: 0.90
95% Bootstrap CI: [0.85, 0.94]
Interpretation: True DP between 0.85-0.94 with 95% confidence
```

---

### 3. Cross-Validation

**Purpose**: Assess robustness across data splits.

**Process**:
1. Split data into K folds (typically K=5)
2. Train intervention on K-1 folds
3. Evaluate on held-out fold
4. Repeat for all folds
5. Report mean and variance

**Interpretation**:
- Low variance: Robust results
- High variance: Results may be data-dependent

---

## Validation Criteria

### Minimum Requirements

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| p-value | < 0.05 | Statistical significance |
| CI width | < 0.20 | Reasonable precision |
| CV stability | σ < 0.10 | Robust to data variation |

### Reporting Standards

Every result should include:
1. Point estimate (e.g., DP = 0.90)
2. Confidence interval (e.g., 95% CI: [0.85, 0.94])
3. Statistical test (e.g., permutation p < 0.001)
4. Sample size (e.g., n = 50,000)

---

## Trade-off Analysis

### Fairness-Accuracy Pareto Frontier

Visualize the trade-off between fairness improvement and accuracy cost:

```
Accuracy
   ^
   │    * Baseline
   │     \
   │      * Stage 1
   │       \
   │        * Stage 2
   │         \
   │          * Final
   └────────────────> Fairness
```

### Acceptable Trade-off Guidelines

| Fairness Improvement | Maximum Accuracy Cost |
|---------------------|----------------------|
| < 20% | < 1% |
| 20-50% | < 2% |
| 50-100% | < 3% |
| > 100% | < 5% |

---

## Common Pitfalls

### 1. Multiple Testing
**Problem**: Testing many techniques inflates Type I error.
**Solution**: Apply Bonferroni or FDR correction.

### 2. Data Leakage
**Problem**: Using test data to tune interventions.
**Solution**: Strict train/validation/test splits.

### 3. Overfitting to Fairness
**Problem**: Optimizing fairness at cost of utility.
**Solution**: Monitor accuracy throughout pipeline.

### 4. Ignoring Intersectionality
**Problem**: Aggregate fairness hides subgroup disparities.
**Solution**: Report metrics for intersectional groups.

---

## Validation Checklist

- [ ] Permutation test conducted (p < 0.05)
- [ ] Bootstrap CIs reported (95%)
- [ ] Cross-validation performed (K ≥ 5)
- [ ] Trade-off analysis documented
- [ ] Intersectional analysis included
- [ ] Sample sizes reported
- [ ] Multiple testing correction applied

---

## Navigation

- [Back to Methodology](./README.md)
- [Fairness Metrics](./fairness-metrics.md)
- [RAG System](./rag-system.md)
