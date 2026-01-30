# Fairness Metrics

## Overview

Fairness metrics quantify the degree to which an AI system treats different demographic groups equitably. Different metrics capture different fairness concepts, and the choice depends on the application context.

---

## Primary Metrics

### Demographic Parity (Statistical Parity)

**Definition**: Equal positive prediction rates across protected groups.

```
DP = P(Ŷ=1 | A=0) / P(Ŷ=1 | A=1)
```

- **Target**: DP = 1.0 (perfect parity)
- **Acceptable Range**: 0.8 - 1.25 (80% rule)
- **Use Case**: When outcomes should be independent of group membership

**Example**: Loan approval rates should be similar across racial groups.

---

### Equal Opportunity

**Definition**: Equal true positive rates across protected groups.

```
EO = TPR_group0 / TPR_group1
```

- **Target**: EO = 1.0
- **Use Case**: When qualified individuals from all groups should have equal chances of positive outcomes

**Example**: Qualified candidates from all backgrounds should have equal interview callback rates.

---

### Equalized Odds

**Definition**: Equal true positive rates AND equal false positive rates across groups.

```
TPR_group0 ≈ TPR_group1
FPR_group0 ≈ FPR_group1
```

- **Target**: Both ratios = 1.0
- **Use Case**: When both accepting qualified and rejecting unqualified applicants matters

**Example**: Criminal risk assessment should have equal error rates across groups.

---

## Secondary Metrics

### Disparate Impact Ratio

**Definition**: Ratio of positive outcomes between groups.

```
DIR = min(P(Ŷ=1|A=0), P(Ŷ=1|A=1)) / max(P(Ŷ=1|A=0), P(Ŷ=1|A=1))
```

- **Legal Threshold**: DIR ≥ 0.8 (80% rule from US employment law)

### Predictive Parity

**Definition**: Equal positive predictive value across groups.

```
PPV_group0 ≈ PPV_group1
```

---

## Metric Selection Guide

| Context | Recommended Metric | Rationale |
|---------|-------------------|-----------|
| Lending | Demographic Parity | ECOA/Fair Lending Act focus |
| Hiring | Equal Opportunity | Qualified candidates should have equal chances |
| Criminal Justice | Equalized Odds | Both types of errors matter |
| Healthcare | Calibration | Predictions should be equally accurate |

---

## Trade-offs

**Impossibility Theorem**: It is mathematically impossible to achieve perfect scores on all fairness metrics simultaneously when base rates differ between groups.

**Practical Implication**: Choose metrics based on:
1. Legal requirements
2. Stakeholder values
3. Application context
4. Downstream impact

---

## Navigation

- [Back to Methodology](./README.md)
- [RAG System](./rag-system.md)
- [Validation Framework](./validation-framework.md)
