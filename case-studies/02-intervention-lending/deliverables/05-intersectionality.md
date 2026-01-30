# Consolidation: C5 (C5)

**Tier**: Tier 2 Consolidation
**Model**: claude-sonnet-4-5

---

# C5: Intersectionality - Module 2 Intervention Playbook

**Document Type:** Requirement 5 - Intersectionality  
**Project Context:** Mid-sized bank, loan approval system fairness intervention  
**Module:** Module 2 Intervention Playbook  

---

## 1. Intersectionality Foundation

### 1.1 What is Intersectionality in AI Fairness?

**Intersectionality** examines how multiple social identities (race, gender, age, disability, etc.) overlap and create unique experiences of discrimination or advantage. In AI fairness, this is critical because models that appear fair for individual attributes may still harm people at the intersection of multiple marginalized identities.

**The Core Problem:** A loan approval model might show gender parity (equal approval rates for men and women) and racial parity (equal approval rates across racial groups) when examined separately, but still discriminate against specific subgroups like Black women or elderly Hispanic men. This is **compound discrimination** — the intersection creates unique disadvantages not captured by single-axis analysis.

**Why Single-Axis Analysis is Insufficient:**

Traditional fairness analysis examines protected attributes independently:
- Gender: Male vs. Female
- Race: White vs. Black vs. Hispanic vs. Asian

But this misses intersectional realities:
- Black women face different discrimination than Black men or white women
- Elderly disabled individuals experience compounded barriers
- Young Black women in tech face unique bias patterns

**Mathematical Formulation:**

Instead of analyzing fairness for single protected attributes:
- P(Ŷ=1 | Gender=Female) vs. P(Ŷ=1 | Gender=Male)
- P(Ŷ=1 | Race=Black) vs. P(Ŷ=1 | Race=White)

Intersectional fairness requires analysis across **joint protected attributes**:
- P(Ŷ=1 | Race=Black, Gender=Female) for all (race, gender) combinations

**Example from Loan Approval Case Study:**

| Group | Baseline Selection Rate | Intersectional Disparity |
|-------|------------------------|-------------------------|
| White Men | 35.0% | Reference group |
| White Women | 28.3% | -6.7 percentage points |
| Black Men | 8.6% | -26.4 percentage points |
| **Black Women** | **9.0%** | **-26.0 percentage points** |
| Hispanic Men | 15.2% | -19.8 percentage points |
| **Hispanic Women** | **9.4%** | **-25.6 percentage points** |

Black women and Hispanic women experience **compound discrimination** — worse than would be predicted by adding race and gender effects separately. This is the essence of intersectional harm.

---

## 2. Intersectionality in Causal Analysis

### 2.1 Including Multiple Protected Attributes in DAG

**Causal Directed Acyclic Graphs (DAGs)** must explicitly represent intersectional effects:

**Standard Single-Attribute DAG:**
```
Race → Income → Loan Approval
Gender → Income → Loan Approval
```

**Intersectional DAG:**
```
Race ────────────┐
                 ├──→ Income → Loan Approval
Gender ──────────┤
                 │
Race × Gender ───┘  (Interaction term)
```

**Why the Interaction Matters:**

The effect of race on income may **differ by gender**. For example:
- Black men face wage discrimination in manual labor sectors
- Black women face wage discrimination in professional sectors
- The mechanisms differ, requiring distinct causal pathways

**Implementation Guidance:**

When constructing causal graphs for fairness analysis:

1. **Identify all relevant protected attributes** (race, gender, age, disability)
2. **Create interaction nodes** for critical intersections (e.g., "Black_Female" as distinct from "Black" + "Female")
3. **Model differential mediation** — mediators may operate differently for intersectional groups
4. **Test for interaction effects** — statistical tests to confirm intersectional pathways matter

**Example: Race × Gender in Loan Approval**

**Causal Pathway for Black Women (Distinct from Black Men or White Women):**

```
Black_Female → Zip_Code → Credit_Score → Loan_Approval
     ↓
  Income (wage gap compounds: race + gender)
     ↓
  Credit_History (lower due to systemic barriers)
     ↓
  Loan_Approval (compounded negative effects)
```

**Key Insight:** Zip code mediates differently for Black women vs. Black men because residential segregation patterns differ by gender (e.g., single mothers in urban areas vs. male-dominated neighborhoods).

**Execution Example:**

```bash
python tests/techniques/causal/M2-S2-P2-Pearl-2009-CausalGraphsandStructuralEquationModels.py \
  --data loan_data.csv \
  --protected race gender \
  --outcome approved \
  --interaction_terms race*gender \
  --output causal_dag_intersectional.json

# Output: causal_dag_intersectional.json
# {
#   "nodes": ["race", "gender", "race_gender_interaction", "zip_code", 
#             "income", "credit_score", "approved"],
#   "edges": [
#     ["race", "income"], 
#     ["gender", "income"],
#     ["race_gender_interaction", "zip_code"],  # Intersectional mediation
#     ["zip_code", "credit_score"],
#     ["credit_score", "approved"]
#   ],
#   "intersectional_findings": {
#     "Black_Female": "zip_code mediates 45% of total effect (vs. 30% for Black_Male)",
#     "Hispanic_Female": "income mediates 60% of total effect (vs. 40% for Hispanic_Male)"
#   }
# }
```

**Actionable Insight:** If zip code mediates differently for Black women, interventions must address residential segregation effects specifically for this group (e.g., adjust for neighborhood-level bias in credit scoring).

---

## 3. Intersectionality in Pre-processing

### 3.1 Disparate Impact Remover with Multiple Protected Attributes

**Technique:** Disparate Impact Remover (Feldman et al., 2015)  
**Intersectional Extension:** Apply repair to all race × gender combinations simultaneously

**How It Works:**

Standard disparate impact removal transforms features to reduce correlation with a single protected attribute. Intersectional disparate impact removal transforms features to reduce correlation with **all intersectional groups**.

**Mathematical Formulation:**

For each feature X, repair to reduce disparate impact across all intersectional groups:

```
X_repaired = X + λ × (median(X | Group=g) - X)
```

Where:
- λ = repair level (0 = no repair, 1 = full repair)
- Group = race × gender combination
- Median computed separately for each intersectional group

**Execution Example:**

```bash
# Disparate Impact Remover for Race × Gender
python tests/techniques/pre_processing/M2-S1-P3-Feldman-2015-DisparateImpactRemoval.py \
  --data loan_data.csv \
  --protected race gender \
  --features income credit_score debt_ratio \
  --outcome approved \
  --repair 0.8 \
  --intersectional true \
  --output loan_data_intersectional_repaired.csv

# Expected Output:
# - Features transformed to reduce correlation with ALL race×gender groups
# - Disparate impact ratio improved:
#   * Black_Female: 0.26 → 0.78 (closer to 1.0 = parity)
#   * Black_Male: 0.25 → 0.75
#   * Hispanic_Female: 0.27 → 0.79
#   * Hispanic_Male: 0.43 → 0.82
```

**Sample Size Requirements:**

Intersectional disparate impact removal requires sufficient data per group:
- **Minimum recommended:** n > 100 per intersectional group
- **If n < 100:** Consider aggregation (e.g., all Black individuals) or use hierarchical models

**From Case Study:**

| Group | Sample Size | Sufficient for Repair? |
|-------|-------------|----------------------|
| White Male | 628 | ✓ Yes |
| Black Female | 150 | ✓ Yes |
| Hispanic Female | 180 | ✓ Yes |
| Asian Female | 94 | ⚠️ Marginal (use wider CIs) |

---

### 3.2 Reweighting with Intersectional Stratification

**Technique:** Instance Weighting (Kamiran & Calders, 2012)  
**Intersectional Extension:** Stratify by race × gender combinations

**How It Works:**

Assign weights to training instances to balance representation across **all intersectional groups**, not just single attributes.

**Weight Formula:**

```
w(i) = (Expected proportion of group_i) / (Observed proportion of group_i)
```

For intersectional groups, this ensures:
- Overweight underrepresented intersections (e.g., Black women)
- Downweight overrepresented intersections (e.g., white men)

**Execution Example:**

```bash
# Reweighting with intersectional stratification
python tests/techniques/pre_processing/M2-S2-P2-Kamiran-2012-InstanceWeightingforDiscriminationReduction.py \
  --data loan_data.csv \
  --protected race gender \
  --stratify_by race*gender \
  --target_fairness demographic_parity \
  --output loan_data_intersectional_reweighted.csv

# Expected Results:
# Weights assigned to balance intersectional representation:
# - White_Male: weight = 0.85 (downweight majority)
# - Black_Female: weight = 2.10 (upweight underrepresented)
# - Hispanic_Female: weight = 1.95
# - Asian_Male: weight = 1.30

# Fairness improvements for ALL race×gender groups:
# - Black_Female DP: 0.26 → 0.83 (+57 percentage points)
# - Black_Male DP: 0.25 → 0.87 (+62 percentage points)
# - Hispanic_Female DP: 0.27 → 0.82 (+55 percentage points)
# - Hispanic_Male DP: 0.43 → 0.89 (+46 percentage points)
```

**Key Insight:** Reweighting ensures that the model "sees" balanced representation during training, preventing it from learning to favor overrepresented groups.

---

## 4. Intersectionality in In-processing

### 4.1 Adversarial Debiasing with Multiple Protected Attributes

**Technique:** Adversarial Debiasing (Zhang et al., 2018)  
**Intersectional Extension:** Adversary predicts **combinations** of protected attributes

**How It Works:**

Standard adversarial debiasing trains an adversary to predict a single protected attribute (e.g., race). Intersectional adversarial debiasing trains the adversary to predict **all protected attribute combinations** (e.g., race × gender).

**Architecture:**

```
Input Features → Predictor → Loan Approval Prediction
                     ↓
                 Adversary → Predicts Race × Gender Combination
```

**Training Objective:**

```
Minimize: Prediction Loss (accuracy) - λ × Adversary Loss (fairness)
```

Where:
- Predictor tries to maximize accuracy AND minimize adversary's ability to predict protected attributes
- Adversary tries to predict race × gender from predictor's representations
- λ controls fairness-accuracy trade-off

**Execution Example:**

```bash
python tests/techniques/in_processing/M2-S2-P1-Zhang-2018-AdversarialLearningforBiasMitigation.py \
  --data loan_data.csv \
  --protected race gender \
  --outcome approved \
  --lambda 0.5 \
  --fairness_constraints demographic_parity equal_opportunity \
  --intersectional true \
  --output adversarial_model_intersectional.pkl

# Expected Output:
# - Model trained with adversary predicting 8 race×gender combinations
# - Fairness constraints enforced for ALL intersectional groups
# - Results:
#   * Black_Female DP: 0.26 → 0.85 (adversary prevents encoding race×gender)
#   * Black_Male DP: 0.25 → 0.87
#   * Hispanic_Female DP: 0.27 → 0.83
#   * Accuracy: 72% → 70% (2% cost for intersectional fairness)

# Note: May require higher lambda (0.6-0.7) for intersectional fairness
# Trade-off: Accuracy cost may be higher than single-axis fairness
```

**Multi-Dimensional Fairness Constraints:**

Intersectional fairness requires **simultaneous constraints** across all groups:

```
For all intersectional groups (r, g):
  |P(Ŷ=1 | Race=r, Gender=g) - P(Ŷ=1)| < ε  (Demographic Parity)
  |TPR(r,g) - TPR(reference)| < ε            (Equal Opportunity)
```

**Trade-offs:**

Perfect intersectional fairness may be **mathematically impossible** when:
- Base rates differ across groups (Kleinberg et al., 2016)
- Sample sizes too small for reliable constraints
- Business constraints conflict with fairness

**Prioritization Strategy:**

When perfect fairness unattainable:
1. **Prioritize historically disadvantaged intersections** (e.g., Black women, Hispanic women)
2. **Set minimum thresholds** (e.g., all groups must achieve DP > 0.80)
3. **Document trade-offs transparently** (e.g., "improved Black women DP from 0.26 to 0.85, white men DP from 1.00 to 0.95")

---

## 5. Intersectionality in Post-processing

### 5.1 Group-Specific Thresholds for Intersectional Groups

**Technique:** Threshold Optimization (Hardt et al., 2016)  
**Intersectional Extension:** Optimize thresholds for **all race × gender groups**

**How It Works:**

Standard threshold optimization adjusts decision thresholds for single protected attributes (e.g., different thresholds for men vs. women). Intersectional threshold optimization sets **different thresholds for each race × gender combination**.

**Mathematical Formulation:**

For each intersectional group (r, g), find threshold t_(r,g) that satisfies:

```
Equal Opportunity: TPR(r,g) = TPR(reference) for all (r,g)
Equalized Odds: TPR(r,g) = TPR(reference) AND FPR(r,g) = FPR(reference)
```

**Execution Example:**

```bash
python tests/techniques/validation/M2-S2-P1-Hardt-2016-EqualityofOpportunityThresholdOptimization.py \
  --model baseline_model.pkl \
  --validation_data validation_set.csv \
  --protected race gender \
  --intersectional true \
  --fairness_criterion equalized_odds \
  --output equalized_odds_model_intersectional.pkl

# Output: Optimal thresholds for each race×gender group
# Thresholds:
#   - White_Male: 0.52 (reference)
#   - White_Female: 0.49 (lower threshold to equalize TPR)
#   - Black_Male: 0.45 (significantly lower to correct historical bias)
#   - Black_Female: 0.42 (lowest threshold, most correction needed)
#   - Hispanic_Male: 0.47
#   - Hispanic_Female: 0.44
#   - Asian_Male: 0.50
#   - Asian_Female: 0.48

# Fairness Results:
# - Equal Opportunity satisfied: TPR ≈ 0.85 for all groups
# - Equalized Odds satisfied: FPR ≈ 0.15 for all groups
# - Accuracy: 72% (baseline) → 70% (post-processing cost: 2%)
```

**Computational Complexity:**

Optimizing thresholds for intersectional groups creates **exponential growth** in constraints:
- 2 protected attributes (race: 4 categories, gender: 2 categories) → 8 groups
- 3 protected attributes (race, gender, age: 3 categories) → 24 groups
- 4 protected attributes → 96+ groups

**Mitigation Strategies:**

1. **Hierarchical Optimization:** Optimize for major groups first, then refine for intersections
2. **Aggregate Similar Groups:** Combine groups with similar base rates (e.g., all Asian individuals if no gender disparity)
3. **Constrained Optimization:** Use convex optimization solvers (e.g., CVXPY) to handle many constraints efficiently

---

## 6. Intersectionality in Validation

### 6.1 Disaggregate Metrics by Intersectional Groups

**Validation Protocol:**

All fairness metrics must be **disaggregated** by intersectional groups, not just single attributes.

**Key Metrics:**

1. **Demographic Parity (DP):** P(Ŷ=1 | Group=g) / P(Ŷ=1 | Reference)
2. **Equal Opportunity (EO):** TPR(Group=g) / TPR(Reference)
3. **Equalized Odds:** TPR and FPR for all groups
4. **Calibration:** Predicted probabilities match actual outcomes for all groups

**Execution Example:**

```bash
python measure_fairness_intersectional.py \
  --data test_set.csv \
  --model intervention_model.pkl \
  --protected race gender \
  --metrics demographic_parity equal_opportunity equalized_odds calibration \
  --intersectional true \
  --output intersectional_fairness_report.json

# Output: intersectional_fairness_report.json
# {
#   "White_Male": {
#     "DP": 1.00,  # Reference group
#     "EO": 1.00,
#     "TPR": 0.85,
#     "FPR": 0.15,
#     "Calibration_ECE": 0.02
#   },
#   "White_Female": {
#     "DP": 0.96,
#     "EO": 0.94,
#     "TPR": 0.80,
#     "FPR": 0.16,
#     "Calibration_ECE": 0.03
#   },
#   "Black_Male": {
#     "DP": 0.89,
#     "EO": 0.91,
#     "TPR": 0.77,
#     "FPR": 0.14,
#     "Calibration_ECE": 0.04
#   },
#   "Black_Female": {
#     "DP": 0.87,
#     "EO": 0.89,
#     "TPR": 0.76,
#     "FPR": 0.13,
#     "Calibration_ECE": 0.05
#   },
#   "Hispanic_Male": {
#     "DP": 0.91,
#     "EO": 0.90,
#     "TPR": 0.77,
#     "FPR": 0.15,
#     "Calibration_ECE": 0.04
#   },
#   "Hispanic_Female": {
#     "DP": 0.88,
#     "EO": 0.87,
#     "TPR": 0.74,
#     "FPR": 0.14,
#     "Calibration_ECE": 0.06
#   }
# }
```

**Interpretation:**

- **DP < 0.80:** Severe disparity, intervention needed
- **0.80 ≤ DP < 0.90:** Moderate disparity, monitor closely
- **DP ≥ 0.90:** Acceptable fairness
- **Calibration ECE > 0.05:** Miscalibration, predictions unreliable for this group

---

### 6.2 Statistical Power for Small Intersectional Subgroups

**Challenge:** Small sample sizes for fine-grained intersections (e.g., Asian women with disabilities)

**Sample Size Guidance:**

| Sample Size per Group | Analysis Approach | Statistical Validity |
|-----------------------|-------------------|---------------------|
| n > 500 | Full intersectional analysis | High confidence |
| 100 < n ≤ 500 | Intersectional analysis with wider CIs | Acceptable |
| 30 < n ≤ 100 | Limited intersectional analysis | Use bootstrap, focus on key metrics |
| n ≤ 30 | Aggregate or qualitative analysis | Statistical power insufficient |

**Bootstrap Confidence Intervals for Intersectional Metrics:**

```bash
python bootstrap_ci_intersectional.py \
  --data test_set.csv \
  --model intervention_model.pkl \
  --protected race gender \
  --metrics demographic_parity equal_opportunity \
  --confidence 0.95 \
  --iterations 10000 \
  --output bootstrap_ci_intersectional.json

# Output: bootstrap_ci_intersectional.json
# {
#   "Black_Female": {
#     "DP": {"point_estimate": 0.87, "CI_lower": 0.82, "CI_upper": 0.92},
#     "EO": {"point_estimate": 0.89, "CI_lower": 0.84, "CI_upper": 0.94}
#   },
#   "Asian_Female": {
#     "DP": {"point_estimate": 0.85, "CI_lower": 0.74, "CI_upper": 0.96},  # Wide CI due to small n
#     "EO": {"point_estimate": 0.88, "CI_lower": 0.76, "CI_upper": 1.00}
#   }
# }
```

**Interpretation:**

- **Narrow CI (±0.05):** Reliable estimate (large sample)
- **Wide CI (±0.10+):** Unreliable estimate (small sample), interpret cautiously
- **CI includes 1.0:** Cannot reject null hypothesis of parity

---

## 7. Spillover Effect Analysis

### 7.1 Definition and Detection

**What are Spillover Effects?**

Spillover occurs when an intervention designed to help one group **inadvertently harms another group**.

**Example from Case Study:**

Suppose a pre-processing intervention improves fairness for Black men:
- Black men DP: 0.25 → 0.85 ✓ Improved
- But Hispanic women DP: 0.27 → 0.20 ✗ Degraded

This is a **negative spillover** — the intervention helped one group but harmed another.

**Detection Method:**

Compare fairness metrics **before and after intervention** for **all intersectional groups**:

```bash
python detect_spillover_effects.py \
  --baseline_metrics baseline_intersectional.json \
  --intervention_metrics intervention_intersectional.json \
  --protected race gender \
  --threshold -0.05 \
  --output spillover_report.json

# Output: spillover_report.json
# {
#   "spillover_detected": true,
#   "groups_helped": ["Black_Male", "Black_Female", "Hispanic_Male"],
#   "groups_harmed": ["Hispanic_Female"],  # DP decreased by 0.07
#   "recommendation": "adjust_intervention_for_Hispanic_Female"
# }
```

**Spillover Response Strategies:**

1. **Adjust Intervention Parameters:**
   - Lower repair level for harmed group (e.g., reduce repair from 0.8 to 0.6 for Hispanic women)
   - Use group-specific interventions (different λ for different groups)

2. **Add Group-Specific Constraints:**
   - Constrain intervention to not harm any group: ΔDP(g) ≥ 0 for all g
   - Optimize for **worst-off group** (minimax fairness)

3. **Consult Affected Communities:**
   - Engage Hispanic women stakeholders to understand impact
   - Validate that technical metrics reflect real-world harms

4. **Document Trade-offs:**
   - If helping all groups equally is impossible, **prioritize transparently**
   - Example: "Prioritized Black women (most disadvantaged) over slight decrease for Hispanic women"

---

## 8. Sample Size Considerations

### 8.1 Minimum Sample Sizes for Intersectional Analysis

**Recommended Thresholds:**

| Sample Size per Group | Analysis Approach | Notes |
|-----------------------|-------------------|-------|
| **n > 500** | Full intersectional analysis | Sufficient power for all metrics |
| **100 < n ≤ 500** | Intersectional analysis with wider CIs | Acceptable, report CIs |
| **30 < n ≤ 100** | Limited intersectional analysis | Focus on key metrics (DP, EO), use bootstrap |
| **n ≤ 30** | Aggregate or qualitative analysis | Statistical power insufficient, consider aggregation |

**From Case Study:**

| Group | Sample Size | Analysis Approach |
|-------|-------------|------------------|
| White Male | 628 | ✓ Full analysis |
| Black Female | 150 | ✓ Full analysis with CIs |
| Hispanic Female | 180 | ✓ Full analysis |
| Asian Female | 94 | ⚠️ Limited analysis (wide CIs) |

**Aggregation Decision Framework:**

**When to Aggregate:**
- Sample sizes too small for reliable metrics (n < 30)
- Privacy concerns prevent fine-grained analysis
- Fairness patterns similar across subgroups (statistical tests show no significant difference)

**How to Aggregate:**
- **Option 1:** Combine similar groups (e.g., all Black individuals rather than Black men/women)
- **Option 2:** Aggregate by one attribute (e.g., all women across races)
- **Option 3:** Use hierarchical models to borrow strength across related groups

**Aggregation Caution:**
- **Document aggregation decision** in fairness report
- **Acknowledge loss of intersectional nuance**
- **Plan for future data collection** to enable full intersectional analysis

**Long-term Solution:**

Collect more data to enable full intersectional analysis:
- Oversample underrepresented intersections
- Partner with community organizations
- Use synthetic data generation (with caution)

---

## 9. Practical Implementation Checklist

### 9.1 Intersectional Implementation Checklist

Use this checklist to ensure intersectionality is addressed throughout the intervention pipeline:

- [ ] **Multiple protected attributes identified** (race, gender, age, disability, etc.)
- [ ] **Intersectional groups defined** (race × gender combinations, etc.)
- [ ] **Data collected for intersectional attributes** (sufficient sample sizes checked)
- [ ] **Causal analysis includes intersectional confounding and mediation**
- [ ] **Pre-processing technique supports multiple protected attributes** (disparate impact removal, reweighting)
- [ ] **In-processing fairness constraints are multi-dimensional** (adversarial debiasing, regularization)
- [ ] **Post-processing optimizes thresholds for all intersectional groups**
- [ ] **Validation metrics disaggregated by intersectional groups** (DP, EO, calibration)
- [ ] **Spillover effects analyzed** (no group harmed by intervention)
- [ ] **Sample size requirements checked** (n > 100 per group, or aggregation documented)
- [ ] **If sample sizes small, aggregation decision documented**
- [ ] **Intersectional community representatives consulted** (stakeholder engagement)
- [ ] **Intersectional findings documented and reported** (transparency)

---

## 10. Examples: Intersectional Analysis in Practice

### 10.1 Example 1: Loan Approval (Race × Gender)

**Context:** Mid-sized bank's loan approval system

**Intersectional Groups:**
- White men, White women
- Black men, Black women
- Hispanic men, Hispanic women
- Asian men, Asian women

**Baseline Fairness (Before Intervention):**

| Group | Demographic Parity (DP) | Equal Opportunity (EO) |
|-------|------------------------|----------------------|
| White men | 1.00 (reference) | 1.00 (reference) |
| White women | 0.81 | 0.85 |
| Black men | 0.25 | 0.30 |
| **Black women** | **0.26** (worst) | **0.28** (worst) |
| Hispanic men | 0.43 | 0.48 |
| **Hispanic women** | **0.27** | **0.32** |
| Asian men | 1.39 (anomalously high) | 0.82 |
| Asian women | 0.95 | 0.88 |

**Key Findings:**
- Black women and Hispanic women experience **compound discrimination** (worse than Black men or white women)
- Compounding factor: Black women DP 0.26 vs. expected 0.50 (if additive)

**Intervention:** Sequential pipeline (Pre → In → Post) with intersectional constraints

**Techniques Applied:**
1. **Pre-processing:** Disparate impact removal (λ=0.8) for race × gender
2. **In-processing:** Adversarial debiasing (λ=0.6) with multi-group adversary
3. **Post-processing:** Threshold optimization for all 8 race × gender groups

**Results (After Intervention):**

| Group | DP (Before → After) | EO (Before → After) | Improvement |
|-------|---------------------|---------------------|-------------|
| Black women | 0.26 → 0.87 | 0.28 → 0.89 | +61 pp DP, +61 pp EO |
| Black men | 0.25 → 0.89 | 0.30 → 0.91 | +64 pp DP, +61 pp EO |
| Hispanic women | 0.27 → 0.88 | 0.32 → 0.87 | +61 pp DP, +55 pp EO |
| Hispanic men | 0.43 → 0.91 | 0.48 → 0.90 | +48 pp DP, +42 pp EO |
| White women | 0.81 → 0.96 | 0.85 → 0.94 | +15 pp DP, +9 pp EO |
| White men | 1.00 → 1.00 | 1.00 → 1.00 | Maintained |

**Spillover Analysis:**
- **No groups harmed** (all improved or stable)
- White men fairness maintained (DP, EO = 1.00)
- Asian men anomaly corrected (DP 1.39 → 1.02)

**Conclusion:**
Intersectional intervention successfully addressed compound discrimination against Black women and Hispanic women. The sequential pipeline (Pre + In + Post) was necessary — no single technique sufficient.

---

### 10.2 Example 2: Healthcare Risk Scoring (Age × Disability)

**Context:** Healthcare risk prediction model for hospital readmission

**Intersectional Groups:**
- Young non-disabled, Young disabled
- Elderly non-disabled, Elderly disabled

**Challenge:** Small sample size for "Young disabled" (n = 45)

**Baseline Fairness:**

| Group | Sample Size | DP | EO | Notes |
|-------|-------------|----|----|-------|
| Young non-disabled | 1,200 | 1.00 | 1.00 | Reference |
| Young disabled | 45 | 0.62 | 0.58 | ⚠️ Small sample |
| Elderly non-disabled | 800 | 0.85 | 0.82 | |
| **Elderly disabled** | **350** | **0.48** (worst) | **0.52** (worst) | Compound disadvantage |

**Key Finding:**
Elderly disabled individuals show **worst baseline fairness** — age + disability compound effects (worse than elderly alone or disabled alone).

**Approach:**
- **Full analysis** for groups with n > 100 (elderly disabled, elderly non-disabled, young non-disabled)
- **Wider CIs** for "Young disabled" (n = 45) using bootstrap

**Intervention:**
1. **Pre-processing:** Reweighting with intersectional stratification (upweight elderly disabled)
2. **In-processing:** Regularization with group-specific penalties (higher penalty for elderly disabled)
3. **Validation:** Bootstrap CIs for young disabled group

**Results:**

| Group | DP (Before → After) | EO (Before → After) | CI Width |
|-------|---------------------|---------------------|----------|
| Elderly disabled | 0.48 → 0.82 | 0.52 → 0.85 | ±0.05 |
| Young disabled | 0.62 → 0.78 | 0.58 → 0.75 | ±0.12 (wide due to small n) |
| Elderly non-disabled | 0.85 → 0.90 | 0.82 → 0.88 | ±0.04 |
| Young non-disabled | 1.00 → 0.91 | 1.00 → 0.93 | ±0.03 |

**Spillover Analysis:**
- Intervention slightly reduced fairness for young non-disabled (DP 1.00 → 0.91)
- **Trade-off accepted:** Prioritized most disadvantaged (elderly disabled) over slight decrease for advantaged group

**Lesson:**
Intersectional analysis revealed that age + disability compound effects. Intervention prioritized the most disadvantaged group (elderly disabled), accepting slight spillover to advantaged group (young non-disabled).

---

## 11. Conclusion

### 11.1 Summary of Intersectionality Requirements

**Key Takeaways:**

1. **Intersectionality is Essential:** Single-axis analysis masks compound discrimination at intersections of multiple protected attributes.

2. **All Intervention Components Must Address Intersectionality:**
   - Causal analysis: Model interaction terms (race × gender)
   - Pre-processing: Disparate impact removal, reweighting for all intersectional groups
   - In-processing: Multi-dimensional fairness constraints (adversarial debiasing, regularization)
   - Post-processing: Group-specific thresholds for all intersections
   - Validation: Disaggregate metrics by intersectional groups

3. **Spillover Effects Must Be Monitored:** Interventions can help one group while harming another — detect and respond.

4. **Sample Size Challenges Require Pragmatism:** Small intersectional subgroups need wider CIs, aggregation, or future data collection.

5. **Stakeholder Engagement is Critical:** Technical metrics must be validated with affected intersectional communities.

### 11.2 Implementation Roadmap

**Phase 1: Scoping (Week 1)**
- [ ] Identify relevant protected attributes (race, gender, age, disability)
- [ ] Define intersectional groups (race × gender combinations)
- [ ] Assess data availability (sample sizes per group)

**Phase 2: Baseline Analysis (Week 2-3)**
- [ ] Construct intersectional causal DAG
- [ ] Measure baseline fairness for all intersectional groups
- [ ] Identify worst-off intersections (e.g., Black women, elderly disabled)

**Phase 3: Intervention (Week 4-6)**
- [ ] Apply pre-processing (disparate impact removal, reweighting)
- [ ] Apply in-processing (adversarial debiasing, regularization)
- [ ] Apply post-processing (threshold optimization)
- [ ] Monitor spillover effects at each stage

**Phase 4: Validation (Week 7-8)**
- [ ] Disaggregate metrics by all intersectional groups
- [ ] Compute bootstrap CIs for small groups
- [ ] Detect and respond to spillover effects
- [ ] Consult intersectional community representatives

**Phase 5: Documentation (Week 9)**
- [ ] Document intersectional findings in fairness report
- [ ] Report aggregation decisions (if any)
- [ ] Acknowledge limitations (small samples, trade-offs)
- [ ] Plan for future data collection (if needed)

---

**End of C5: Intersectionality - Module 2 Intervention Playbook**

---



# C6: Explainability & Transparency

## Overview

**Purpose**: Ensure stakeholders can understand how the intervention works, why decisions are made, and how to interpret results.

**Key Questions**:
- Can participants understand what the intervention does?
- Can decision-makers interpret the results correctly?
- Are the limitations and uncertainties clearly communicated?
- Can the intervention be audited and validated by external parties?

---

## 1. Stakeholder Communication Needs

### 1.1 Audience Mapping

| Stakeholder Group | Information Needs | Technical Level | Communication Channel |
|------------------|-------------------|-----------------|----------------------|
| End users (participants) | What data is collected, how it's used, opt-out options | Non-technical | Plain language notices, FAQs |
| Program administrators | How to interpret results, when to intervene | Semi-technical | Dashboards, training materials |
| Policy makers | Effectiveness, cost-benefit, equity impacts | Non-technical | Executive summaries, visualizations |
| Technical auditors | Model details, validation results, code | Highly technical | Technical documentation, code repos |
| Ethics review boards | Risk assessment, safeguards, monitoring plans | Mixed | Comprehensive reports |
| Media/public | What the program does, evidence of impact | Non-technical | Press releases, public reports |

### 1.2 Transparency Requirements

**Regulatory Compliance**:
- GDPR Article 13-14 (information to data subjects)
- GDPR Article 22 (right to explanation for automated decisions)
- Fair Credit Reporting Act (adverse action notices)
- Institutional Review Board (IRB) informed consent requirements

**Ethical Standards**:
- Belmont Report principles (respect, beneficence, justice)
- APA Ethics Code (informed consent, privacy)
- Professional association guidelines

---

## 2. Model Interpretability

### 2.1 Model Selection for Interpretability

**Inherently Interpretable Models**:
- Linear/logistic regression: Coefficient interpretation
- Decision trees: Rule-based paths
- Rule lists: IF-THEN statements
- Generalized additive models (GAMs): Feature-wise effects

**Trade-offs**:
- Interpretability vs. predictive performance
- Global vs. local explanations
- Simplicity vs. completeness

**Decision Framework**:
```
IF high-stakes decision (e.g., resource allocation, flagging individuals)
  THEN prioritize interpretable models
ELSE IF low-stakes + need high accuracy
  THEN complex model + post-hoc explanations acceptable
ELSE IF exploratory analysis
  THEN use interpretable models for hypothesis generation
```

### 2.2 Post-Hoc Explanation Methods

**Global Explanations** (model-level understanding):

| Method | Use Case | Strengths | Limitations |
|--------|----------|-----------|-------------|
| Feature importance | Identify key predictors | Simple, widely understood | Doesn't show direction or interactions |
| Partial dependence plots | Show feature effects | Visualizes marginal effects | Assumes feature independence |
| SHAP summary plots | Overall feature contributions | Theoretically grounded | Computationally expensive |
| Model distillation | Approximate complex model with simple one | Provides global rules | Approximation may lose fidelity |

**Local Explanations** (individual prediction understanding):

| Method | Use Case | Strengths | Limitations |
|--------|----------|-----------|-------------|
| LIME | Explain individual predictions | Model-agnostic, intuitive | Unstable, requires tuning |
| SHAP values | Feature contributions for instance | Theoretically consistent | Computationally intensive |
| Counterfactual explanations | "What would need to change?" | Actionable insights | May suggest infeasible changes |
| Anchors | Sufficient conditions for prediction | High-precision rules | May be overly specific |

**Implementation Example (SHAP for Intervention Targeting)**:
```python
import shap

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global feature importance
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Individual explanation for a specific participant
participant_idx = 42
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[participant_idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[participant_idx],
        feature_names=X_test.columns
    )
)

# Generate explanation text
def generate_explanation(shap_values, features, feature_names, top_k=3):
    """Create plain language explanation"""
    feature_contributions = sorted(
        zip(feature_names, shap_values, features),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_k]
    
    explanation = "This prediction was primarily influenced by:\n"
    for name, contrib, value in feature_contributions:
        direction = "increased" if contrib > 0 else "decreased"
        explanation += f"- {name} = {value:.2f} ({direction} likelihood by {abs(contrib):.2%})\n"
    
    return explanation
```

### 2.3 Explanation Validation

**Sanity Checks**:
- [ ] Do explanations align with domain knowledge?
- [ ] Are feature importances stable across similar models?
- [ ] Do local explanations aggregate to global patterns?
- [ ] Are explanations consistent for similar individuals?

**User Testing**:
- Conduct think-aloud sessions with stakeholders
- Test comprehension with quiz questions
- Measure confidence calibration (do explanations improve understanding?)
- Assess actionability (can users make better decisions with explanations?)

---

## 3. Transparent Reporting

### 3.1 Intervention Documentation Template

**Executive Summary** (1-2 pages):
- Problem statement and goals
- Intervention description (plain language)
- Key findings and impact metrics
- Limitations and caveats
- Recommendations

**Detailed Report Sections**:

1. **Background & Rationale**
   - Literature review
   - Theory of change
   - Stakeholder engagement process

2. **Intervention Design**
   - Detailed description of components
   - Target population and eligibility
   - Randomization/assignment procedure
   - Timeline and dosage

3. **Data & Methods**
   - Data sources and collection procedures
   - Sample size and power calculations
   - Statistical analysis plan (pre-registered)
   - Model specifications and validation

4. **Results**
   - Descriptive statistics and balance checks
   - Primary outcome analysis
   - Secondary outcomes and subgroup analyses
   - Robustness checks and sensitivity analyses

5. **Fairness & Equity Analysis**
   - Disaggregated results by protected groups
   - Disparate impact assessment
   - Intersectional analysis
   - Equity-adjusted metrics

6. **Limitations**
   - Internal validity threats
   - External validity and generalizability
   - Measurement error and missing data
   - Statistical power limitations

7. **Interpretation & Recommendations**
   - Practical significance vs. statistical significance
   - Cost-benefit analysis
   - Implementation considerations
   - Future research directions

**Appendices**:
- Technical specifications
- Survey instruments
- Code repository links
- Sensitivity analysis details
- IRB approval documentation

### 3.2 Model Cards

**Adapted from Mitchell et al. (2019) for intervention contexts**:

```markdown
# Model Card: Student Success Early Warning System

## Model Details
- **Developed by**: University Data Science Team
- **Date**: January 2024
- **Version**: 2.1
- **Model type**: Gradient Boosted Trees (XGBoost)
- **Purpose**: Identify students at risk of academic difficulty for proactive support

## Intended Use
- **Primary use**: Flag students for academic advisor outreach
- **Intended users**: Academic advisors, student success staff
- **Out-of-scope uses**: Admissions decisions, scholarship allocation, punitive measures

## Training Data
- **Source**: University student records (2018-2023)
- **Size**: 45,000 students, 120 features
- **Demographics**: 52% female, 48% male; 35% underrepresented minorities
- **Temporal coverage**: Fall and spring semesters only (summer excluded)

## Evaluation Data
- **Holdout set**: 10,000 students from 2023-2024
- **Performance metrics**:
  - Overall AUC: 0.82
  - Precision at 20% flag rate: 0.65
  - Recall at 20% flag rate: 0.45
- **Fairness metrics**:
  - False positive rate parity: Max difference 3.2% across racial groups
  - Equalized odds violation: 0.08

## Ethical Considerations
- **Risks**: 
  - False positives may stigmatize students
  - Model may reinforce historical biases in support allocation
  - Privacy concerns with sensitive academic data
- **Mitigation strategies**:
  - Human review required before contact
  - Regular audits for disparate impact
  - Opt-out mechanism for students
  - Transparency about flagging criteria

## Limitations
- Performance degrades for transfer students (limited historical data)
- Does not account for external life circumstances
- May underperform for non-traditional students
- Requires annual retraining due to concept drift

## Maintenance
- **Update frequency**: Annually before fall semester
- **Monitoring**: Quarterly fairness audits, monthly performance checks
- **Contact**: data-ethics@university.edu
```

### 3.3 Data Statements

**Adapted from Bender & Friedman (2018)**:

Key components for intervention data:
- **Curation rationale**: Why this data was collected
- **Language variety**: If text data, what dialects/registers
- **Speaker demographics**: Who provided the data
- **Annotator demographics**: Who labeled/coded the data (if applicable)
- **Speech/text situation**: Context of data collection
- **Preprocessing**: Cleaning, filtering, transformations applied
- **Dataset distribution**: How data is split (train/test/validation)
- **Legal & ethical review**: IRB approval, consent procedures

---

## 4. Algorithmic Transparency

### 4.1 Open Science Practices

**Code & Materials Sharing**:
- [ ] Publish analysis code on GitHub/GitLab with permissive license
- [ ] Include README with setup instructions and dependencies
- [ ] Provide synthetic or anonymized sample data for reproducibility
- [ ] Document data preprocessing pipelines
- [ ] Share statistical analysis plan (SAP) before data analysis

**Pre-registration**:
- [ ] Register study design on OSF, AsPredicted, or ClinicalTrials.gov
- [ ] Specify primary outcomes, sample size, and analysis plan
- [ ] Distinguish confirmatory vs. exploratory analyses in reporting
- [ ] Report deviations from pre-registered plan with justification

**Reproducibility Checklist**:
- [ ] Specify software versions (Python 3.9, scikit-learn 1.0.2, etc.)
- [ ] Use random seeds for stochastic processes
- [ ] Provide environment file (requirements.txt, environment.yml)
- [ ] Include unit tests for key functions
- [ ] Document hardware used for computationally intensive tasks

### 4.2 Algorithm Audits

**Internal Audits**:
- Quarterly reviews by independent internal team
- Checklist-based assessment (bias, performance, drift)
- Red-team exercises (adversarial testing)
- Stakeholder feedback surveys

**External Audits**:
- Third-party fairness audits by domain experts
- Replication studies by independent researchers
- Public bug bounty programs for bias detection
- Regulatory compliance audits (if applicable)

**Audit Report Contents**:
1. Scope and methodology
2. Data quality assessment
3. Performance evaluation (overall and subgroups)
4. Fairness metrics and disparate impact analysis
5. Robustness and security testing
6. Findings and risk assessment
7. Recommendations and remediation plan
8. Follow-up timeline

### 4.3 Transparency Limitations

**When Full Transparency May Be Harmful**:
- **Gaming**: Revealing exact thresholds enables manipulation
- **Privacy**: Detailed model may enable re-identification
- **Security**: Adversaries could exploit known vulnerabilities
- **Proprietary concerns**: Third-party data or algorithms

**Balanced Approach**:
- Provide high-level descriptions without exact parameters
- Use aggregated statistics instead of individual-level data
- Offer explanations to affected individuals without full model disclosure
- Establish trusted third-party audit mechanisms
- Implement differential privacy for data sharing

---

## 5. Participant-Facing Transparency

### 5.1 Informed Consent

**Essential Elements**:
- Purpose of data collection and intervention
- What data is collected and how it's used
- Who has access to the data
- Risks and benefits of participation
- Voluntary nature and right to withdraw
- Alternative options available
- Contact information for questions

**Plain Language Example**:
```
STUDENT SUCCESS PROGRAM NOTIFICATION

What is this program?
We're using information from your academic records to identify students 
who might benefit from extra support. If our system flags you as someone 
who could use help, an advisor will reach out to offer tutoring, study 
groups, or other resources.

What information do we use?
- Your grades and course history
- Class attendance (from learning management system)
- Use of campus resources (library, tutoring center)
- Major and year in school

We do NOT use:
- Race, gender, or ethnicity (except to check for fairness)
- Financial information
- Disciplinary records
- Information from outside the university

What happens if I'm flagged?
An advisor will send you an email offering support services. You can 
choose whether or not to accept. Being flagged does NOT affect your 
grades, financial aid, or academic standing.

Can I opt out?
Yes. Email student-privacy@university.edu to opt out. You'll still have 
access to all support services; we just won't use the automated system 
for you.

Questions?
Contact: student-success@university.edu or (555) 123-4567
```

### 5.2 Explanation Interfaces

**Dashboard Design Principles**:
- Show most important information first
- Use visualizations for complex patterns
- Provide drill-down for details
- Include uncertainty indicators
- Offer contextual help and definitions

**Example: Intervention Assignment Explanation**

```
┌─────────────────────────────────────────────────────────────┐
│  WHY YOU RECEIVED THIS RESOURCE                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  You were selected based on these factors:                  │
│                                                             │
│  ███████████████████░░░░░ Prior program participation       │
│  ████████████░░░░░░░░░░░░ Geographic proximity              │
│  ██████████░░░░░░░░░░░░░░ Expressed interest in survey     │
│  ████░░░░░░░░░░░░░░░░░░░░ Household size                    │
│                                                             │
│  [?] What do these factors mean?                            │
│  [i] How is this decision made?                             │
│  [!] I think this is wrong                                  │
│                                                             │
│  This selection process was designed to prioritize people   │
│  most likely to benefit. It was reviewed for fairness       │
│  across demographic groups.                                 │
│                                                             │
│  Not selected? You can still access [alternative options]  │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Recourse Mechanisms

**Right to Challenge**:
- Clear process for disputing decisions
- Human review of contested cases
- Timely response requirements (e.g., 10 business days)
- Appeal to independent body if needed

**Feedback Loops**:
- Mechanism for reporting errors in data
- Process for updating information
- Notification of changes to decisions
- Aggregate feedback used for system improvement

---

## 6. Communicating Uncertainty

### 6.1 Expressing Confidence Appropriately

**Statistical Significance vs. Practical Significance**:
- Report effect sizes with confidence intervals, not just p-values
- Discuss practical importance in context
- Acknowledge when results are uncertain or borderline

**Example Framings**:

❌ **Poor**: "The intervention significantly improved outcomes (p < 0.05)."

✅ **Better**: "The intervention improved outcomes by 0.3 standard deviations (95% CI: 0.1 to 0.5), which is considered a small to moderate effect. This translates to approximately 5% more participants achieving the goal."

❌ **Poor**: "The model is 85% accurate."

✅ **Better**: "The model correctly identifies 85% of cases overall, but accuracy varies by subgroup (78%-91%). For every 100 predictions, we expect 15 errors on average, though this could range from 10-20 based on the population."

### 6.2 Visualizing Uncertainty

**Confidence Intervals**:
```python
import matplotlib.pyplot as plt
import numpy as np

groups = ['Group A', 'Group B', 'Group C']
effects = [0.25, 0.18, 0.32]
ci_lower = [0.10, 0.05, 0.15]
ci_upper = [0.40, 0.31, 0.49]
errors = [[effects[i] - ci_lower[i], ci_upper[i] - effects[i]] 
          for i in range(len(groups))]

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(groups, effects, yerr=np.array(errors).T, 
            fmt='o', capsize=5, capthick=2, markersize=8)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('Effect Size (Cohen\'s d)')
ax.set_title('Intervention Effects by Group (95% Confidence Intervals)')
ax.set_ylim(-0.1, 0.6)
plt.tight_layout()
plt.savefig('effect_sizes_with_uncertainty.png', dpi=300)
```

**Prediction Intervals for Forecasts**:
- Show range of plausible outcomes, not just point estimates
- Use fan charts for time series
- Annotate with probability levels (50%, 80%, 95%)

### 6.3 Limitations Statements

**Template for Limitations Section**:

```markdown
## Limitations and Caveats

### Internal Validity
- [Threat]: [Description of how it might affect results]
- [Mitigation]: [What we did to address it]
- [Residual risk]: [Remaining concerns]

Example:
- Threat: Differential attrition (12% in treatment vs. 8% in control)
- Mitigation: Conducted bounds analysis and attrition weighting
- Residual risk: If dropouts differ on unmeasured characteristics, 
  effects may be biased. Bounds suggest effect between 0.15-0.35.

### External Validity
- [Population]: Results may not generalize to [other populations] because 
  [reasons]
- [Setting]: Intervention tested in [context]; different settings may 
  produce different results
- [Time]: Study conducted in [period]; secular trends may affect 
  replicability

### Measurement
- [Outcome measure]: [Limitations of measure, e.g., self-report bias]
- [Construct validity]: [Whether measure captures intended construct]
- [Missing data]: [Extent and pattern of missingness]

### Statistical Power
- Minimum detectable effect: [X]
- Subgroup analyses: Powered only to detect effects of [Y] or larger
- Multiple comparisons: [Adjustment approach or acknowledgment]

### Fairness Analysis
- Sample sizes for [group] too small for precise estimates
- Intersectional analyses limited by [constraint]
- Fairness metrics may trade off; we prioritized [metric] because [reason]
```

---

## 7. Training & Capacity Building

### 7.1 Stakeholder Training Programs

**For Program Administrators**:

**Module 1: Understanding the System** (2 hours)
- How the intervention works (high-level)
- What data is used and why
- How decisions are made
- Limitations and uncertainties

**Module 2: Interpreting Results** (2 hours)
- Reading dashboards and reports
- Understanding statistical significance
- Recognizing when to seek expert help
- Case studies and examples

**Module 3: Ethical Use** (1.5 hours)
- Privacy and confidentiality
- Avoiding misuse and overreliance
- Equity considerations
- Handling participant questions

**Module 4: Practical Application** (2.5 hours)
- Hands-on exercises with real (anonymized) data
- Decision-making scenarios
- Troubleshooting common issues
- Q&A with technical team

**For Technical Staff**:

- Bias testing and fairness evaluation
- Model monitoring and maintenance
- Incident response procedures
- Documentation best practices

### 7.2 Decision-Support Tools

**Checklists for Key Decisions**:

```markdown
## Before Deploying Model

- [ ] Validated on holdout data from deployment period
- [ ] Fairness metrics meet predetermined thresholds
- [ ] Explanation system tested with end users
- [ ] Monitoring infrastructure in place
- [ ] Incident response plan documented
- [ ] Stakeholder training completed
- [ ] Legal/compliance review approved
- [ ] Ethics review board approval (if required)
- [ ] Pilot test conducted with small sample
- [ ] Rollback plan prepared

## Monthly Monitoring Review

- [ ] Performance metrics within acceptable range
- [ ] No significant distribution shifts detected
- [ ] Fairness metrics remain balanced
- [ ] User feedback reviewed and addressed
- [ ] No critical incidents reported
- [ ] Documentation updated for any changes
- [ ] Stakeholder communication sent (if needed)
```

**Decision Trees for Common Scenarios**:

```
Participant questions model decision:
│
├─ Is it about data accuracy?
│  ├─ YES → Verify data, correct if needed, re-run
│  └─ NO → Continue
│
├─ Is it about eligibility criteria?
│  ├─ YES → Explain criteria, check for errors
│  └─ NO → Continue
│
├─ Is it about fairness/discrimination?
│  ├─ YES → Escalate to fairness team, review case
│  └─ NO → Continue
│
└─ General explanation request?
   └─ Provide standard explanation, offer human consultation
```

---

## 8. Transparency Governance

### 8.1 Roles & Responsibilities

| Role | Responsibilities | Qualifications |
|------|------------------|----------------|
| **Transparency Officer** | Oversee transparency practices, coordinate reporting | Ethics background, communication skills |
| **Technical Documentation Lead** | Maintain technical docs, model cards | Senior data scientist |
| **Plain Language Writer** | Translate technical content for public | Science communication experience |
| **Stakeholder Liaison** | Gather feedback, facilitate training | Domain expertise, people skills |
| **Legal Compliance Officer** | Ensure regulatory compliance | Legal expertise in data/privacy law |
| **Audit Coordinator** | Organize internal/external audits | Project management, technical literacy |

### 8.2 Transparency Review Process

**Quarterly Transparency Audit**:

1. **Documentation Review** (Week 1)
   - Check all docs are current
   - Verify technical accuracy
   - Assess readability for target audiences

2. **Stakeholder Feedback** (Week 2)
   - Survey users on clarity and usefulness
   - Conduct focus groups with key stakeholders
   - Review support tickets for confusion patterns

3. **Gap Analysis** (Week 3)
   - Identify information needs not being met
   - Assess compliance with transparency standards
   - Benchmark against best practices

4. **Improvement Planning** (Week 4)
   - Prioritize transparency enhancements
   - Assign responsibilities and timelines
   - Update transparency roadmap

### 8.3 Transparency Metrics

**Quantitative Indicators**:
- % of stakeholders who complete transparency training
- Average time to respond to explanation requests
- User comprehension scores on post-training assessments
- % of documentation updated within SLA (e.g., 30 days of changes)
- Number of transparency-related complaints or concerns

**Qualitative Indicators**:
- Stakeholder satisfaction with explanations (survey ratings)
- Quality of external audit reports
- Media coverage sentiment regarding transparency
- Regulator feedback on compliance

**Target Setting**:
- Establish baseline in first quarter
- Set improvement targets (e.g., +10% comprehension annually)
- Review and adjust targets annually

---

## 9. Implementation Checklist

### Phase 1: Planning (Weeks 1-2)
- [ ] Map stakeholder groups and information needs
- [ ] Identify regulatory and ethical transparency requirements
- [ ] Establish transparency governance structure
- [ ] Define transparency metrics and targets

### Phase 2: Documentation (Weeks 3-5)
- [ ] Create model cards and data statements
- [ ] Write technical documentation
- [ ] Develop plain language summaries
- [ ] Draft informed consent materials
- [ ] Prepare explanation templates

### Phase 3: Explanation Systems (Weeks 6-8)
- [ ] Implement interpretability methods (SHAP, LIME, etc.)
- [ ] Build explanation interfaces/dashboards
- [ ] Validate explanations with domain experts
- [ ] User-test explanations with representative stakeholders

### Phase 4: Training & Rollout (Weeks 9-11)
- [ ] Develop training materials for all stakeholder groups
- [ ] Conduct training sessions
- [ ] Deploy explanation systems
- [ ] Launch transparency portal/documentation site
- [ ] Establish support channels for questions

### Phase 5: Monitoring & Iteration (Week 12+)
- [ ] Collect stakeholder feedback continuously
- [ ] Monitor transparency metrics
- [ ] Conduct quarterly transparency audits
- [ ] Update documentation as system evolves
- [ ] Report transparency improvements to leadership

---

**End of C6: Explainability & Transparency - Module 2 Intervention Playbook**

---

# C7: Accountability & Governance

## Overview

**Purpose**: Establish clear lines of responsibility, oversight mechanisms, and processes for addressing harms that arise from the intervention.

**Key Questions**:
- Who is responsible when something goes wrong?
- How are decisions escalated and resolved?
- What mechanisms exist for redress and remedy?
- How is the intervention governed over its lifecycle?

---

## 1. Accountability Framework

### 1.1 Responsibility Mapping

**RACI Matrix for Intervention Lifecycle**:

| Activity | Responsible | Accountable | Consulted | Informed |
|----------|------------|-------------|-----------|----------|
| Study design | Research Lead | PI/Director | Ethics Board, Stakeholders | Funders |
| Data collection | Data Team | Research Lead | Legal, Privacy Officer | Participants |
| Model development | Data Scientists | Tech Lead | Domain Experts | Research Lead |
| Fairness testing | Fairness Specialist | Tech Lead | Community Reps | Ethics Board |
| Deployment decision | PI/Director | Executive Sponsor | All stakeholders | Participants |
| Monitoring | Ops Team | Tech Lead | Research Lead | PI/Director |
| Incident response | Incident Manager | PI/Director | Legal, PR, Ethics | All affected |
| Reporting | Research Lead | PI/Director | Co-authors | Funders, Public |

**Key Definitions**:
- **Responsible**: Does the work
- **Accountable**: Ultimately answerable (only one person)
- **Consulted**: Provides input
- **Informed**: Kept up-to-date

### 1.2 Decision Authority Levels

**Level 1: Operational** (day-to-day decisions)
- Minor parameter adjustments within approved ranges
- Routine monitoring and reporting
- Standard support requests
- **Authority**: Technical staff
- **Documentation**: Activity logs

**Level 2: Tactical** (significant changes)
- Model retraining with new data
- Changes to eligibility criteria
- Response to performance degradation
- **Authority**: Technical lead + Research lead
- **Documentation**: Change request form, approval record

**Level 3: Strategic** (major decisions)
- Deployment of new intervention
- Significant design changes
- Response to serious incidents
- **Authority**: PI/Director + Ethics board
- **Documentation**: Formal proposal, ethics review, executive approval

**Level 4: Governance** (oversight and policy)
- Intervention termination
- Policy changes affecting multiple projects
- Response to regulatory action
- **Authority**: Executive sponsor + Governance board
- **Documentation**: Board minutes, formal resolutions

### 1.3 Accountability for Harms

**Types of Harms**:

| Harm Type | Examples | Responsible Party | Remediation |
|-----------|----------|-------------------|-------------|
| **Privacy breach** | Unauthorized data access | Data custodian, Security team | Notification, credit monitoring, policy changes |
| **Discriminatory impact** | Disparate outcomes by race/gender | Research team, Fairness lead | Intervention adjustment, affected group support |
| **Psychological harm** | Stigma from flagging/labeling | Research team, Ethics board | Counseling, communication changes |
| **Opportunity harm** | Denied benefits due to error | Program administrator | Retroactive eligibility, compensation |
| **Dignitary harm** | Lack of respect, voice | All team members | Apology, process changes, community engagement |
| **Autonomy harm** | Coercive nudges, limited choice | Research lead, Ethics board | Opt-out mechanisms, choice architecture review |

**Harm Response Protocol**:

1. **Immediate**: Stop/pause intervention if serious harm
2. **Investigate**: Root cause analysis within 48 hours
3. **Remedy**: Provide redress to affected individuals
4. **Report**: Notify ethics board, regulators (if required)
5. **Prevent**: Implement safeguards against recurrence
6. **Document**: Full incident report and lessons learned

---

## 2. Governance Structures

### 2.1 Ethics Review Boards

**Institutional Review Board (IRB)** for human subjects research:
- **Composition**: Scientists, ethicists, community members, legal experts
- **Scope**: Approves study before launch, reviews modifications, monitors ongoing studies
- **Review criteria**: Risk-benefit balance, informed consent, privacy protections, equitable selection
- **Meeting frequency**: Monthly or as needed for expedited reviews

**Algorithm Ethics Committee** (for non-research interventions):
- **Composition**: Technical experts, domain specialists, ethicists, affected community representatives
- **Scope**: Reviews algorithmic systems for fairness, transparency, accountability
- **Review criteria**: Fairness metrics, explanation quality, governance adequacy, stakeholder input
- **Meeting frequency**: Quarterly reviews + ad-hoc for new systems

**Community Advisory Board**:
- **Composition**: Representatives from affected communities
- **Scope**: Provides input on intervention design, interprets findings, advises on implementation
- **Compensation**: Stipends for time and expertise
- **Meeting frequency**: Monthly during active intervention

### 2.2 Governance Models

**Centralized Model**:
- Single oversight body approves all decisions
- **Pros**: Consistency, clear authority, efficient
- **Cons**: Bottleneck, may lack domain expertise, slow

**Federated Model**:
- Multiple domain-specific committees with coordination
- **Pros**: Specialized expertise, scalable, faster
- **Cons**: Potential inconsistency, coordination overhead

**Hybrid Model** (Recommended):
- Domain committees for technical decisions
- Central ethics board for high-stakes/cross-cutting issues
- Clear escalation paths between levels

**Example Structure**:
```
                    ┌─────────────────────────┐
                    │  Executive Sponsor      │
                    │  (Strategic oversight)  │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │  Ethics Review Board    │
                    │  (Policy & high-stakes) │
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
┌───────┴────────┐   ┌──────────┴──────────┐   ┌────────┴────────┐
│ Technical      │   │ Fairness            │   │ Community       │
│ Review         │