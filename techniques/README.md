# Fairness Technique Library

**74 Production-Ready Implementations** across 5 intervention categories.

Each technique includes:
- Complete Python implementation
- Academic citation
- Usage documentation
- Integration examples

---

## Technique Categories

| Category | Count | Purpose |
|----------|-------|---------|
| [Pre-Processing](./pre_processing/) | 11 | Transform data before training |
| [In-Processing](./in_processing/) | 16 | Apply constraints during training |
| [Post-Processing](./post_processing/) | 7 | Calibrate outputs after training |
| [Causal](./causal/) | 17 | Causal inference for fairness |
| [Validation](./validation/) | 23 | Statistical testing and metrics |

---

## Pre-Processing Techniques

Transform training data to reduce bias before model training.

| Technique | Citation | Description |
|-----------|----------|-------------|
| Disparate Impact Remover | Feldman et al., 2015 | Remove correlation with protected attributes |
| Reweighting | Kamiran & Calders, 2012 | Adjust sample weights for balance |
| Massaging | Kamiran & Calders, 2009 | Strategic label flipping |
| Uniform Sampling | - | Balance representation across groups |
| SMOTE for Fairness | Chawla et al., 2002 | Synthetic minority oversampling |

**When to use**: When bias originates in training data and you have control over data preparation.

---

## In-Processing Techniques

Apply fairness constraints during model training.

| Technique | Citation | Description |
|-----------|----------|-------------|
| Adversarial Debiasing | Zhang et al., 2018 | Learn representations that hide protected attributes |
| Prejudice Remover | Kamishima et al., 2012 | Add fairness regularization term |
| Exponentiated Gradient | Agarwal et al., 2018 | Constrained optimization for fairness |
| Meta Fair Classifier | Celis et al., 2019 | Meta-learning for fair predictions |
| Reductions Approach | Agarwal et al., 2019 | Reduce fairness to cost-sensitive classification |

**When to use**: When you can modify the training process and need integrated fairness optimization.

---

## Post-Processing Techniques

Adjust model outputs to improve fairness after training.

| Technique | Citation | Description |
|-----------|----------|-------------|
| Equalized Odds | Hardt et al., 2016 | Calibrate thresholds per group |
| Calibrated Equalized Odds | Pleiss et al., 2017 | Preserve calibration while achieving fairness |
| Reject Option Classification | Kamiran et al., 2012 | Give favorable outcomes to uncertain cases |
| Platt Scaling | Platt, 1999 | Probability calibration |
| Isotonic Regression | Zadrozny & Elkan, 2002 | Non-parametric calibration |

**When to use**: When you cannot modify the model but can adjust predictions.

---

## Causal Techniques

Apply causal inference methods to understand and address bias.

| Technique | Citation | Description |
|-----------|----------|-------------|
| Counterfactual Fairness | Kusner et al., 2017 | Fair under counterfactual interventions |
| Path-Specific Effects | Chiappa, 2019 | Decompose causal paths |
| Causal Graph Construction | Pearl, 2009 | Model causal relationships |
| Instrumental Variables | Angrist & Imbens, 2008 | Handle unobserved confounding |
| Sensitivity Analysis | Wu et al., 2019 | Test robustness to assumptions |

**When to use**: When you need to understand *why* bias occurs, not just detect it.

---

## Validation Techniques

Statistical methods to measure and validate fairness.

| Technique | Citation | Description |
|-----------|----------|-------------|
| Permutation Testing | - | Statistical significance testing |
| Bootstrap Confidence Intervals | Efron, 1979 | Uncertainty quantification |
| Cross-Validation for Fairness | - | Robustness across data splits |
| Disparate Impact Ratio | Feldman et al., 2015 | Legal compliance metric |
| Equalized Odds Difference | Hardt et al., 2016 | Multi-metric fairness score |

**When to use**: Always - validation is essential for any fairness intervention.

---

## File Naming Convention

```
M2-{Source}-{Paper}-{Author}-{Year}-{TechniqueName}.py
```

Example: `M2-S1-P3-Feldman-2015-DisparateImpactRemoval.py`

- `M2`: Module 2 origin
- `S1`: Source category
- `P3`: Paper reference
- `Feldman-2015`: First author and year
- `DisparateImpactRemoval`: Technique name

---

## Usage Example

```python
from techniques.pre_processing import DisparateImpactRemover

# Load your data
X_train, y_train, protected_attrs = load_data()

# Apply technique
dir = DisparateImpactRemover(repair_level=0.8)
X_fair = dir.fit_transform(X_train, protected_attrs)

# Train model on fair data
model.fit(X_fair, y_train)
```

---

## Integration with Case Studies

These techniques were applied in our case studies:

| Case Study | Techniques Used | Result |
|------------|-----------------|--------|
| [MidCity Bank](../case-studies/02-intervention-lending/) | DIR, Adversarial Debiasing, EO Threshold | +231% DP |
| [EquiHire](../case-studies/03-governance-recruitment/) | DIR, Adversarial Debiasing, EO Threshold | +109% DP |

---

## Academic References

All implementations are based on peer-reviewed research. Key references:

1. Feldman et al. (2015). "Certifying and Removing Disparate Impact"
2. Zhang et al. (2018). "Mitigating Unwanted Biases with Adversarial Learning"
3. Hardt et al. (2016). "Equality of Opportunity in Supervised Learning"
4. Kusner et al. (2017). "Counterfactual Fairness"
5. Agarwal et al. (2018). "A Reductions Approach to Fair Classification"

---

## Navigation

- [Back to Main Portfolio](../)
- [Methodology](../methodology/)
- [Case Studies](../case-studies/)
