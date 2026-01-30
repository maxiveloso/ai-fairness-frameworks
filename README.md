# AI Fairness Frameworks

![Case Studies](https://img.shields.io/badge/Case%20Studies-3-blue)
![Techniques](https://img.shields.io/badge/Techniques-74%20Implementations-green)
![Validated](https://img.shields.io/badge/Results-Statistically%20Validated-success)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Citations](https://img.shields.io/badge/Academic%20Citations-50+-orange)

> **Evidence-based fairness interventions with measurable results.**
> From bias diagnosis to organizational transformation.

---

## The Fairness Maturity Model

Organizations progress through three phases when implementing AI fairness:

```
[Phase 1: Assessment]  -->  [Phase 2: Intervention]  -->  [Phase 3: Governance]
        |                           |                            |
   Diagnosis                    Treatment                 Transformation
   "What bias exists?"      "How do we fix it?"      "How do we scale it?"
        |                           |                            |
   Meridian Financial         MidCity Bank                  EquiHire
     (Lending)                 (Lending)                 (Recruitment)
```

---

## Headline Results

| Case Study | Domain | Demographic Parity | Accuracy Impact | Validation |
|------------|--------|-------------------|-----------------|------------|
| **MidCity Bank** | Lending | 0.27 -> 0.90 **(+231%)** | -0.7% | p < 0.001 |
| **EquiHire** | Recruitment | 0.18 -> 0.37 **(+109%)** | -1.9% | p = 0.0002 |

All results from **real execution** on synthetic datasets with **statistical validation**.

---

## Case Studies

### [01. Assessment: Meridian Financial Services](./case-studies/01-assessment-lending/)
**Phase**: Diagnosis | **Domain**: Consumer Lending

Comprehensive fairness audit framework for a loan approval AI system processing 2,000+ daily applications. Established baseline measurements and identified 34% approval rate disparities.

**Deliverables**: Audit framework, metrics definitions, bias source identification, intersectionality analysis

---

### [02. Intervention: MidCity Community Bank](./case-studies/02-intervention-lending/)
**Phase**: Treatment | **Domain**: Consumer Lending

Complete fairness intervention pipeline executed on 50,000 historical loan applications. Applied pre-processing, in-processing, and post-processing techniques with measured outcomes.

**Key Results**:
- Demographic Parity: 0.27 -> 0.90 (**+231%**)
- Equal Opportunity: +32% improvement
- Accuracy Cost: Only **-0.7%**

**Techniques Applied**: Disparate Impact Remover, Adversarial Debiasing, Equalized Odds Threshold

[View Case Study ->](./case-studies/02-intervention-lending/)

---

### [03. Governance: EquiHire Recruitment Platform](./case-studies/03-governance-recruitment/)
**Phase**: Transformation | **Domain**: AI Recruitment (EU SaaS)

Enterprise-scale fairness implementation for a multi-team AI recruitment platform. Includes governance frameworks, CEO presentation, and organizational change management.

**Key Results**:
- Demographic Parity: 0.18 -> 0.37 (**+109%**)
- Equal Opportunity: +36.6% improvement
- Accuracy Cost: Only **-1.9%**
- Business Impact: Retained $2.3M at-risk contract

[View Case Study ->](./case-studies/03-governance-recruitment/)

---

## Technique Library

**[74 Production-Ready Implementations](./techniques/)** across 5 intervention categories:

| Category | Count | Purpose | Key Techniques |
|----------|-------|---------|----------------|
| [Pre-Processing](./techniques/pre_processing/) | 11 | Transform data before training | Disparate Impact Remover, Reweighting, SMOTE |
| [In-Processing](./techniques/in_processing/) | 16 | Apply constraints during training | Adversarial Debiasing, Prejudice Remover, Exponentiated Gradient |
| [Post-Processing](./techniques/post_processing/) | 7 | Calibrate outputs after training | Equalized Odds, Reject Option, Calibration |
| [Causal](./techniques/causal/) | 17 | Causal inference for fairness | Counterfactual Fairness, Path-Specific Effects |
| [Validation](./techniques/validation/) | 23 | Statistical testing and metrics | Permutation Tests, Bootstrap CIs |

Each implementation includes:
- Complete Python code with academic citation
- Usage documentation and integration examples
- Tested on real case studies with measured outcomes

[Browse All Techniques ->](./techniques/)

---

## Methodology

### 2-Tier RAG System

Our approach uses a systematic Retrieval-Augmented Generation pipeline:

1. **Tier 1 (Discovery)**: Semantic search across 74 fairness techniques and academic literature
2. **Tier 2 (Synthesis)**: LLM-powered consolidation into actionable deliverables
3. **Execution (Validation)**: Real technique application with statistical validation

[Read Full Methodology ->](./methodology/)

### Fairness Metrics

| Metric | Definition | Use Case |
|--------|------------|----------|
| **Demographic Parity** | P(Y=1\|A=0) / P(Y=1\|A=1) | Equal selection rates across groups |
| **Equal Opportunity** | TPR ratio across protected groups | Equal true positive rates |
| **Equalized Odds** | Both TPR and FPR parity | Balance between fairness and accuracy |

[View All Metrics ->](./methodology/fairness-metrics.md)

---

## Services

### Phase 1: Assessment & Audit
Comprehensive bias diagnosis including historical context analysis, fairness metrics framework, and bias source identification.

[Learn More ->](./services/assessment-audit.md)

### Phase 2: Intervention & Implementation
Evidence-based fairness interventions with technique selection, implementation, and measurable outcomes.

[Learn More ->](./services/intervention-implementation.md)

### Phase 3: Governance & Scale
Enterprise-wide fairness programs with governance frameworks, executive communication, and organizational change management.

[Learn More ->](./services/governance-scale.md)

---

## Repository Structure

```
ai-fairness-frameworks/
├── techniques/                     # 74 production-ready implementations
│   ├── pre_processing/             # 11 data transformation techniques
│   ├── in_processing/              # 16 training constraint techniques
│   ├── post_processing/            # 7 output calibration techniques
│   ├── causal/                     # 17 causal inference techniques
│   └── validation/                 # 23 statistical testing techniques
├── case-studies/
│   ├── 01-assessment-lending/      # Meridian Financial - Diagnosis
│   ├── 02-intervention-lending/    # MidCity Bank - Treatment (+231% DP)
│   └── 03-governance-recruitment/  # EquiHire - Transformation (+109% DP)
├── methodology/
│   ├── rag-system.md               # 2-tier RAG architecture
│   ├── fairness-metrics.md         # Metric definitions
│   └── validation-framework.md     # Statistical validation
├── services/
│   ├── assessment-audit.md         # Phase 1 service
│   ├── intervention-implementation.md  # Phase 2 service
│   └── governance-scale.md         # Phase 3 service
└── data/
    ├── loan-approval/              # Lending datasets
    └── recruitment/                # Recruitment datasets
```

---

## What Makes This Different

| Aspect | This Framework | Typical Approaches |
|--------|----------------|-------------------|
| **Techniques** | 74 production-ready implementations | Conceptual descriptions |
| **Results** | Real execution with measured outcomes | Theoretical recommendations |
| **Validation** | Permutation tests, bootstrap CIs | Anecdotal or none |
| **Trade-offs** | Exact fairness vs. accuracy costs | Vague "minimal impact" |
| **Reproducibility** | Code, data, and metrics provided | Black box |
| **Scale** | From team practices to board presentations | Single-level focus |

---

## Technology Stack

`Python` `scikit-learn` `XGBoost` `AIF360` `Fairlearn` `Supabase` `pgvector` `Claude`

---

## Contact

For consulting inquiries or questions about implementing these frameworks in your organization:

**Maximiliano Veloso**
- GitHub: [@maxiveloso](https://github.com/maxiveloso)
- LinkedIn: [Maximiliano Veloso](https://linkedin.com/in/maxiveloso)

---

*Evidence-based AI fairness consulting - from diagnosis to transformation.*
