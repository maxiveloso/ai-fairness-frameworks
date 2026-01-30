# Case Study: EquiHire Recruitment Platform

**Phase**: Organizational Scale & Governance (Transformation)
**Domain**: AI Recruitment (EU SaaS)
**Result**: +109% Demographic Parity, $2.3M Contract Retained

---

## Executive Summary

EquiHire, an EU-based AI recruitment platform, implemented enterprise-wide fairness governance across multiple ML teams. This case study demonstrates how to scale fairness interventions from individual systems to organizational transformation, including executive communication and regulatory compliance.

---

## Headline Results

| Metric | Baseline | After Intervention | Change |
|--------|----------|-------------------|--------|
| Demographic Parity | 0.177 | 0.371 | **+109.5%** |
| Equal Opportunity | 0.725 | 0.990 | **+36.6%** |
| Accuracy | 92.2% | 90.4% | **-1.9%** |
| Statistical Significance | - | p = 0.0002 | Validated |

### Business Impact

| Metric | Value |
|--------|-------|
| At-risk contract retained | $2.3M |
| Upsell opportunity | 15% |
| Implementation cost | $40K |
| Cost-benefit ratio | **57.65:1** |

---

## Client Context

| Attribute | Detail |
|-----------|--------|
| Industry | AI Recruitment Platform (SaaS) |
| Region | European Union (EU AI Act compliance) |
| Scale | Multi-team (3 ML teams + compliance + product) |
| System | CV Screening + Candidate Matching AI |
| Protected Attributes | Race, Gender |
| Risk Classification | High-risk AI system under EU AI Act |

---

## Challenge

EquiHire faced a complex challenge requiring organizational transformation:

1. **Technical**: Black candidates hired at 17.7% the rate of White candidates
2. **Organizational**: Multiple ML teams with inconsistent practices
3. **Regulatory**: EU AI Act compliance requirements
4. **Commercial**: Enterprise client threatening contract cancellation

---

## Approach

### Integration Framework

Connected four fairness components across the organization:

1. **Fair AI Scrum Toolkit** - Team-level practices
2. **Organizational Integration Toolkit** - Governance frameworks
3. **Advanced Architecture Cookbook** - Specialized strategies
4. **Regulatory Compliance Guide** - EU AI Act alignment

### Three-Stage Intervention

| Stage | Technique | Result |
|-------|-----------|--------|
| Pre-processing | Disparate Impact Remover | DP: 0.18 -> 0.62 |
| In-processing | Adversarial Debiasing | DP: 0.62 -> 0.55 (overshoot correction) |
| Post-processing | Equalized Odds Threshold | DP: 0.55 -> 0.37 (stabilized) |

### Governance Implementation

- AI Ethics Committee established (monthly reviews)
- Fairness gates integrated into CI/CD pipeline
- RACI matrix for fairness responsibilities
- Executive dashboard for ongoing monitoring

---

## Results by Protected Attribute

### Race-Based Hiring Rates

| Group | Baseline | After Intervention | Change |
|-------|----------|-------------------|--------|
| White | 31.9% | 27.0% | -4.9pp |
| Asian | 37.0% | 22.9% | -14.1pp |
| Hispanic | 10.1% | 18.2% | +8.1pp |
| Black | 5.6% | 10.0% | +4.4pp |

### Intersectional Analysis

| Group | Baseline | After | Change |
|-------|----------|-------|--------|
| White Male | 39.9% | 34.1% | -5.8pp |
| Black Female | 3.4% | 4.8% | +1.4pp |
| Asian Male | 47.2% | 32.1% | -15.1pp |

---

## Execution Evidence

Raw metrics available in [execution-results/](./execution-results/):

- `baseline_results.json` - Pre-intervention by group
- `intervention_results.json` - Post-intervention by stage
- `validation_results.json` - Statistical validation

---

## Deliverables

| Document | Description |
|----------|-------------|
| [Introduction](./deliverables/00-introduction.md) | Playbook overview |
| [Integration Framework](./deliverables/01-integration-framework.md) | Component integration |
| [Implementation Guide](./deliverables/02-implementation-guide.md) | Organizational rollout |
| [Case Study](./deliverables/03-case-study.md) | Complete narrative |
| [Validation Framework](./deliverables/04-validation-framework.md) | Enterprise testing |
| [Adaptability Guidelines](./deliverables/05-adaptability-guidelines.md) | Domain adaptation |
| [Future Improvements](./deliverables/06-future-improvements.md) | Enhancement roadmap |
| [Gap Analysis](./deliverables/07-gap-analysis.md) | Coverage assessment |
| [**CEO Presentation**](./deliverables/08-ceo-presentation.md) | Board-ready briefing |

---

## CEO Presentation Highlights

The [CEO Presentation](./deliverables/08-ceo-presentation.md) includes:

- **Problem Statement**: Clear business risk articulation
- **Solution Overview**: Technical approach in business terms
- **ROI Analysis**: 57.65:1 cost-benefit ratio
- **Risk Assessment**: Regulatory, reputational, commercial
- **Board-Level Asks**: Budget, governance, timeline
- **Success Metrics**: KPIs for ongoing monitoring

---

## Key Takeaways

1. **Organizational change required**: Technical fixes alone insufficient
2. **Governance enables scale**: Consistent practices across teams
3. **Executive buy-in critical**: CEO presentation secured budget
4. **Regulatory alignment**: EU AI Act compliance achieved
5. **Business case clear**: 57:1 ROI justifies investment

---

## Navigation

- [Back to Case Studies](../)
- [Previous: Intervention Case Study](../02-intervention-lending/)
- [View CEO Presentation](./deliverables/08-ceo-presentation.md)
- [View Execution Results](./execution-results/)
