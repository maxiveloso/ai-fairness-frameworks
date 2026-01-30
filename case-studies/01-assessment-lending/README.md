# Case Study: Meridian Financial Services

**Phase**: Assessment & Audit (Diagnosis)
**Domain**: Consumer Lending
**System**: SmartLend AI Loan Approval

---

## Executive Summary

Meridian Financial Services, a mid-size consumer lender processing 2,000+ daily loan applications, engaged us to conduct a comprehensive fairness audit of their AI-powered loan approval system ("SmartLend") before implementing any bias mitigation interventions.

---

## Client Context

| Attribute | Detail |
|-----------|--------|
| Industry | Consumer Lending |
| Scale | 850,000+ historical loan records |
| Daily Volume | ~2,000 applications |
| System | SmartLend (XGBoost-based) |
| Protected Attributes | Race, Gender, Age |

---

## Challenge

Meridian received regulatory inquiries about potential disparate impact in their loan approval process. Before implementing fixes, they needed to:

1. **Understand the scope** of any existing bias
2. **Identify root causes** in their data and model
3. **Establish metrics** for measuring progress
4. **Document findings** for regulatory compliance

---

## Approach

### 1. Historical Context Analysis
- Reviewed lending practices and regulatory history
- Identified potential sources of historical discrimination
- Mapped data lineage to bias entry points

### 2. Fairness Metrics Framework
- Established appropriate metrics (demographic parity, equal opportunity, equalized odds)
- Set baseline measurements
- Defined acceptable thresholds

### 3. Bias Source Identification
- Analyzed 147 features for proxy discrimination
- Evaluated training data representativeness
- Assessed model architecture decisions

### 4. Intersectionality Analysis
- Examined compound effects (race + gender, race + age)
- Identified particularly disadvantaged subgroups
- Prioritized intervention targets

---

## Key Findings

### Approval Rate Disparities
- **34% difference** between highest and lowest demographic groups
- Gender disparities compounded by age in certain categories
- Geographic features acting as proxies for protected attributes

### Bias Entry Points
1. **Historical Labels**: Past approvals reflected discriminatory practices
2. **Feature Engineering**: Zip code-derived features correlated with race
3. **Sample Bias**: Underrepresentation of minority applicants in training data

### Intersectional Effects
- Black women over 50: Highest denial rates
- Young Hispanic males: Steepest income-to-approval slope
- Asian applicants: Lower disparities but still significant

---

## Deliverables

| Document | Description |
|----------|-------------|
| [Audit Framework](./deliverables/implementation-guide.md) | Complete audit methodology |
| [Metrics Framework](./deliverables/validation-framework.md) | Fairness metrics definitions and thresholds |
| [Intersectionality Guide](./deliverables/intersectionality-guide.md) | Multi-attribute analysis framework |
| [Implementation Checklist](./deliverables/implementation-checklist.md) | Step-by-step audit process |
| [Example Audit](./deliverables/example-audit.md) | Worked example with real data |

---

## Recommendations Generated

1. **Immediate**: Remove or transform zip code-derived features
2. **Short-term**: Rebalance training data with synthetic oversampling
3. **Medium-term**: Implement in-processing constraints during model training
4. **Long-term**: Establish ongoing monitoring and governance framework

---

## Impact

This assessment phase established the foundation for subsequent intervention work (see [MidCity Bank Case Study](../02-intervention-lending/)), providing:

- Clear baseline metrics for measuring improvement
- Prioritized list of bias sources to address
- Regulatory-ready documentation
- Stakeholder alignment on fairness goals

---

## Navigation

- [Back to Case Studies](../)
- [Next: Intervention Case Study](../02-intervention-lending/)
- [View Deliverables](./deliverables/)
