# 05_Adaptability_Guidelines.md

## 1. Adaptability Overview

The Adaptability Guidelines enable teams to deploy the AI Fairness Playbook across any domain, problem type, or organizational context without rebuilding governance infrastructure from scratch. This modular framework translates core fairness requirements into domain-specific implementations while preserving organizational velocity. Technical leads use this to select appropriate metrics and interventions, product managers apply it to scope fairness work realistically, and compliance officers map regulatory requirements to playbook controls. The guidelines provide pre-validated templates for six regulated domains, five ML problem types, and three organizational scales, reducing adaptation costs by 40-60% through reusable governance modules and standardized customization protocols. All adaptations maintain integration with the base playbook's validation framework, ensuring that domain-specific modifications undergo rigorous fairness testing before deployment. The scope covers pre-deployment customization, post-deployment monitoring adjustments, and governance scaling, with built-in triggers for emergency regulatory updates.

---

## 2. Domain Adaptation Matrix

| Domain | Key Fairness Concerns | Protected Groups | Metrics Priority | Regulatory Focus |
|--------|----------------------|------------------|------------------|------------------|
| **Healthcare** | Diagnostic equity, treatment access, clinical validation | Race, age, disability, gender, socioeconomic status | Equal Opportunity (≥0.85), Calibration Error (≤0.05) | HIPAA, FDA AI/ML Guidance, State Medical Boards |
| **Finance** | Credit access, pricing fairness, proxy discrimination | Race, gender, age, income proxy, geography | Demographic Parity (≥0.80), Adverse Impact Ratio | ECOA, Fair Lending, Basel III, SR 11-7 |
| **Recruitment** | Hiring equity, promotion fairness, algorithmic screening | Race, gender, age, disability, veteran status | Demographic Parity (≥0.85), Precision Parity (±3%) | EEOC, EU AI Act, Title VII |
| **Education** | Admissions equity, grading fairness, resource allocation | Race, socioeconomic status, disability, language minority | Equal Opportunity (≥0.90), Exposure Parity | FERPA, Civil Rights Act, Title IX |
| **Criminal Justice** | Risk assessment accuracy, sentencing parity, bail decisions | Race, socioeconomic status, gender | Calibration (±2% across groups), False Positive Parity | COMPAS case law, Equal Protection Clause |
| **Insurance** | Underwriting fairness, claims processing, pricing equity | Race, gender, health status, genetic information | Demographic Parity (≥0.80), Conditional Parity | State Insurance Regulations, ACA Section 1557 |

**Implementation Note:** Healthcare and finance adaptations require additional 1.2-1.8x resource multiplier per S5 calculations. Each domain triggers specific gateway requirements: Healthcare mandates IRB protocols and clinical validation cohorts; Finance requires SR 11-7 model risk documentation and proxy detection analysis.

---

## 3. Problem Type Customization

### 3.1 Binary Classification (Default - EquiHire Pattern)

**Fairness Metrics:** Demographic Parity (DP ≥ 0.80), Equal Opportunity (EO ≥ 0.85), Calibration Error (≤ 0.05). Use disparate impact ratio as primary gate.

**Intervention Techniques:** Threshold optimization using Pareto fairness frontier, adversarial debiasing with demographic parity constraints, preprocessing with Disparate Impact Remover. Deploy post-processing calibration adjustment for high-stakes domains.

**Validation Protocol:** Permutation tests with 10,000 iterations, bootstrap confidence intervals at 95% level, adversarial testing with 3+ red team scenarios per PBI. Monitor fairness drift daily using S4 dashboard.

**Team Composition:** Base playbook team + 0.5 FTE Domain Compliance Officer. FCP multiplier: 1.2x.

### 3.2 Multi-class Classification

**Fairness Metrics:** Per-class Demographic Parity (≥0.80 for each class), macro-averaged Equal Opportunity (≥0.85), worst-case class parity (minimum ratio across classes).

**Intervention Techniques:** Class-weighted training with fairness-aware loss functions, per-class threshold optimization using ROC convex hull, label distribution smoothing for minority classes.

**Validation Protocol:** Stratified testing across all class-protected group intersections, chi-square tests for label distribution parity, confusion matrix analysis per subgroup. Require 100+ samples per intersection for statistical validity.

**Team Composition:** Base playbook team + 1 FTE Data Scientist specializing in imbalanced data. FCP multiplier: 1.4x.

### 3.3 Regression Problems

**Fairness Metrics:** Conditional mean parity (|E[Y|A=a] - E[Y|A=b]| ≤ 0.1σ), residual parity (mean residual difference ≤ 0.05σ), quantile parity at 10th/50th/90th percentiles.

**Intervention Techniques:** Fair regression using LinProg constraints, residual debiasing with post-hoc calibration, adversarial regression networks to remove protected attribute signal.

**Validation Protocol:** Mean difference tests with Bonferroni correction, variance analysis across protected groups, quantile regression disparity checks. Bootstrap resampling mandatory for small datasets (n < 5000).

**Team Composition:** Base playbook team + 1 FTE Econometrician (finance) or Biostatistician (healthcare). FCP multiplier: 1.3x.

### 3.4 Ranking/Recommendation Systems

**Fairness Metrics:** Exposure parity (|Exp(A) - Exp(B)| ≤ 0.10), attention fairness (click-through rate parity ≥ 0.85), position-weighted demographic parity using NDCG@K.

**Intervention Techniques:** Re-ranking with fairness constraints using Linear Programming, exposure-based regularization during training, diversity-aware beam search for generation.

**Validation Protocol:** Position-weighted metrics across top-10, top-100 positions, A/B testing with 2-week minimum runtime, user engagement disparity analysis. Monitor filter bubble effects via serendipity metrics.

**Team Composition:** Base playbook team + 1 FTE UX Researcher + 0.5 FTE Behavioral Economist. FCP multiplier: 1.4x.

### 3.5 LLM/Generative AI

**Fairness Metrics:** Sentiment parity (±5% across groups), stereotype detection rate (≤ 2% toxic generation), demographic representation in generated content (±10% vs. population).

**Intervention Techniques:** Prompt engineering with fairness constraints, fine-tuning on debiased datasets using RLHF with fairness rewards, content filtering with bias detection classifiers.

**Validation Protocol:** A/B testing with 10K+ user sample, human evaluation by 3+ annotators from diverse backgrounds, automated stereotype benchmarking using BOLD and BBQ datasets. Monitor PII leakage and demographic hallucination rates.

**Team Composition:** Base playbook team + 1 FTE NLP Bias Specialist + 1 FTE Prompt Governance Lead. FCP multiplier: 1.8x.

---

## 4. Organizational Scaling Guidelines

### 4.1 Startup (1-50 employees)

| Component | Adaptation |
|-----------|------------|
| **Fair AI Scrum** | Informal SAFE ceremonies, combined AI Ethics Lead + Product Manager role, weekly fairness standup (15 min) |
| **Governance** | Founder-led ethics decisions, lightweight fairness pre-mortem per PBI, community advisory panel (3-5 external members) |
| **Architecture** | Single technique focus (e.g., threshold optimization only), cloud-based fairness dashboard (pre-built templates) |
| **Compliance** | Basic documentation: model cards + fairness report per release, adverse action logging, annual review |
| **Resource Model** | 3-5 FTE total: 1 ML Engineer (fairness owner), 1 Data Scientist, 1 Product Manager, 0.5 Legal, 0.5 Compliance |
| **Budget** | $150K-$250K annually, open-source tools preferred, vendor audit every 6 months |

### 4.2 Mid-size (50-500 employees)

| Component | Adaptation |
|-----------|------------|
| **Fair AI Scrum** | Formal SAFE ceremonies, dedicated Fairness Champion (1 FTE), bi-weekly fairness review with stakeholders |
| **Governance** | Ethics Committee formation (5-7 members: CDO, Legal, HR, Tech, 2 external), quarterly fairness audits, RACI matrix mapped |
| **Architecture** | Multi-technique pipeline (preprocessing + in-processing + postprocessing), fairness API service for reusability |
| **Compliance** | Full documentation per S2 templates, internal audit quarterly, regulatory pre-submission consultations |
| **Resource Model** | 8-12 FTE: 2 ML Engineers, 2 Data Scientists, 1 Fairness Specialist, 1 Privacy Officer, 2 Compliance Analysts, 1 PM, 1 Legal, 1 Ethics Researcher |
| **Budget** | $500K-$800K annually, enterprise fairness toolkit (Fairlearn, What-If Tool), external audit annually |

### 4.3 Enterprise (500+ employees)

| Component | Adaptation |
|-----------|------------|
| **Fair AI Scrum** | Multi-team coordination via SAFe, dedicated Fairness Team (5-7 FTE), fairness guild for knowledge sharing |
| **Governance** | Board-level AI Committee, domain-specific sub-committees (Healthcare, Finance), veto power for high-impact models |
| **Architecture** | Centralized fairness platform with model registry, automated fairness gates in CI/CD, enterprise dashboard |
| **Compliance** | External audit annually, regulatory engagement (FDA, OCC), public fairness reports, whistleblower protections |
| **Resource Model** | 15-25 FTE: 4 ML Engineers, 3 Data Scientists, 2 Fairness Researchers, 2 Privacy Officers, 4 Compliance Analysts, 2 PMs, 2 Legal, 2 Ethics Leads, 1 MLOps Engineer |
| **Budget** | $1.2M-$2M annually, custom fairness infrastructure, dedicated compute for validation, legal retainer for regulatory affairs |

---

## 5. Customization Checklist

#### Step 1: Context Assessment
- [ ] Identify domain from matrix (Section 2) and confirm regulatory focus with Legal
- [ ] Determine problem type (Section 3) and calculate FCP multiplier for sprint planning
- [ ] Assess organizational scale (Section 4) and map current FTEs to required roles
- [ ] Complete S1 Workflow Domain Identification Form and submit to AI Governance Lead within 48 hours
- [ ] Review S4 Risk Tier classification; if Tier 1, schedule Board briefing within 1 week

#### Step 2: Metric Selection
- [ ] Select primary fairness metric based on domain matrix (e.g., Healthcare = Equal Opportunity)
- [ ] Define acceptable threshold using domain baseline: Healthcare (≥0.85), Finance (≥0.80), Education (≥0.90)
- [ ] Identify secondary metrics for monitoring: calibration, stereotype rate, exposure variance
- [ ] Set early warning triggers: metric drops 5% below threshold = automatic escalation to Ethics Committee
- [ ] Document metric rationale in model card per S2 Section 3.1 template

#### Step 3: Technique Selection
- [ ] Match techniques to problem type from Section 3; avoid technique forcing
- [ ] Consider computational constraints: LLM fine-tuning requires 4x GPU hours vs. threshold optimization
- [ ] Plan validation approach: bootstrap CI for small data, A/B test for production systems
- [ ] Evaluate vendor fairness capabilities; if insufficient, budget for custom implementation
- [ ] Complete FAIR checklist from P1-3-1 for each PBI before sprint planning

#### Step 4: Governance Adaptation
- [ ] Map RACI matrix to organizational structure using S2 Appendix C template
- [ ] Establish escalation paths: fairness drift → Fairness Champion → Ethics Committee → Board (Tier 1 only)
- [ ] Define review cadence: daily fairness standup, sprint-end metrics review, quarterly audit
- [ ] Integrate domain-specific stakeholders: Patient Advocate (Healthcare), Fair Lending Officer (Finance)
- [ ] Set up fairness incident response plan with 24-hour response SLA for high-impact issues

#### Step 5: Compliance Mapping
- [ ] Identify applicable regulations from Section 2 matrix; create compliance checklist per S2 Appendix F/H
- [ ] Map documentation requirements: model risk docs (Finance), IRB protocols (Healthcare), EEOC logs (Recruitment)
- [ ] Plan audit preparation: pre-audit self-assessment using S4 test suite, external auditor selection (if required)
- [ ] Schedule legal review gates: pre-deployment (2 weeks before), post-deployment (30 days after)
- [ ] Establish regulatory change monitoring: subscribe to FDA, CFPB, EEOC updates; review S5 quarterly

---

## 6. Cross-Domain Examples

### Example 1: Healthcare Diagnostic AI (Chest X-Ray Triage)

**Context:** 300-bed hospital deploying binary classification model to prioritize urgent chest X-rays across ER departments serving diverse urban population.

**Adaptation Path:**
- **Domain:** Healthcare (HIPAA, FDA Class II SaMD)
- **Problem Type:** Binary Classification (FCP = 1.2x)
- **Scale:** Mid-size (250 employees, dedicated Fairness Champion)

**Fairness Configuration:**
- **Primary Metric:** Equal Opportunity (sensitivity parity) with threshold ≥0.90 across race, age, disability
- **Technique:** Calibration adjustment by demographic using Platt scaling per protected group; adversarial debiasing during training
- **Validation:** Bootstrap CI with 10,000 samples, permutation tests for subgroup sensitivity, clinical validation on 500-patient cohort (IRB-approved)

**Governance Integration:**
- **Ethics Committee:** Chief Medical Officer chairs, includes 2 patient advocates, 1 radiologist, 1 AI Ethics Lead
- **Approval Authority:** Medical Executive Committee sign-off required; patient representative veto power for deployment
- **Monitoring:** Daily fairness dashboard tracked by Fairness Champion; adverse event reporting to FDA within 15 days if disparity detected

**Compliance Execution:**
- **Documentation:** Clinical validation study (per FDA AI/ML guidance), model card with race-stratified performance, PHI protection audit log
- **Timeline:** Base 6 weeks + 3 weeks IRB review + 2 weeks clinical pilot = 11 weeks total
- **Budget:** $180K base × 1.4 healthcare multiplier = $252K (includes 0.5 FTE Clinical Informaticist)

**Outcome:** Achieved 0.92 sensitivity parity across racial groups, identified need for additional training data for Hispanic/Latino subgroup, leading to targeted data collection in Sprint 4.

---

### Example 2: Financial Credit Scoring (Personal Loan Approval)

**Context:** Regional bank updating legacy credit scoring model to machine learning system for $50M loan portfolio, serving historically redlined communities.

**Adaptation Path:**
- **Domain:** Finance (ECOA, Fair Lending, SR 11-7)
- **Problem Type:** Binary Classification (FCP = 1.2x) with regression components for pricing
- **Scale:** Enterprise (1,200 employees, Board AI Committee)

**Fairness Configuration:**
- **Primary Metric:** Demographic Parity (approval rate ratio ≥0.80) for protected classes; secondary metric: calibration within ±3% for credit risk
- **Technique:** Disparate Impact Remover preprocessing + threshold optimization on Pareto frontier + post-hoc adverse action reason codes
- **Validation:** Fair lending regression testing across 40+ proxy variables (ZIP code, education, email domain), 3-year historical portfolio analysis, OCC pre-submission review

**Governance Integration:**
- **Ethics Committee:** Chief Risk Officer chairs Fair Lending Subcommittee with Compliance, Legal, Community Development Officer
- **Approval Authority:** Board AI Committee approval required; quarterly reporting to OCC on disparate impact metrics
- **Monitoring:** Real-time dashboard tracking approval rates by race/gender; automated alert if ratio drops below 0.85 for 3 consecutive days

**Compliance Execution:**
- **Documentation:** Full SR 11-7 model risk documentation, proxy variable analysis report, adverse action notice templates with explainability
- **Timeline:** Base 8 weeks + 4 weeks fair lending testing + 1 week legal review = 13 weeks total
- **Budget:** $220K base × 1.6 finance multiplier = $352K (includes 1 FTE Quantitative Risk Analyst, 0.5 FTE Fair Lending Specialist)

**Outcome:** Reduced disparate impact ratio from 0.72 to 0.84 while maintaining profitability; identified proxy discrimination via email domain variable, removed from model after legal review.

---

### Example 3: Educational Recommendations (MOOC Course Suggestions)

**Context:** EdTech platform with 2M users launching personalized course recommendation system to increase completion rates among underserved learners.

**Adaptation Path:**
- **Domain:** Education (FERPA, Civil Rights Act)
- **Problem Type:** Ranking/Recommendation System (FCP = 1.4x)
- **Scale:** Mid-size (150 employees, Ethics Committee forming)

**Fairness Configuration:**
- **Primary Metric:** Exposure Parity (probability of seeing advanced courses within 5% across socioeconomic status)
- **Technique:** Re-ranking with fairness constraints using Linear Programming; diversity-aware beam search to prevent filter bubbles
- **Validation:** A/B testing over 4-week period with 100K user cohort, NDCG@10 parity analysis, serendipity metrics to measure content diversity

**Governance Integration:**
- **Ethics Committee:** VP of Product chairs with Academic Advisor, Accessibility Officer, 1 student representative
- **Approval Authority:** Ethics Committee majority vote; escalation to CEO if completion rate gap increases >5%
- **Monitoring:** Weekly exposure parity reports by SES quartile; quarterly accessibility audit for disability accommodations

**Compliance Execution:**
- **Documentation:** FERPA-compliant data handling procedures, accessibility conformance report (WCAG 2.1), civil rights impact assessment
- **Timeline:** Base 5 weeks + 2 weeks accessibility testing = 7 weeks total
- **Budget:** $95K base × 1.3 education multiplier = $124K (includes 0.5 FTE UX Researcher, 0.5 FTE Accessibility Specialist)

**Outcome:** Achieved exposure parity of 0.96 for low-SES users; discovered recommendation engine was under-suggesting STEM courses to female learners, corrected via re-ranking weights in Sprint 2.

---

## 7. Common Adaptation Pitfalls

| Pitfall | Description | Mitigation |
|---------|-------------|------------|
| **Metric Mismatch** | Using Demographic Parity when Equal Opportunity is clinically required (e.g., healthcare diagnosis) | Consult domain matrix (Section 2) during Step 1; require Clinical Informaticist or Fair Lending Specialist sign-off on metric selection |
| **Over-Engineering** | Applying enterprise-grade governance (Board oversight, external audit) to startup with single low-risk model | Use organizational scale table (Section 4) strictly; startups should default to informal SAFE ceremonies and founder-led decisions |
| **Regulation Gap** | Deploying in new domain without mapping domain-specific requirements (e.g., missing FDA pre-submission for SaMD) | Complete Step 5 compliance mapping before sprint planning; maintain M3 Regulatory Intelligence Feed subscription for real-time updates |
| **Technique Forcing** | Using binary classification threshold optimization for ranking problem, resulting in irrelevant fairness guarantees | Match problem type per Section 3; ranking requires re-ranking constraints, not threshold tuning; enforce FCP-based technique review |
| **Cultural Resistance** | Imposing Fair AI Scrum ceremonies in organization with rigid waterfall culture, causing adoption failure | Integrate SAFE additions incrementally: start with +15 min fairness risk mapping in existing planning meetings; use P1-3-1 FAIR checklist as lightweight entry point |
| **Resource Underestimation** | Budgeting base playbook cost without domain multiplier, leading to 60-80% cost overruns | Apply S5 resource multiplier (1.2-1.8x) during Step 1; include FTE additions from Section 4 in headcount planning; secure CFO approval for adapted budget |
| **Proxy Blindness** | Failing to detect proxy variables (ZIP codes, email domains) that recreate protected attribute discrimination | Mandate proxy detection analysis for finance, insurance, recruitment domains; use S4 adversarial testing suite with 40+ proxy variables; require MRM team sign-off |

---

## 8. Adaptation Validation

Before deploying adapted playbook, the AI Governance Lead must certify completion of all validation gates:

### Pre-Deployment Validation
- [ ] **Domain-specific metrics defined** with thresholds documented in model card and approved by domain specialist (Clinical Informaticist, Fair Lending Officer, etc.)
- [ ] **Techniques validated** on representative holdout data showing fairness metric improvement ≥10% over baseline; adversarial testing completed with zero critical findings
- [ ] **Governance structure mapped** to organization chart with RACI matrix signed by all stakeholders; escalation paths tested via tabletop exercise
- [ ] **Compliance requirements documented** in S2 Implementation Guide project plan with legal review completed and regulatory pre-submission scheduled (if applicable)
- [ ] **Stakeholder sign-off obtained** from Ethics Committee (or Board for Tier 1), domain specialist, and Fairness Champion; documented in S4 Governance Validation Dashboard

### Post-Deployment Validation
- [ ] **Fairness monitoring** activated with daily automated checks and weekly manual review by Fairness Champion for first 30 days
- [ ] **Incident response plan** tested within 14 days of deployment; response team can be convened within 4 hours of alert
- [ ] **User feedback loop** established: adverse outcome appeals process documented, community advisory panel meeting scheduled quarterly
- [ ] **Regulatory reporting** calendar created with automated reminders: Finance (OCC quarterly), Healthcare (FDA adverse events within 15 days)

### Continuous Validation
- [ ] **Quarterly S5 review** scheduled to assess adaptation effectiveness using KPIs: adaptation velocity (<14 days target), compliance coverage (100% target), resource efficiency (<10% variance)
- [ ] **Emergency trigger** activated: regulatory changes (e.g., new CFPB guidance) prompt S5 review within 30 days; critical updates deployed within 90 days

**Certification Statement:** "I confirm this adaptation meets domain-specific regulatory requirements, maintains fairness metric thresholds, and integrates appropriately scaled governance. Signed: _________________ (AI Governance Lead), Date: __________"

---

**Document Owner:** Chief AI Ethics Officer  
**Approval Authority:** CEO (cross-domain) / Domain C-Suite (single-domain)  
**Version:** 3.0 (Module 3)  
**Last Updated:** 2026-01-29  
**Next Review:** Upon 10% shift in regulatory landscape or 6 months (whichever first)  
**Integration:** Receives from S1 (Component Integration), P1-3-1 (Scrum Toolkit); Produces for S2 (Implementation Guide), S4 (Validation Framework)