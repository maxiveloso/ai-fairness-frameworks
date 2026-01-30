# Validation Framework - Module 3 Implementation Playbook

**File:** `04_Validation_Framework.md`  
**Version:** 1.0  
**Last Updated:** 2026-01-29  
**Audience:** Technical Leads, Quality Assurance, Compliance Officers

---

## 1. Validation Framework Overview

The Validation Framework provides a systematic methodology for verifying that fairness implementation across all four playbook components achieves measurable, sustained improvements in AI system equity. This framework ensures that process changes translate into genuine bias reduction, not merely ceremonial compliance. It establishes rigorous checkpoints, quantitative metrics, and evidence requirements that technical teams must satisfy before advancing implementation phases.

The framework operates across two validation planes: **process fidelity** (are we doing the work correctly?) and **outcome efficacy** (is the work producing fairer AI systems?). Process validation confirms that Fair AI Scrum ceremonies, organizational RACI matrices, architecture techniques, and compliance documentation are executed as designed. Outcome validation measures tangible improvements in demographic parity, equal opportunity, detection latency, and regulatory readiness. This dual approach prevents the common failure mode where teams perform fairness activities without achieving fairness results.

Scope encompasses all four implementation sprints: Fair AI Scrum (Sprint 1), Organizational Integration (Sprint 2), Architecture Cookbook (Sprint 3), and Regulatory Compliance (Sprint 4). Validation occurs continuously within sprints, at formal phase-gate checkpoints, and through quarterly maturity assessments. The framework integrates with existing quality assurance systems, treating fairness as a non-negotiable quality attribute alongside performance and security.

---

## 2. Maturity Assessment Model

| Level | Description | Criteria | Typical Timeline |
|-------|-------------|----------|------------------|
| **1 - Initial** | Ad-hoc fairness activities with no formal process. Teams may discuss bias informally but lack standardized ceremonies, metrics, or accountability structures. | • No documented fairness Definition of Done<br>• Zero fairness user stories in backlog<br>• No assigned fairness roles<br>• Bias detection occurs post-deployment only | Starting point (Week 0) |
| **2 - Defined** | Documented fairness processes established and communicated. Ceremonies exist on paper, metrics are defined, and initial training completed. | • Fairness ceremonies scheduled in >75% of sprints<br>• RACI matrix drafted and socialized<br>• Baseline fairness metrics measured for pilot models<br>• Ethics Committee charter signed | Months 1-3 (Weeks 1-12) |
| **3 - Managed** | Consistent execution with measured outcomes. Processes are followed rigorously, governance actively reviews metrics, and early bias reduction demonstrated. | • Fairness gates block 100% of non-compliant deployments<br>• Escalation paths utilized with <5 day resolution time<br>• DP/EO metrics improved >15% from baseline<br>• 100% of high-risk models have documented trade-off analyses | Months 4-6 (Weeks 13-24) |
| **4 - Optimized** | Continuous improvement with predictive monitoring. Automated gates, proactive bias detection, and industry-leading fairness practices embedded in culture. | • Automated fairness monitoring with <1hr alert latency<br>• Predictive bias detection prevents 90% of issues pre-training<br>• Team adoption rate >90% across all AI teams<br>• External audit scores exceed regulatory minimums by 25% | Months 7+ (Weeks 25+) |

---

## 3. Component-Specific Validation Checklists

### 3.1 Fair AI Scrum (Sprint 1) Validation

**Process Validation:**
- [ ] Fairness ceremonies occur in every sprint (Planning, Standup, Review, Retrospective)
- [ ] Definition of Done includes explicit fairness criteria: disparate impact ≥0.80, equal opportunity ≥0.85, calibration error ≤0.05
- [ ] Minimum one fairness user story per sprint per team, tagged with `FAIRNESS` label and Impact Assessment score
- [ ] Sprint retrospectives dedicate ≥15 minutes to fairness incident review and process improvement
- [ ] FAIR checklist (Fairness dimensions, Auditable trail, Impact assessment, Red team scenarios) completed before PBIs reach "Ready" state
- [ ] Fairness Complexity Points (FCP) assigned to all stories using 1-5 scale, with FCP=5 stories requiring Ethics Lead pre-approval

**Outcome Validation:**
- [ ] Bias issues detected earlier: Average detection sprint ≤2 (measured from first code commit to fairness bug creation in tracking system)
- [ ] Team fairness awareness improved: Survey score ≥4.0/5.0 on questions regarding confidence identifying bias and understanding escalation paths
- [ ] Fairness tasks completed on time: Sprint velocity includes fairness work with <10% carryover to subsequent sprints
- [ ] Stakeholder participation: Minimum one community representative or domain expert reviews fairness acceptance criteria per sprint

**Evidence Required:**
- **Sprint ceremony records:** Jira/Azure DevOps logs showing fairness agenda items, meeting minutes with attendance, time-stamped video recordings for distributed teams
- **User story examples:** 3-5 fully documented stories including fairness AC, FCP scores, Impact Assessment tags, and stakeholder equity weights (e.g., "Historically underserved communities weight: 1.5x")
- **Retrospective notes:** Specific entries capturing fairness learnings, such as "Learned that demographic parity alone insufficient; added equal opportunity metric after Sprint 3 retrospective"
- **SAFE/FAIR execution logs:** Completion timestamps for each checklist item, with blockers flagged and resolution documented

---

### 3.2 Organizational Integration (Sprint 2) Validation

**Process Validation:**
- [ ] RACI matrix documented in Confluence/SharePoint with version control, covering all AI fairness decisions from data collection to model retirement
- [ ] Escalation paths defined with explicit SLAs: Level 1 (Team Lead) <24hrs, Level 2 (Fairness Champion) <72hrs, Level 3 (Ethics Committee) <7 days
- [ ] Cross-team coordination active: Bi-weekly Fairness Sync meeting with representatives from Data Science, Legal, Product, and affected community groups
- [ ] Ethics Committee meets monthly with quorum (≥75% members), documented minutes, and decision log tracking all fairness exceptions and overrides
- [ ] Accountability clear: Fairness objectives included in performance reviews for 100% of technical staff and 50% of product managers

**Outcome Validation:**
- [ ] Decision latency reduced: Median time from fairness issue escalation to resolution ≤5 business days (tracked in dedicated governance log)
- [ ] Escalations handled appropriately: 100% of escalations follow documented path; audit shows zero bypassed escalations for high-risk issues
- [ ] Accountability compliance: Quarterly review shows 90%+ adherence to RACI assignments; misaligned decisions flagged and remediated within 1 sprint

**Evidence Required:**
- **RACI matrix documentation:** Signed PDF with executive sponsor approval, linked to org chart and role descriptions; includes "Accountable" party for each fairness gate
- **Escalation log:** Timestamped entries showing issue ID, severity (Critical/High/Medium), escalation level, time-to-resolution, and final decision; sample entry: "Issue-204: DP ratio 0.72 → Escalated Day 1 → Ethics Committee review Day 3 → Model rollback approved Day 4"
- **Ethics Committee meeting minutes:** Standardized template capturing attendance, agenda items, decisions made, dissenting opinions, and action items with owners
- **Cross-team coordination calendar:** Recurring meeting invites, shared agendas, and documented outcomes (e.g., "Data Science and Legal aligned on proxy variable handling protocol")

---

### 3.3 Architecture Cookbook (Sprint 3) Validation

**Process Validation:**
- [ ] Appropriate techniques selected for model type: Pre-processing for tabular data (e.g., reweighing, disparate impact remover), in-processing for deep learning (e.g., adversarial debiasing), post-processing for legacy models (e.g., equalized odds)
- [ ] Fairness-utility trade-offs documented: Explicit record of accuracy loss vs. fairness gain, with business stakeholder sign-off on acceptable thresholds (e.g., "≤3% accuracy reduction for DP improvement ≥0.15")
- [ ] Technique execution follows prescribed methods: Code reviews confirm implementation matches architecture guide; hyperparameters logged and justified
- [ ] Results statistically validated: Permutation tests (10,000 permutations) and bootstrap confidence intervals (10,000 samples) completed for all fairness metric claims

**Outcome Validation:**
- [ ] Fairness metrics improved: Demographic Parity ratio increased ≥0.15 from baseline; Equal Opportunity difference reduced to ≤0.10
- [ ] Accuracy within acceptable bounds: Validation accuracy loss ≤5% compared to baseline model; business KPIs (e.g., loan approval rate, customer satisfaction) remain within ±2% of targets
- [ ] Statistical significance achieved: Permutation test p-value <0.05; 95% bootstrap CI lower bound exceeds fairness threshold (e.g., DP 95% CI [0.81, 0.89] passes)

**Evidence Required:**
- **Technique selection rationale:** Decision document linking model type to technique choice, including constraints (latency, data availability) and alternatives considered; example: "Selected reweighing over adversarial debiasing due to <50ms inference requirement"
- **Execution logs with parameters:** Weights & Biases or MLflow runs showing fairness technique hyperparameters, training curves for both accuracy and fairness metrics, and environment snapshots for reproducibility
- **Statistical validation results:** Permutation test histograms, p-values, and bootstrap CI calculations; example: "EquiHire case: p=0.0002, DP CI [0.334, 0.409] pre-intervention → p<0.001, DP CI [0.781, 0.823] post-intervention"

---

### 3.4 Regulatory Compliance (Sprint 4) Validation

**Process Validation:**
- [ ] Risk classification completed: All AI systems categorized using 3x3 matrix (Impact: Low/Medium/High × Likelihood: Low/Medium/High); high-risk systems flagged for enhanced oversight
- [ ] Documentation meets regulatory requirements: Model Cards complete per Model Card Toolkit v1.0; Data Sheets include provenance, consent, and bias audit; DPIA conducted for any system processing special category data
- [ ] Audit trail maintained: Immutable logs of all model versions, training data snapshots, fairness evaluations, and deployment decisions stored in compliance-grade repository (e.g., AWS S3 with object lock)
- [ ] Impact assessments conducted: Algorithmic Impact Assessment (AIA) for public sector or DPIA for GDPR-covered systems; includes stakeholder consultation minutes and mitigation plan

**Outcome Validation:**
- [ ] Compliance gaps closed: 100% of high-risk systems pass internal mock audit; gap closure rate tracked monthly with zero critical findings outstanding >30 days
- [ ] Audit readiness: Mock audit score ≥90% using regulator's assessment rubric; all documentation exportable within 4 hours of request
- [ ] Documentation completeness: 100% of required fields filled in Model Cards/Data Sheets; no "N/A" entries without justification approved by Compliance Officer

**Evidence Required:**
- **Risk classification document:** Spreadsheet with system ID, risk rating, justification, and approval signature from Legal; linked to governance tier assignment
- **Model Cards / Data Sheets:** Published to internal model registry with version history; example fields: "Intended Use," "Training Data Demographics," "Fairness Metrics (DP, EO, Calibration)," "Limitations"
- **Audit trail exports:** Demonstrated extract showing model version v2.3 → training data hash abc123 → fairness evaluation ID 456 → deployed by user@company.com at timestamp
- **DPIA (if applicable):** 20+ page assessment covering necessity, proportionality, risk assessment, mitigation measures, and stakeholder consultation; signed by Data Protection Officer

---

## 4. Quantitative Success Metrics

| Metric | Definition | Target | Measurement Method |
|--------|------------|--------|-------------------|
| **Demographic Parity** | min(P(Y=1|A=a)) / max(P(Y=1|A=a)) across all protected groups | ≥ 0.80 | Automated fairness dashboard pulling predictions from production model endpoint; calculated daily on rolling 30-day window |
| **Equal Opportunity** | True Positive Rate parity: min(TPR_a) / max(TPR_a) | ≥ 0.80 | Model evaluation pipeline triggered post-training; uses held-out test set stratified by protected attributes |
| **Detection Time** | Number of sprints from feature development start to fairness bug detection in Jira | ≤ 2 sprints | Issue tracking system query: `created_date - commit_date` filtered by `labels=fairness-bug`; aggregated per sprint |
| **Decision Latency** | Business days from fairness escalation email to resolution decision logged | ≤ 5 days | Governance log timestamp analysis; excludes weekends and holidays; tracked in dedicated Airtable/Asana board |
| **Compliance Score** | Percentage of regulatory requirements met from internal checklist (50 items) | 100% | Compliance checklist in Google Sheets; monthly self-assessment by product owner, quarterly audit by Compliance Officer |
| **Team Adoption** | Percentage of AI teams conducting fairness ceremonies in ≥80% of sprints | ≥ 90% | Sprint audit: Query Confluence/Jira for `ceremony_type=fairness-review` across all teams; denominator = total active AI teams |
| **Accuracy Retention** | (Fair model accuracy) / (Baseline model accuracy) | ≥ 0.95 | MLflow model registry comparison; accuracy measured on same held-out test set; fairness model must pass target DP/EO thresholds |
| **Stakeholder Trust** | Survey score (1-5) from affected community representatives on fairness process transparency | ≥ 4.0 | Quarterly SurveyMonkey to community advisory board; questions: "Do you understand how we test for bias?" "Can you influence model decisions?" |

---

## 5. Validation Checkpoints

### Checkpoint 1: End of Foundation Phase (Week 4)
**Gates:**
- [ ] All pilot teams (minimum 3 teams) have fairness ceremonies scheduled in calendar with recurring invites and named facilitators
- [ ] At least 1 fairness user story per team written, refined, and approved by Product Owner with Fairness AC, FCP score, and Impact Assessment tag
- [ ] Baseline fairness metrics measured for at least 2 production models: DP, EO, calibration error documented in fairness dashboard with 30 days historical data
- [ ] SAFE/FAIR checklist integrated into Definition of Ready; 100% of new PBIs show FAIR completion in Jira custom fields
- [ ] Fairness Champion role assigned and trained; attendance certificate from Fair AI Scrum workshop on file

**Deliverables:** Baseline metrics report, ceremony schedule PDF, FAIR checklist integration proof, Champion training record

---

### Checkpoint 2: End of Governance Phase (Week 8)
**Gates:**
- [ ] RACI matrix approved by leadership (VP Engineering, Chief Legal Officer, Head of Product) with digital signatures
- [ ] First Ethics Committee meeting held with recorded minutes, attendance sheet, and decision log initialized
- [ ] Escalation process tested with mock scenario: Simulated fairness issue injected; tracked end-to-end with timestamps confirming SLA compliance (L1 <24hr, L2 <72hr, L3 <7 days)
- [ ] Cross-team coordination meeting cadence established: Bi-weekly Fairness Sync on calendar with rotating agenda ownership
- [ ] Fairness objectives embedded in performance review templates for all technical staff; HR system screenshots showing field addition

**Deliverables:** Signed RACI matrix, Ethics Committee charter and minutes, escalation test report, performance review template

---

### Checkpoint 3: End of Technical Phase (Week 12)
**Gates:**
- [ ] Fairness technique applied to at least 1 high-impact model: Code merged to main branch, training completed, evaluation metrics logged
- [ ] Fairness metrics improved from baseline: DP ratio increased ≥0.10 or EO difference decreased ≥0.10; dashboard shows clear before/after comparison
- [ ] Statistical validation completed: Permutation test report with p-value <0.05; bootstrap CI report with 95% intervals; both uploaded to model registry
- [ ] Trade-off analysis document signed by business stakeholder accepting any accuracy loss; includes "go/no-go" decision rationale
- [ ] Architecture Cookbook technique selection guide used: Decision tree followed, alternatives documented, final choice justified in 1-page memo

**Deliverables:** Technical validation report, statistical test results, trade-off analysis sign-off, technique selection memo

---

### Checkpoint 4: End of Compliance Phase (Week 16)
**Gates:**
- [ ] Risk classification documented for 100% of AI systems in production: Inventory spreadsheet complete with 3x3 risk ratings, reviewed by Legal
- [ ] All required documentation created: Model Cards for top 5 models; Data Sheets for training datasets; DPIA completed for any special category data processing
- [ ] Mock compliance audit passed: Internal audit team conducts 4-hour review using regulator's rubric; score ≥90% with zero critical findings
- [ ] Audit trail system operational: Demonstrated export capability (4-hour SLA met); logs show immutability and tamper-evident features
- [ ] Regulatory requirement mapping complete: Traceability matrix linking each requirement (e.g., NYC LL 144, EU AI Act) to specific document sections

**Deliverables:** Risk inventory, Model Cards/Data Sheets, mock audit report, audit trail demo video, requirements traceability matrix

---

### Checkpoint 5: Integration Validation (Week 20)
**Gates:**
- [ ] All 4 components integrated: Workflow diagram showing handoffs from Scrum → Governance → Architecture → Compliance with tool integrations (Jira → Confluence → MLflow → GRC platform)
- [ ] Cross-component workflows tested: End-to-end scenario (e.g., bias detected in sprint → escalated → technique applied → documented) executed with all timestamps < defined SLAs
- [ ] End-to-end case study documented: Real or realistic example (e.g., "EquiHire hiring model fairness remediation") showing each component's contribution, metrics at each stage, and lessons learned
- [ ] Maturity assessment conducted: Self-assessment against Level 1-4 criteria; external validation by Fairness Champion; final score documented with improvement plan
- [ ] Governance dashboard live: Executive view showing real-time DP/EO metrics, open escalation count, compliance score, and team adoption rate

**Deliverables:** Integration architecture diagram, workflow test report, case study PDF, maturity assessment scorecard, live dashboard URL

---

## 6. Statistical Validation Methods

### Permutation Test for Fairness Improvement

**Purpose:** Verify that observed fairness improvements result from the applied technique rather than random variation in model training.

**Method:**
1. Compute baseline fairness metric (e.g., Demographic Parity ratio) on original model: DP_base = 0.45
2. Apply fairness technique to produce improved model: DP_treated = 0.82
3. Pool the predictions from both models and randomly reassign group labels 10,000 times
4. For each permutation, calculate DP_ratio to construct null distribution
5. Compute p-value: proportion of permutations where DP_perm ≥ DP_treated

**Pass Criteria:** p < 0.05 (two-tailed)

**Example Execution:**
```python
# Using fairlearn.metrics
from fairlearn.metrics import demographic_parity_ratio
from scipy.stats import permutation_test

def dp_statistic(y_true, y_pred, sensitive_features):
    return demographic_parity_ratio(y_true, y_pred, 
                                   sensitive_features=sensitive_features)

result = permutation_test(
    (y_true, y_pred_base, y_pred_treated, sensitive_features),
    dp_statistic,
    permutation_type='pairing',
    n_resamples=10000
)
print(f"Observed improvement: {dp_treated - dp_base}")
print(f"P-value: {result.pvalue}")  # Target: < 0.05
```

**Example Result:** EquiHire case study showed p = 0.0002, confirming statistically significant improvement from reweighing.

---

### Bootstrap Confidence Intervals for Metric Uncertainty

**Purpose:** Quantify uncertainty in fairness metrics due to finite sample sizes and provide reliable bounds for decision-making.

**Method:**
1. From test set (n=10,000), draw 10,000 bootstrap samples (size n, with replacement)
2. For each bootstrap sample, calculate DP ratio, EO difference, and calibration error
3. Compute 95% percentile confidence intervals from bootstrap distribution

**Pass Criteria:** 95% CI lower bound > threshold (e.g., DP threshold = 0.80)

**Example Execution:**
```python
import numpy as np
from fairlearn.metrics import demographic_parity_ratio

def bootstrap_dp_ci(y_pred, sensitive_features, n_bootstrap=10000, ci=0.95):
    dp_values = []
    n = len(y_pred)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        dp = demographic_parity_ratio(
            y_true[idx], y_pred[idx], 
            sensitive_features=sensitive_features[idx]
        )
        dp_values.append(dp)
    
    lower = np.percentile(dp_values, (1-ci)/2 * 100)
    upper = np.percentile(dp_values, (1+ci)/2 * 100)
    return lower, upper

# EquiHire post-intervention
lower, upper = bootstrap_dp_ci(y_pred_fair, gender_feature)
print(f"DP 95% CI: [{lower:.3f}, {upper:.3f}]")  
# Target: Lower bound > 0.80
# Result: [0.781, 0.823] ✓
```

**Interpretation:** If 95% CI lower bound exceeds threshold, we can be confident the true fairness metric meets requirements despite sampling variability.

---

## 7. Validation Governance

| Role | Validation Responsibility | Frequency | Evidence Reviewed |
|------|---------------------------|-----------|-------------------|
| **Team Lead** | Process validation at sprint level; ensures FAIR checklist completion, ceremony execution, and story quality | Every sprint | Sprint ceremony logs, user story compliance, retrospective action items |
| **Fairness Champion** | Cross-team validation, metric tracking, and escalation facilitation; conducts weekly metric review | Weekly | Fairness dashboard, escalation log, cross-team sync minutes |
| **QA Engineer** | Technical validation of statistical methods; reviews permutation tests, bootstrap CIs, and reproducibility | Per model release | Statistical validation reports, MLflow runs, code review approvals |
| **Compliance Officer** | Regulatory validation; ensures documentation completeness, audit trail integrity, and risk classification accuracy | Monthly | Model Cards, Data Sheets, audit trail exports, DPIAs |
| **Ethics Committee** | Quarterly outcome validation; reviews aggregate metrics, escalation trends, and policy exceptions | Quarterly | Quarterly metrics rollup, escalation case studies, policy amendment proposals |
| **External Auditor** | Annual comprehensive validation; independent assessment of maturity level, compliance posture, and control effectiveness | Annually | All validation artifacts, governance logs, interview transcripts |

**Escalation Path:** Team Lead → Fairness Champion (process issues) → QA Engineer (statistical issues) → Compliance Officer (regulatory issues) → Ethics Committee (strategic issues)

---

## 8. Remediation Procedures

| Failure Type | Trigger | Immediate Action | Root Cause Analysis | Follow-up |
|--------------|---------|------------------|---------------------|-----------|
| **Process Failure** | Checkpoint not met (e.g., <75% ceremony coverage) | Halt new fairness story creation; conduct team retro within 48hrs | 5 Whys analysis: Why did ceremonies drop? (e.g., "no facilitator" → "no backup trained") | Process adjustment: Assign rotating facilitator, add to onboarding checklist; re-audit in 2 sprints |
| **Outcome Failure** | Fairness metric below threshold for 2 consecutive days | Rollback model to last compliant version; trigger incident response | Statistical analysis: Is failure due to data drift, concept drift, or implementation error? | Technique review: If drift, retrain with recent data; if error, code fix and enhanced unit tests; parameter tuning if trade-off misconfigured |
| **Statistical Failure** | Permutation test p > 0.05 or bootstrap CI lower bound < threshold | Block deployment; escalate to QA Engineer and Fairness Champion | Power analysis: Insufficient sample size? Technique inappropriate for data distribution? | Remediation: Collect more data (minimum n=5000 per group), select alternative technique (e.g., switch from reweighing to adversarial debiasing), or adjust thresholds with Ethics Committee approval |
| **Compliance Failure** | Documentation gap identified in mock audit (e.g., missing Model Card field) | Immediate documentation sprint (2-day timebox); assign dedicated technical writer | Gap mapping: Which systems affected? Is gap due to tooling, process, or training? | Update templates and checklists; conduct training for all product owners; re-run mock audit within 1 week |
| **Adoption Failure** | Team adoption rate <80% after Week 12 | 1:1 interviews with non-adopting teams; identify blockers (tooling, time, understanding) | Culture analysis: Are incentives misaligned? Is leadership modeling behavior? | Interventions: Executive mandate, allocate 20% team time to fairness, peer mentoring from high-adopting teams, adjust performance incentives |

**Remediation SLA:** All failures must have documented remediation plan within 5 business days of detection; critical failures (compliance, statistical) require plan within 24 hours.

---

## 9. Continuous Improvement

After each validation cycle (sprint, checkpoint, or quarterly review), teams must execute the following improvement protocol:

1. **Document Findings in Validation Report**  
   Use standardized template capturing: validation date, scope, metrics measured, pass/fail status, remediation actions taken. Store in centralized validation repository (e.g., `fairness-validation/` in Confluence) with version history enabled.

2. **Identify Improvement Opportunities**  
   Conduct 30-minute team huddle analyzing: What validation steps felt redundant? Which metrics were ambiguous? Where did we catch issues too late? Categorize opportunities into "process," "tooling," "training," or "metrics."

3. **Update Playbook Processes if Needed**  
   If ≥2 teams report same issue, initiate playbook revision. Version-controlled changes in GitHub with pull request requiring review from Fairness Champion and Compliance Officer. Document rationale in changelog: "v1.2: Reduced bootstrap samples from 10k to 5k after analysis showed stable CIs; saves 3hrs compute per validation."

4. **Share Learnings Across Teams**  
   Monthly "Fairness Forum" where teams present validation wins and failures. Recorded 15-minute lightning talks with Q&A. Key insights distilled into "Validation Patterns" library (e.g., "Pattern: Early proxy detection," "Pattern: Statistical power pitfalls").

5. **Maturity Re-assessment**  
   Every 6 months, repeat maturity self-assessment. Track progression from Level 1→2→3→4. If plateaued at level for >2 quarters, trigger executive review for resource allocation or cultural intervention.

**Feedback Loop:** All improvement actions must have owner and due date; tracked in same governance system as escalations. Completion rate of improvement actions becomes a meta-metric for validation effectiveness.