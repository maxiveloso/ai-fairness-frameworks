# Consolidation: C6 (C6)

**Tier**: Tier 2 Consolidation
**Model**: claude-sonnet-4-5

---

# C6: Adaptability Guidelines - Module 2 Intervention Playbook

## 1. Adaptability Philosophy

**Core Principle: Structured Flexibility**

The Module 2 Intervention Playbook maintains a constant core methodology—systematic detection, diagnosis, and intervention for algorithmic bias—while adapting its application based on domain context, regulatory environment, and problem type. This philosophy recognizes that fairness is not a universal constant but a context-dependent requirement shaped by legal frameworks, stakeholder values, and domain-specific risks.

**What Remains Constant:**
- The three-stage intervention framework (pre-processing, in-processing, post-processing)
- Systematic fairness metric evaluation before and after intervention
- Documentation and validation requirements
- Stakeholder engagement principles
- Trade-off analysis between fairness and performance

**What Adapts:**
- Applicable fairness definitions (e.g., calibration prioritized in healthcare, demographic parity in lending)
- Regulatory compliance requirements (ECOA vs. HIPAA vs. Constitutional constraints)
- Permissible intervention techniques (e.g., explainability requirements in finance, safety constraints in healthcare)
- Risk tolerance thresholds (zero tolerance for patient harm vs. acceptable credit access reduction)
- Validation protocols (fair lending testing vs. clinical validation vs. legal review)

**Adaptation Decision Framework:**

```
1. DOMAIN IDENTIFICATION
   ↓
2. REGULATORY MAPPING (What laws apply?)
   ↓
3. FAIRNESS DEFINITION SELECTION (What does "fair" mean here?)
   ↓
4. CONSTRAINT IDENTIFICATION (What interventions are permissible?)
   ↓
5. RISK ASSESSMENT (What harms could interventions cause?)
   ↓
6. TECHNIQUE SELECTION (Which interventions meet constraints?)
   ↓
7. VALIDATION PROTOCOL (How do we prove it works safely?)
   ↓
8. DEPLOYMENT WITH MONITORING
```

This framework ensures interventions are both technically sound and contextually appropriate, balancing universal fairness principles with domain-specific realities.

---

## 2. Finance Domain (PRIMARY FOCUS)

### Regulatory Environment

**Key Legislation:**
- **Equal Credit Opportunity Act (ECOA)**: Prohibits credit discrimination based on race, color, religion, national origin, sex, marital status, age, or receipt of public assistance
- **Fair Credit Reporting Act (FCRA)**: Requires adverse action notices explaining credit denials, including specific reasons
- **Regulation B**: Implements ECOA, mandates the 80% rule (approval rate for protected group ≥ 80% of control group)
- **Fair Lending Laws**: Prohibit disparate treatment and disparate impact in lending
- **GDPR (EU)**: Right to explanation for automated decisions affecting individuals

**Regulatory Bodies:**
- Consumer Financial Protection Bureau (CFPB)
- Office of the Comptroller of the Currency (OCC)
- Federal Deposit Insurance Corporation (FDIC)
- Federal Reserve Board

### Key Stakeholders

- **Consumers**: Loan applicants, credit card users, insurance purchasers
- **Financial Institutions**: Banks, credit unions, lenders
- **Regulators**: CFPB, OCC, FDIC examiners
- **Compliance Officers**: Internal fairness and legal compliance teams
- **Advocacy Groups**: Fair lending organizations, consumer protection groups

### Common Applications

- Credit approval (mortgages, auto loans, personal loans, credit cards)
- Loan pricing (interest rate determination)
- Credit limit assignment
- Insurance underwriting and pricing
- Fraud detection
- Collections prioritization

### Fairness Definitions

**Primary: Demographic Parity (80% Rule)**
- Approval rate for protected group ≥ 80% of control group approval rate
- Regulatory standard under Regulation B
- Example: If 60% of White applicants approved, ≥48% of Black applicants must be approved

**Secondary: Equal Opportunity**
- True positive rates equal across groups (qualified applicants approved at equal rates)
- Addresses merit-based fairness concerns

**Tertiary: Calibration**
- Predicted probabilities match actual outcomes within groups
- Critical for risk-based pricing

### Intervention Constraints

**1. Explainability Requirement (FCRA Compliance)**
- All credit denials require adverse action notices with specific reasons
- Model must support feature importance extraction
- "Black box" models require post-hoc explanation mechanisms
- Explanation must reference specific applicant characteristics

**2. Credit Access Preservation**
- Interventions cannot reduce overall credit availability below business viability thresholds
- Must balance fairness with financial institution sustainability
- Regulatory expectation: Expand access to underserved groups, not restrict access to majority

**3. Demographic Parity Enforcement**
- 80% rule is regulatory standard, not aspirational goal
- Interventions must demonstrably close disparate impact gaps
- Documentation required for regulatory examinations

**4. Protected Attribute Handling**
- Cannot use protected attributes for adverse decisions (disparate treatment)
- Can use for fairness monitoring and intervention
- Must document rationale for any use of protected attributes

**5. Model Documentation**
- Full audit trail required: data, features, model architecture, validation results
- Intervention rationale must be documented
- Regulatory examiners will review model risk management processes

### Intervention Considerations by Stage

**Pre-Processing:**
- **Permitted**: Reweighting, disparate impact remover, data augmentation
- **Constraint**: Must preserve ability to explain individual decisions
- **Risk**: Over-correction leading to unqualified approvals
- **Validation**: Check that feature distributions remain interpretable

**In-Processing:**
- **Permitted**: Fairness-constrained optimization, adversarial debiasing, regularization
- **Constraint**: Model must remain interpretable or support post-hoc explanation
- **Risk**: Performance degradation affecting financial viability
- **Validation**: Ensure fairness constraints don't create new disparities

**Post-Processing:**
- **Permitted**: Threshold optimization, calibration adjustments, score transformation
- **Constraint**: Threshold changes must be documentable and explainable
- **Risk**: Arbitrary-seeming cutoffs difficult to defend in regulatory examination
- **Validation**: Test that adjusted thresholds maintain predictive validity

### Intervention Risks

**Credit Access Reduction:**
- Overly aggressive fairness interventions may reduce approvals for all groups
- Impact: Reduced revenue, market share loss, consumer harm
- Mitigation: Set minimum approval rate thresholds, monitor total credit volume

**Discriminatory Impact:**
- Misconfigured interventions can create new disparities (e.g., favoring one minority group over another)
- Impact: Regulatory violations, legal liability, reputational damage
- Mitigation: Test interventions on all protected groups, not just primary focus group

**Regulatory Non-Compliance:**
- Interventions that reduce explainability or violate fair lending laws
- Impact: Consent orders, fines ($10M+), required model cessation
- Mitigation: Legal review before deployment, continuous compliance monitoring

**Adverse Selection:**
- If intervention approves higher-risk applicants without appropriate pricing, default rates increase
- Impact: Financial losses, reduced lending capacity
- Mitigation: Risk-based pricing adjustments, monitoring default rates by group

### Validation Requirements

**Fair Lending Testing (Pre-Deployment):**
```bash
# 80% Rule Compliance Check
python validation/fair_lending_test.py \
  --model intervention_model.pkl \
  --data loan_applications.csv \
  --protected race ethnicity gender \
  --control_group White \
  --threshold 0.80 \
  --output fair_lending_report.json

# Expected Output:
# {
#   "race": {
#     "White": {"approval_rate": 0.62, "n": 10000},
#     "Black": {"approval_rate": 0.51, "n": 3000, "ratio": 0.82, "compliant": true},
#     "Hispanic": {"approval_rate": 0.48, "n": 2500, "ratio": 0.77, "compliant": false}
#   },
#   "overall_compliant": false,
#   "failing_groups": ["Hispanic"]
# }
```

**Adverse Impact Analysis:**
- Compare approval rates, denial rates, interest rates across protected groups
- Statistical significance testing (Fisher's exact test, chi-square)
- Regression analysis controlling for legitimate risk factors

**Explainability Audit:**
```bash
# Generate adverse action explanations for sample denials
python validation/adverse_action_audit.py \
  --model intervention_model.pkl \
  --denied_applicants denied_sample.csv \
  --top_n_reasons 4 \
  --output adverse_action_audit.json

# Validate that explanations:
# 1. Reference specific applicant characteristics
# 2. Cite legitimate credit factors (income, credit history, debt-to-income)
# 3. Do not reference protected attributes
# 4. Are consistent across similar applicants
```

**Regulatory Examination Preparation:**
- Document model development process (data sources, feature engineering, validation)
- Prepare model risk management documentation
- Conduct internal compliance review simulating regulatory examination
- Create executive summary for non-technical regulators

**A/B Testing (Pre-Full Deployment):**
- Deploy intervention to 10-20% of applications
- Monitor approval rates, default rates, customer complaints, regulatory inquiries
- Compare intervention cohort to control cohort over 6-12 months
- Full deployment only after validation period

### Finance Example: Loan Approval Intervention

**Scenario: Mid-Sized Bank Mortgage Lending**

**Baseline Problem:**
- Model approval rate: 65% (White), 48% (Black), 52% (Hispanic)
- 80% rule compliance: Black (0.74 ratio, non-compliant), Hispanic (0.80 ratio, marginally compliant)
- Regulatory examination flagged disparate impact
- Business requirement: Maintain >60% overall approval rate for profitability

**Intervention Strategy:**

```bash
# STEP 1: Baseline Compliance Check
python validation/fair_lending_test.py \
  --model baseline_model.pkl \
  --data mortgage_applications_2024.csv \
  --protected race ethnicity \
  --control_group White \
  --threshold 0.80 \
  --output compliance_baseline.json

# Result: Non-compliant (Black ratio 0.74)

# STEP 2: Pre-Processing - Disparate Impact Remover
python tests/techniques/pre_processing/M2-S2-P2-Feldman-2015-CertifyingAndRemovingDisparateImpact.py \
  --data mortgage_applications_2024.csv \
  --protected race \
  --repair_level 0.8 \
  --output preprocessed_data.csv

# STEP 3: In-Processing - Adversarial Debiasing with Explainability
python tests/techniques/in_processing/M2-S2-P1-Zhang-2018-AdversarialLearningforBiasMitigation.py \
  --data preprocessed_data.csv \
  --protected race ethnicity \
  --lambda 0.5 \
  --fairness_constraint demographic_parity \
  --explainability_mode shap \
  --min_approval_rate 0.60 \
  --output intervention_model.pkl

# STEP 4: Post-Processing - Threshold Optimization
python tests/techniques/post_processing/M2-S2-P3-Hardt-2016-EqualityOfOpportunity.py \
  --model intervention_model.pkl \
  --data mortgage_applications_2024.csv \
  --protected race ethnicity \
  --constraint equalized_odds \
  --output optimized_thresholds.json

# STEP 5: Post-Intervention Compliance Check
python validation/fair_lending_test.py \
  --model intervention_model.pkl \
  --thresholds optimized_thresholds.json \
  --data mortgage_applications_2024.csv \
  --protected race ethnicity \
  --control_group White \
  --threshold 0.80 \
  --output compliance_intervention.json

# Result: Compliant (Black ratio 0.93, Hispanic ratio 0.89)
# Overall approval rate: 61% (above 60% business requirement)

# STEP 6: Explainability Validation
python validation/adverse_action_audit.py \
  --model intervention_model.pkl \
  --thresholds optimized_thresholds.json \
  --denied_applicants denied_sample.csv \
  --top_n_reasons 4 \
  --output adverse_action_explanations.json

# STEP 7: Generate Regulatory Documentation
python validation/regulatory_report.py \
  --baseline_results compliance_baseline.json \
  --intervention_results compliance_intervention.json \
  --model_card intervention_model_card.json \
  --output regulatory_examination_report.pdf

# STEP 8: A/B Testing (3 months)
python deployment/ab_test.py \
  --control_model baseline_model.pkl \
  --treatment_model intervention_model.pkl \
  --treatment_proportion 0.15 \
  --duration_days 90 \
  --monitor approval_rate default_rate complaints \
  --output ab_test_results.json
```

**Results:**
- **Fairness**: Black approval ratio improved from 0.74 to 0.93 (compliant)
- **Business**: Overall approval rate 61% (above 60% threshold)
- **Explainability**: All adverse actions include 4+ specific reasons (FCRA compliant)
- **Regulatory**: Full audit trail documented, ready for examination
- **A/B Test**: Default rates unchanged (treatment 3.2% vs. control 3.1%), customer complaints unchanged

**Documentation for Regulatory Examination:**
1. Model Risk Management Framework document
2. Fair lending testing results (baseline vs. intervention)
3. Adverse action explanation samples (100 cases reviewed)
4. A/B test results (treatment vs. control performance)
5. Third-party model validation report
6. Executive summary for non-technical examiners

---

## 3. Healthcare Domain

### Regulatory Environment

**Key Legislation:**
- **HIPAA (Health Insurance Portability and Accountability Act)**: Protects patient privacy, regulates use of protected health information (PHI)
- **FDA Guidance on AI/ML Medical Devices**: Requires clinical validation, safety monitoring, and adverse event reporting for AI-based diagnostics and treatment tools
- **Clinical Trial Requirements**: Interventions affecting patient care may require IRB approval and clinical trial validation
- **Section 1557 of ACA**: Prohibits discrimination in healthcare on basis of race, color, national origin, sex, age, or disability

**Regulatory Bodies:**
- Food and Drug Administration (FDA)
- Institutional Review Boards (IRBs)
- Office for Civil Rights (OCR)
- Centers for Medicare & Medicaid Services (CMS)

### Key Stakeholders

- **Patients**: Individuals receiving diagnoses, treatment recommendations, or resource allocation decisions
- **Clinicians**: Physicians, nurses, radiologists using AI tools
- **Hospital Administrators**: Decision-makers for AI procurement and deployment
- **Regulatory Bodies**: FDA (for medical devices), IRBs (for clinical research)
- **Advocacy Groups**: Patient rights organizations, health equity advocates

### Common Applications

- Diagnostic tools (medical imaging, pathology, risk prediction)
- Treatment recommendation systems (clinical decision support)
- Resource allocation (ICU bed assignment, organ transplant prioritization, ventilator allocation)
- Patient triage (emergency department prioritization)
- Clinical trial recruitment
- Remote patient monitoring

### Fairness Definitions

**Primary: Equal Opportunity (Sensitivity/Specificity Parity)**
- True positive rates (sensitivity) equal across demographic groups
- True negative rates (specificity) equal across demographic groups
- Rationale: Ensures disease detection accuracy doesn't vary by protected group

**Secondary: Calibration (Clinical Validity)**
- Predicted probabilities match actual disease prevalence within groups
- Critical for risk-based treatment decisions
- Example: If model predicts 30% cancer risk for a group, actual prevalence should be ~30%

**Tertiary: Positive Predictive Value (PPV) Parity**
- Among patients flagged as high-risk, actual disease prevalence equal across groups
- Reduces unnecessary invasive procedures for false positives

### Intervention Constraints

**1. Safety Paramount (Do No Harm)**
- Interventions cannot reduce sensitivity or specificity below clinical standards
- Patient safety overrides fairness when conflict exists
- Example: If fairness intervention reduces cancer detection sensitivity from 0.92 to 0.87, intervention is unacceptable

**2. Protected Health Information (HIPAA Compliance)**
- All patient data must be de-identified or handled under HIPAA-compliant protocols
- Interventions cannot create new privacy risks
- Audit trails required for all PHI access

**3. Clinical Validation Required**
- Interventions affecting patient care require IRB approval
- Clinical trials may be necessary for high-risk applications
- Validation must demonstrate clinical utility, not just statistical fairness

**4. Do No Harm Principle**
- Interventions cannot increase patient harm (false negatives, delayed treatment, unnecessary procedures)
- Risk-benefit analysis required for all interventions
- Patient consent may be required for AI-assisted decisions

**5. Clinician Acceptance**
- AI tools must integrate into clinical workflows
- Clinicians must trust and understand AI recommendations
- Explainability critical for clinical adoption

### Intervention Considerations by Stage

**Pre-Processing:**
- **Permitted**: Reweighting, oversampling underrepresented groups, synthetic data generation
- **Constraint**: Cannot introduce systematic diagnostic errors
- **Risk**: Oversampling rare diseases may reduce specificity (more false positives)
- **Validation**: Clinical review of augmented data for realism

**In-Processing:**
- **Permitted**: Fairness-constrained optimization, multi-task learning (fairness + safety)
- **Constraint**: Safety constraints must override fairness when conflict exists
- **Risk**: Fairness regularization may reduce overall model accuracy
- **Validation**: Clinical validation on diverse patient populations

**Post-Processing:**
- **Permitted**: Threshold optimization, calibration adjustments, group-specific cutoffs
- **Constraint**: Threshold changes require clinical review and approval
- **Risk**: Different thresholds for different groups may be ethically problematic
- **Validation**: Clinical outcomes monitoring after threshold adjustment

### Intervention Risks

**Patient Harm from Biased Diagnostics:**
- False negatives: Missed diagnoses, delayed treatment, disease progression
- False positives: Unnecessary invasive procedures, patient anxiety, healthcare costs
- Impact: Patient morbidity/mortality, medical malpractice liability
- Mitigation: Safety constraints in intervention design, clinical validation, continuous monitoring

**Unequal Treatment Outcomes:**
- Interventions may improve fairness metrics without improving patient outcomes
- Example: Equal sensitivity across groups, but treatment access still unequal
- Impact: Persistent health disparities despite "fair" AI
- Mitigation: Monitor downstream outcomes (treatment, survival), not just AI metrics

**Liability for AI-Driven Clinical Decisions:**
- If AI tool causes patient harm, who is liable? (Developer, hospital, clinician)
- Fairness interventions may introduce new liability risks
- Impact: Legal exposure, insurance costs, clinician reluctance to use AI
- Mitigation: Legal review, informed consent, clinician override mechanisms

**Erosion of Clinical Trust:**
- If interventions reduce accuracy or create unexplainable recommendations, clinicians may distrust AI
- Impact: AI tool abandonment, reversion to biased human decision-making
- Mitigation: Clinician involvement in intervention design, transparent communication

### Validation Requirements

**Clinical Validation (Pre-Deployment):**

```bash
# Clinical Performance by Protected Group
python validation/clinical_validation.py \
  --model intervention_model.pkl \
  --test_data clinical_test_set.csv \
  --protected race gender age \
  --metrics sensitivity specificity ppv npv auroc \
  --min_sensitivity 0.90 \
  --min_specificity 0.85 \
  --output clinical_validation_report.json

# Expected Output:
# {
#   "race": {
#     "White": {"sensitivity": 0.93, "specificity": 0.87, "ppv": 0.82, "npv": 0.95, "auroc": 0.94},
#     "Black": {"sensitivity": 0.92, "specificity": 0.86, "ppv": 0.80, "npv": 0.94, "auroc": 0.93},
#     "Asian": {"sensitivity": 0.91, "specificity": 0.88, "ppv": 0.83, "npv": 0.95, "auroc": 0.94}
#   },
#   "all_groups_meet_minimums": true,
#   "max_sensitivity_gap": 0.02,
#   "max_specificity_gap": 0.02
# }
```

**IRB Approval:**
- Submit intervention protocol to Institutional Review Board
- Demonstrate patient safety, informed consent process, data privacy protections
- Required for interventions affecting patient care in research settings
- May require clinical trial for high-risk applications

**Continuous Monitoring of Patient Outcomes:**
```bash
# Post-Deployment Outcome Monitoring
python monitoring/patient_outcomes.py \
  --model intervention_model.pkl \
  --deployment_start 2024-01-01 \
  --deployment_end 2024-06-30 \
  --outcomes diagnosis_accuracy treatment_initiation survival_rate \
  --protected race gender \
  --alert_threshold 0.05 \
  --output outcome_monitoring_report.json

# Alerts triggered if:
# - Sensitivity drops >5% for any group
# - Treatment initiation rates diverge >10% between groups
# - Survival rates show unexpected disparities
```

**Re-Validation After Intervention:**
- Interventions may change model behavior in unexpected ways
- Re-run full clinical validation after any intervention
- Compare intervention model to baseline on held-out clinical data
- Document changes in clinical performance metrics

### Healthcare Example: Skin Cancer Diagnosis Intervention

**Scenario: Dermatology AI for Melanoma Detection**

**Baseline Problem:**
- Model sensitivity: 0.91 (White patients), 0.82 (Black patients), 0.85 (Hispanic patients)
- Clinical standard: Sensitivity ≥0.90 for all groups (missing 1 in 10 cancers is unacceptable)
- Specificity: 0.88 (White), 0.84 (Black), 0.86 (Hispanic)
- Issue: Black patients have 9% higher false negative rate (missed cancers)

**Clinical Impact:**
- Missed melanomas lead to delayed treatment, metastasis, reduced survival
- Dermatologists rely on AI tool for preliminary screening
- Health equity concern: AI perpetuates existing disparities in skin cancer outcomes

**Intervention Strategy:**

```bash
# STEP 1: Baseline Clinical Validation
python validation/clinical_validation.py \
  --model baseline_model.pkl \
  --test_data dermatology_test_set.csv \
  --protected race \
  --metrics sensitivity specificity ppv npv \
  --min_sensitivity 0.90 \
  --output baseline_clinical_validation.json

# Result: Black patients below minimum sensitivity (0.82 < 0.90)

# STEP 2: Data Augmentation (Pre-Processing)
# Problem: Training data has fewer examples of melanoma on darker skin
python tests/techniques/pre_processing/data_augmentation.py \
  --data dermatology_images/ \
  --protected race \
  --target_group Black Hispanic \
  --augmentation_factor 2.0 \
  --method synthetic_minority_oversampling \
  --output augmented_dermatology_data/

# STEP 3: In-Processing with Safety Constraints
python tests/techniques/in_processing/fairness_constrained_learning.py \
  --data augmented_dermatology_data/ \
  --protected race \
  --fairness_constraint equal_opportunity \
  --safety_constraint sensitivity:0.92 \
  --lambda 0.3 \
  --output intervention_model.pkl

# Safety constraint: Minimum sensitivity 0.92 (hard constraint, overrides fairness if conflict)
# Fairness constraint: Sensitivity parity across race groups (soft constraint)

# STEP 4: Post-Intervention Clinical Validation
python validation/clinical_validation.py \
  --model intervention_model.pkl \
  --test_data dermatology_test_set.csv \
  --protected race \
  --metrics sensitivity specificity ppv npv \
  --min_sensitivity 0.90 \
  --output intervention_clinical_validation.json

# Result: 
# - White: sensitivity 0.93, specificity 0.87
# - Black: sensitivity 0.92, specificity 0.86
# - Hispanic: sensitivity 0.92, specificity 0.87
# All groups meet minimum sensitivity, fairness gap reduced from 9% to 1%

# STEP 5: IRB Approval for Clinical Deployment
# Submit intervention protocol to IRB:
# - Patient population: Dermatology patients at 5 hospital sites
# - Intervention: AI-assisted melanoma screening with fairness-enhanced model
# - Primary outcome: Melanoma detection rate by race
# - Secondary outcomes: Biopsy rates, time to treatment, patient satisfaction
# - Safety monitoring: Monthly review of false negative cases
# - Informed consent: Patients informed AI tool used, can opt out

# STEP 6: Clinical Trial (6-month pilot)
python deployment/clinical_trial.py \
  --control_model baseline_model.pkl \
  --treatment_model intervention_model.pkl \
  --sites hospital_1 hospital_2 hospital_3 hospital_4 hospital_5 \
  --duration_months 6 \
  --randomization patient_level \
  --outcomes melanoma_detection_rate biopsy_rate time_to_treatment \
  --protected race \
  --output clinical_trial_results.json

# STEP 7: Continuous Outcome Monitoring (Post-Deployment)
python monitoring/patient_outcomes.py \
  --model intervention_model.pkl \
  --deployment_start 2024-07-01 \
  --outcomes melanoma_detection_rate stage_at_diagnosis survival_rate \
  --protected race \
  --alert_threshold 0.05 \
  --output outcome_monitoring_dashboard.json
```

**Results:**
- **Fairness**: Sensitivity gap reduced from 9% (0.91 vs. 0.82) to 1% (0.93 vs. 0.92)
- **Safety**: All groups meet clinical minimum sensitivity (≥0.90)
- **Clinical Trial**: Melanoma detection rate increased 8% for Black patients, unchanged for White patients
- **Patient Outcomes**: Time to treatment reduced 12 days for Black patients (earlier detection)
- **Clinician Feedback**: Dermatologists report increased confidence in AI recommendations for diverse patients

**IRB-Approved Monitoring Plan:**
- Monthly review of false negative cases (missed melanomas) by race
- Quarterly audit of biopsy rates (check for overdiagnosis)
- Annual survival analysis by race and stage at diagnosis
- Immediate alert if sensitivity drops >5% for any group

---

## 4. Criminal Justice Domain

### Constitutional Constraints

**Key Constitutional Protections:**
- **Due Process (5th and 14th Amendments)**: Right to fair procedures before deprivation of liberty
- **Equal Protection (14th Amendment)**: Prohibits discrimination by state actors based on race, ethnicity, or other protected characteristics
- **8th Amendment**: Prohibits cruel and unusual punishment
- **6th Amendment**: Right to confront witnesses and challenge evidence (implications for AI explainability)

**Legal Precedents:**
- **Bolling v. Sharpe (1954)**: Equal protection applies to federal government via 5th Amendment
- **Yick Wo v. Hopkins (1886)**: Facially neutral laws applied in discriminatory manner violate Equal Protection
- **State v. Loomis (2016)**: Wisconsin Supreme Court allows risk assessment tools but requires warnings about accuracy and bias

### Key Stakeholders

- **Defendants**: Individuals subject to risk assessments, sentencing recommendations, or pretrial detention decisions
- **Judges**: Decision-makers using AI tools for sentencing, bail, parole
- **Prosecutors**: May use AI for charging decisions, plea bargaining
- **Public Defenders**: Challenge AI tools on behalf of defendants
- **Victims**: May have interests in accurate risk assessment for public safety
- **Advocacy Groups**: Criminal justice reform organizations, civil rights groups

### Common Applications

- Recidivism prediction (risk of reoffending)
- Pretrial risk assessment (bail decisions)
- Sentencing recommendations (guidelines, aggravating/mitigating factors)
- Parole decisions (early release eligibility)
- Resource allocation (probation supervision intensity)
- Predictive policing (crime hotspot prediction)

### Fairness Definitions

**Primary: Calibration (Within-Group Accuracy)**
- Predicted risk scores match actual recidivism rates within demographic groups
- Example: If 100 Black defendants scored "high risk" (70% predicted recidivism), ~70 should actually reoffend
- Rationale: Ensures predictions are equally accurate across groups

**Secondary: Equal Opportunity (False Positive Rate Parity)**
- Among defendants who do not reoffend, false positive rates equal across groups
- Reduces wrongful detention/incarceration for non-recidivists
- Critical for pretrial detention decisions

**Tertiary: Individual Fairness**
- Similar individuals receive similar risk scores, regardless of group membership
- Addresses concerns about group-based stereotyping

**Note: Demographic Parity Problematic**
- Equal risk score distributions across groups may not be appropriate if base rates differ
- However, base rate differences may themselves reflect systemic bias in policing and prosecution

### Intervention Constraints

**1. Transparency Requirements (Explainability for Judicial Review)**
- Defendants have right to understand and challenge AI-driven decisions
- Judges must be able to explain sentencing rationale
- "Black box" models face constitutional challenges under Due Process Clause
- Explainability must be accessible to non-technical legal actors

**2. Prohibition on Direct Use of Race**
- Equal Protection Clause prohibits explicit consideration of race in sentencing/bail
- Even for "fairness corrections," direct use of race is constitutionally suspect
- Must use race-neutral proxies (geography, education, employment) with caution
- Exception: Monitoring and auditing for discrimination (not decision-making)

**3. Due Process (Right to Challenge AI Decisions)**
- Defendants must have opportunity to contest AI risk scores
- Procedural safeguards required before AI-assisted deprivation of liberty
- Human decision-maker must have discretion to override AI recommendations
- Disclosure of AI methodology may be required for meaningful challenge

**4. Avoid Disparate Impact**
- Facially neutral AI tools that produce discriminatory outcomes may violate Equal Protection
- Interventions must reduce, not exacerbate, racial disparities
- Statistical evidence of disparate impact can support legal challenges

**5. Validation with Legal Experts**
- Interventions must be reviewed by legal experts for constitutional compliance
- Technical fairness metrics don't guarantee legal compliance
- Legal review should precede deployment in criminal justice settings

### Intervention Considerations by Stage

**Pre-Processing:**
- **Permitted**: Reweighting, removing biased features (e.g., arrest history in high-policing areas), synthetic data generation
- **Constraint**: Cannot directly use race to adjust data
- **Risk**: Removing legitimate risk factors may reduce predictive accuracy, increasing public safety risk
- **Validation**: Legal review of feature removal rationale

**In-Processing:**
- **Permitted**: Fairness-constrained optimization (calibration, equal opportunity), adversarial debiasing
- **Constraint**: Model must remain explainable to judges and defendants
- **Risk**: Fairness constraints may reduce predictive accuracy, affecting public safety and wrongful detention rates
- **Validation**: Explainability audit, legal review

**Post-Processing:**
- **Permitted**: Threshold optimization, calibration adjustments
- **Constraint**: Different thresholds for different groups may violate Equal Protection (group-based treatment)
- **Risk**: Threshold adjustments may be perceived as "racial quotas" in criminal justice
- **Validation**: Legal review of threshold adjustment rationale, constitutional analysis

### Intervention Risks

**Wrongful Incarceration:**
- False positives lead to pretrial detention or harsher sentences for non-recidivists
- Disproportionate impact on minorities if fairness intervention misconfigured
- Impact: Liberty deprivation, family disruption, employment loss, collateral consequences
- Mitigation: Prioritize false positive rate parity, human override mechanisms

**Discriminatory Sentencing:**
- Interventions may create new disparities (e.g., favoring one minority group over another)
- Historical bias in training data (arrests, convictions) may persist despite intervention
- Impact: Perpetuation of systemic racism, legal challenges, loss of legitimacy
- Mitigation: Audit for intersectional disparities, validate on historically discriminated groups

**Due Process Violations:**
- Unexplainable AI tools may violate defendants' right to challenge evidence
- Lack of transparency in AI methodology may constitute procedural unfairness
- Impact: Appellate reversals, constitutional challenges, case dismissals
- Mitigation: Explainability requirements, disclosure of AI methodology, human oversight

**Constitutional Challenges to AI Systems:**
- Legal challenges under Equal Protection, Due Process, 6th Amendment
- Interventions may be subject to strict scrutiny if race-conscious
- Impact: Court orders to cease AI use, policy changes, reputational damage
- Mitigation: Legal review before deployment, proactive transparency, stakeholder engagement

### Validation Requirements

**Calibration Testing (Within-Group Accuracy):**

```bash
# Calibration by Protected Group
python validation/calibration_test.py \
  --model intervention_model.pkl \
  --test_data recidivism_test_set.csv \
  --protected race \
  --risk_bins 10 \
  --output calibration_report.json

# Expected Output:
# {
#   "race": {
#     "White": {
#       "bin_1_predicted": 0.10, "bin_1_actual": 0.12,
#       "bin_10_predicted": 0.90, "bin_10_actual": 0.88,
#       "calibration_error": 0.03
#     },
#     "Black": {
#       "bin_1_predicted": 0.10, "bin_1_actual": 0.11,
#       "bin_10_predicted": 0.90, "bin_10_actual": 0.87,
#       "calibration_error": 0.03
#     }
#   },
#   "max_calibration_error": 0.03,
#   "well_calibrated": true
# }
```

**Equal Opportunity Analysis (False Positive/Negative Rates):**

```bash
# False Positive Rate Parity (Among Non-Recidivists)
python validation/equal_opportunity_test.py \
  --model intervention_model.pkl \
  --test_data recidivism_test_set.csv \
  --protected race \
  --outcome recidivism \
  --output equal_opportunity_report.json

# Expected Output:
# {
#   "false_positive_rate": {
#     "White": 0.18,
#     "Black": 0.19,
#     "Hispanic": 0.17,
#     "max_gap": 0.02
#   },
#   "false_negative_rate": {
#     "White": 0.22,
#     "Black": 0.21,
#     "Hispanic": 0.23,
#     "max_gap": 0.02
#   },
#   "equal_opportunity_satisfied": true
# }
```

**Explainability Audit (Can Decisions Be Explained in Court?):**

```bash
# Generate explanations for high-risk classifications
python validation/explainability_audit.py \
  --model intervention_model.pkl \
  --high_risk_defendants high_risk_sample.csv \
  --explanation_method decision_tree_surrogate \
  --output explainability_audit.json

# Validate that explanations:
# 1. Reference specific defendant characteristics (prior convictions, age, employment)
# 2. Do not reference race directly
# 3. Are understandable to judges and defendants (no technical jargon)
# 4. Are consistent across similar defendants
# 5. Can withstand cross-examination by defense attorneys
```

**Legal Review Before Deployment:**
- Constitutional analysis by legal experts (Due Process, Equal Protection)
- Review of intervention methodology for race-neutrality
- Assessment of explainability for judicial review
- Stakeholder engagement (public defenders, civil rights groups)
- Policy review for alignment with criminal justice reform goals

### Criminal Justice Example: Recidivism Prediction Intervention

**Scenario: State Court System Pretrial Risk Assessment**

**Baseline Problem:**
- Model predicts recidivism risk for pretrial detention decisions
- Calibration issues: Black defendants over-predicted (predicted 65% recidivism, actual 52%)
- False positive rate disparity: 28% (Black) vs. 19% (White) among non-recidivists
- Impact: Black defendants disproportionately detained pretrial, lose jobs, families disrupted
- Legal concern: Potential Equal Protection and Due Process violations

**Constitutional Considerations:**
- Cannot use race directly in model (Equal Protection Clause)
- Must be explainable for defendants to challenge (Due Process)
- Must not create disparate impact (Yick Wo v. Hopkins)
- Judges must retain discretion to override AI recommendations

**Intervention Strategy:**

```bash
# STEP 1: Baseline Calibration and Fairness Analysis
python validation/calibration_test.py \
  --model baseline_model.pkl \
  --test_data pretrial_defendants.csv \
  --protected race \
  --output baseline_calibration.json

python validation/equal_opportunity_test.py \
  --model baseline_model.pkl \
  --test_data pretrial_defendants.csv \
  --protected race \
  --output baseline_equal_opportunity.json

# Result: 
# - Calibration error: Black 0.13 (over-prediction), White 0.04
# - False positive rate: Black 0.28, White 0.19 (9% gap)

# STEP 2: Feature Audit (Remove Biased Features)
python tests/techniques/pre_processing/feature_audit.py \
  --data pretrial_defendants.csv \
  --protected race \
  --identify_proxies true \
  --output biased_features.json

# Identified biased features:
# - "arrests_in_high_policing_zip": Proxy for race (over-policing in Black neighborhoods)
# - "public_defender": Proxy for income and race
# Remove these features before retraining

# STEP 3: In-Processing with Calibration Constraint (No Direct Use of Race)
python tests/techniques/in_processing/calibration_constrained_learning.py \
  --data pretrial_defendants_cleaned.csv \
  --protected race \
  --use_protected false \
  --fairness_constraint calibration \
  --lambda 0.4 \
  --explainability_mode decision_tree \
  --output intervention_model.pkl

# Constraint: Calibration across race groups (predicted risk matches actual recidivism)
# Explainability: Decision tree surrogate for interpretability
# Race not used as feature (constitutional requirement)

# STEP 4: Post-Intervention Validation
python validation/calibration_test.py \
  --model intervention_model.pkl \
  --test_data pretrial_defendants.csv \
  --protected race \
  --output intervention_calibration.json

python validation/equal_opportunity_test.py \
  --model intervention_model.pkl \
  --test_data pretrial_defendants.csv \
  --protected race \
  --output intervention_equal_opportunity.json

# Result:
# - Calibration error: Black 0.04, White 0.04 (calibration achieved)
# - False positive rate: Black 0.20, White 0.19 (1% gap, down from 9%)

# STEP 5: Explainability Audit
python validation/explainability_audit.py \
  --model intervention_model.pkl \
  --high_risk_defendants high_risk_sample.csv \
  --explanation_method decision_tree_surrogate \
  --output explainability_audit.json

# Sample explanation for defendant ID 12345 (Black, classified high risk):
# - Prior felony convictions: 2 (weight: 0.35)
# - Age at first arrest: 16 (weight: 0.25)
# - Unemployed: Yes (weight: 0.20)
# - Failed to appear in past: Yes (weight: 0.15)
# - Current charge severity: Felony (weight: 0.05)
# Predicted risk: 68% (actual recidivism rate for similar defendants: 67%)

# STEP 6: Legal Review
# Constitutional analysis by legal experts:
# - Due Process: Model is explainable, defendants can challenge risk scores
# - Equal Protection: No direct use of race, calibration reduces disparate impact
# - Discretion: Judges retain authority to override AI recommendations
# - Transparency: Methodology disclosed to defense attorneys

# STEP 7: Pilot Deployment with Judicial Training
python deployment/pilot_deployment.py \
  --model intervention_model.pkl \
  --courts court_1 court_2 court_3 \
  --duration_months 6 \
  --judge_training true \
  --human_override_required true \
  --output pilot_results.json

# Judicial training includes:
# - How to interpret risk scores
# - Limitations of AI predictions
# - Constitutional obligations (Due Process, Equal Protection)
# - When to override AI recommendations
# - How to explain decisions to defendants

# STEP 8: Continuous Monitoring
python monitoring/fairness_monitoring.py \
  --model intervention_model.pkl \
  --deployment_start 2024-01-01 \
  --metrics calibration false_positive_rate detention_rate \
  --protected race \
  --alert_threshold 0.05 \
  --output monitoring_dashboard.json

# Alerts triggered if:
# - Calibration error increases >5% for any group
# - False positive rate gap increases >5%
# - Detention rates diverge unexpectedly
```

**Results:**
- **Fairness**: Calibration error reduced from 0.13 to 0.04 for Black defendants (over-prediction corrected)
- **False Positive Rate**: Gap reduced from 9% to 1% (fewer wrongful detentions of Black non-recidivists)
- **Explainability**: Decision tree surrogate provides understandable explanations for judges and defendants
- **Constitutional Compliance**: Legal review confirms Due Process and Equal Protection compliance
- **Judicial Acceptance**: Judges report increased confidence in risk scores, use override authority in 8% of cases
- **Pretrial Detention Rate**: Black detention rate decreased 12%, White rate unchanged (disparity reduced)

**Legal Safeguards:**
- Defendants receive written explanation of risk score factors
- Defense attorneys can challenge AI methodology and individual risk scores
- Judges must document rationale for accepting or overriding AI recommendations
- Annual audit by independent legal experts for constitutional compliance
- Public reporting of fairness metrics by race, ethnicity, gender

---

## 5. Problem Type Adaptations

### Classification (Binary and Multi-Class)

**Problem Structure:**
- **Binary**: Two outcomes (e.g., loan approved/denied, disease present/absent, high risk/low risk)
- **Multi-Class**: Multiple outcomes (e.g., risk tiers: low/medium/high/very high; credit scores: excellent/good/fair/poor)

**Fairness Definitions:**
- **Demographic Parity**: Outcome rates equal across groups (e.g., approval rate parity)
- **Equal Opportunity**: True positive rates equal across groups (qualified individuals treated equally)
- **Equalized Odds**: True positive and false positive rates equal across groups
- **Calibration**: Predicted probabilities match actual outcome rates within groups

**Intervention Techniques:**
- **Pre-Processing**: Reweighting, disparate impact remover, data augmentation
- **In-Processing**: Fairness-constrained optimization, adversarial debiasing, regularization
- **Post-Processing**: Threshold optimization, calibration adjustments, reject option classification

**Domain Examples:**
- **Finance**: Loan approval (binary), credit risk tier (multi-class)
- **Healthcare**: Disease diagnosis (binary), patient triage (multi-class: immediate/urgent/non-urgent)
- **Criminal Justice**: Recidivism prediction (binary), risk tier assignment (multi-class)

**Implementation Guidance:**

```bash
# Binary Classification: Loan Approval
python tests/techniques/in_processing/M2-S2-P1-Zhang-2018-AdversarialLearningforBiasMitigation.py \
  --data loan_applications.csv \
  --protected race gender \
  --task binary_classification \
  --fairness_constraint demographic_parity \
  --lambda 0.5 \
  --output loan_approval_model.pkl

# Multi-Class Classification: Risk Tier Assignment
python tests/techniques/in_processing/fairness_constrained_multiclass.py \
  --data pretrial_defendants.csv \
  --protected race \
  --task multiclass_classification \
  --classes low medium high very_high \
  --fairness_constraint equal_opportunity \
  --lambda 0.4 \
  --output risk_tier_model.pkl

# Validation: Check fairness across all classes
python validation/multiclass_fairness_test.py \
  --model risk_tier_model.pkl \
  --test_data pretrial_test.csv \
  --protected race \
  --classes low medium high very_high \
  --output multiclass_fairness_report.json
```

**Multi-Class Fairness Considerations:**
- Fairness metrics must be evaluated for all classes, not just overall accuracy
- Class imbalance may affect fairness (e.g., few "very high risk" cases)
- Interventions may improve fairness for some classes while worsening others
- Consider pairwise fairness (e.g., low vs. medium, medium vs. high) in addition to overall fairness

---

### Regression (Continuous Outcomes)

**Problem Structure:**
- Predict continuous values (e.g., loan amount, insurance premium, treatment dosage, sentence length)
- No clear decision threshold (unlike classification)

**Fairness Definitions:**
- **Mean Prediction Parity**: Average predicted values equal across groups
- **Error Parity**: Prediction errors (MAE, RMSE) equal across groups
- **Calibration**: Predicted values match actual values within groups (no systematic over/under-prediction)

**Intervention Techniques:**
- **Pre-Processing**: Reweighting, feature transformation, data augmentation
- **In-Processing**: Fairness regularization (penalize prediction disparities), multi-task learning
- **Post-Processing**: Less applicable (no thresholds to adjust), but can apply calibration corrections

**Challenge: Post-Processing Limited:**
- Classification post-processing relies on threshold adjustment
- Regression has no thresholds, so post-processing techniques less effective
- Calibration adjustments possible (e.g., shift predictions for underestimated groups)

**Domain Examples:**
- **Finance**: Loan amount prediction, insurance premium calculation, credit limit assignment
- **Healthcare**: Treatment dosage recommendation, hospital length of stay prediction, medical cost estimation
- **Criminal Justice**: Sentence length recommendation (in jurisdictions using AI for sentencing guidelines)

**Implementation Guidance:**

```bash
# Regression: Loan Amount Prediction
python tests/techniques/in_processing/fairness_constrained_regression.py \
  --data loan_applications.csv \
  --protected race gender \
  --task regression \
  --target loan_amount \
  --fairness_constraint mean_prediction_parity \
  --lambda 0.3 \
  --output loan_amount_model.pkl

# Validation: Check prediction parity and error parity
python validation/regression_fairness_test.py \
  --model loan_amount_model.pkl \
  --test_data loan_test.csv \
  --protected race gender \
  --metrics mean_prediction mae rmse \
  --output regression_fairness_report.json

# Expected Output:
# {
#   "race": {
#     "White": {"mean_prediction": 245000, "mae": 18000, "rmse": 24000},
#     "Black": {"mean_prediction": 238000, "mae": 19000, "rmse": 25000},
#     "Hispanic": {"mean_prediction": 242000, "mae": 18500, "rmse": 24500}
#   },
#   "mean_prediction_gap": 7000,
#   "mae_gap": 1000,
#   "fairness_acceptable": true
# }

# Calibration Correction (Post-Processing)
python tests/techniques/post_processing/calibration_correction_regression.py \
  --model loan_amount_model.pkl \
  --data loan_applications.csv \
  --protected race \
  --method group_specific_shift \
  --output calibrated_loan_amount_model.pkl

# Method: Shift predictions for underestimated groups to match mean actual values
```

**Regression Fairness Considerations:**
- Mean prediction parity may not be appropriate if groups have different legitimate needs (e.g., income differences)
- Error parity ensures prediction quality equal across groups
- Calibration critical: Systematic over/under-prediction can lead to discriminatory outcomes
- Domain expertise needed to define "fair" continuous outcomes

---

### Ranking (Ordered Lists)

**Problem Structure:**
- Produce ordered lists of items (e.g., job candidates, college applicants, search results, loan applicants)
- Fairness concerns: Representation at top of list, exposure across positions

**Fairness Definitions:**
- **Exposure Fairness**: Protected groups receive proportional exposure across list positions (top-k representation)
- **Position-Based Fairness**: Probability of appearing in top-k positions equal across groups (for equally qualified items)
- **Demographic Parity in Top-k**: Protected group representation in top-k matches base rate or qualification rate

**Intervention Techniques:**
- **Post-Processing**: Re-ranking algorithms (adjust list order to improve fairness), stochastic ranking
- **In-Processing**: Fairness-aware ranking loss functions, learning to rank with fairness constraints

**Challenge: Fairness vs. Relevance Trade-off:**
- Ranking by pure relevance/qualification may produce unfair lists
- Fairness interventions may reduce relevance (e.g., promoting less-qualified candidates)
- Need to balance fairness and utility

**Domain Examples:**
- **Finance**: Loan applicant ranking for manual review, credit card offer targeting
- **Healthcare**: Organ transplant waitlist, clinical trial recruitment prioritization
- **Criminal Justice**: Parole candidate ranking, diversion program eligibility ranking
- **Employment**: Job candidate ranking, promotion candidate ranking

**Implementation Guidance:**

```bash
# Ranking: Job Candidate Ranking
python tests/techniques/post_processing/fair_ranking.py \
  --data job_candidates.csv \
  --protected gender race \
  --ranking_score qualification_score \
  --fairness_constraint top_k_representation \
  --k 20 \
  --target_representation proportional \
  --output fair_ranked_candidates.csv

# Validation: Check top-k representation
python validation/ranking_fairness_test.py \
  --ranked_list fair_ranked_candidates.csv \
  --protected gender race \
  --k_values 10 20 50 \
  --output ranking_fairness_report.json

# Expected Output:
# {
#   "top_10": {
#     "gender": {"Female": 0.40, "Male": 0.60},
#     "race": {"White": 0.60, "Black": 0.20, "Hispanic": 0.10, "Asian": 0.10}
#   },
#   "top_20": {
#     "gender": {"Female": 0.45, "Male": 0.55},
#     "race": {"White": 0.55, "Black": 0.25, "Hispanic": 0.10, "Asian": 0.10}
#   },
#   "base_rate": {
#     "gender": {"Female": 0.48, "Male": 0.52},
#     "race": {"White": 0.52, "Black": 0.28, "Hispanic": 0.12, "Asian": 0.08}
#   },
#   "representation_gap_top_20": {"Female": 0.03, "Black": 0.03}
# }

# In-Processing: Fairness-Aware Learning to Rank
python tests/techniques/in_processing/fairness_aware_learning_to_rank.py \
  --data job_candidates.csv \
  --protected gender race \
  --features qualification_score experience education \
  --fairness_constraint exposure_fairness \
  --lambda 0.4 \
  --output fair_ranking_model.pkl
```

**Ranking Fairness Considerations:**
- Top-k representation critical (most attention/resources go to top-ranked items)
- Position bias: Items at top of list receive disproportionate attention
- Stochastic ranking: Randomize rankings within fairness constraints to provide exposure over time
- Domain-specific: Job rankings may prioritize qualification, college admissions may prioritize diversity
- Legal considerations: Some jurisdictions prohibit "quotas" in ranking (e.g., employment)

---

## 6. Adaptation Decision Trees

### Domain Selection Decision Tree

```
┌─────────────────────────────────────────────────────────┐
│ STEP 1: What domain are you working in?                │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
    ┌───▼────┐      ┌────▼─────┐     ┌────▼──────┐
    │FINANCE │      │HEALTHCARE│     │CRIMINAL   │
    │        │      │          │     │JUSTICE    │
    └───┬────┘      └────┬─────┘     └────┬──────┘
        │                │                 │
        │                │                 │
┌───────▼────────────────────────────────────────────────┐
│ FINANCE DOMAIN                                         │
├────────────────────────────────────────────────────────┤
│ Regulations: ECOA, FCRA, Regulation B, Fair Lending   │
│ Key Constraint: Explainability (adverse action notices)│
│ Fairness Priority: Demographic Parity (80% rule)      │
│ Validation: Fair lending testing, regulatory docs     │
│ Risk: Credit access reduction, regulatory fines       │
└────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 2: What is your specific application?             │
├─────────────────────────────────────────────────────────┤
│ ☐ Credit approval (loans, credit cards)                │
│ ☐ Loan pricing (interest rates)                        │
│ ☐ Credit limit assignment                              │
│ ☐ Insurance underwriting                               │
│ ☐ Fraud detection                                      │
└─────────────────────────────────────────────────────────┘
        │
        ▼ (Example: Credit approval selected)
┌─────────────────────────────────────────────────────────┐
│ STEP 3: Select Intervention Approach                   │
├─────────────────────────────────────────────────────────┤
│ Recommended Pipeline:                                   │
│ 1. Pre-Processing: Disparate Impact Remover (repair 0.8│
│ 2. In-Processing: Adversarial Debiasing + SHAP         │
│ 3. Post-Processing: Threshold Optimization             │
│ 4. Validation: Fair lending test, adverse action audit │
│ 5. Deployment: A/B test (3-6 months)                   │
└─────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────┐
│ HEALTHCARE DOMAIN                                     │
├───────────────────────────────────────────────────────┤
│ Regulations: HIPAA, FDA, IRB, Section 1557 ACA       │
│ Key Constraint: Safety (cannot reduce sensitivity)   │
│ Fairness Priority: Equal Opportunity (sensitivity     │
│                    parity), Calibration               │
│ Validation: Clinical validation, IRB approval,        │
│             patient outcome monitoring                │
│ Risk: Patient harm, unequal treatment outcomes       │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 2: What is your specific application?             │
├─────────────────────────────────────────────────────────┤
│ ☐ Diagnostic tool (imaging, pathology, risk prediction)│
│ ☐ Treatment recommendation                             │
│ ☐ Resource allocation (ICU beds, organs, ventilators)  │
│ ☐ Patient triage                                       │
│ ☐ Clinical trial recruitment                           │
└─────────────────────────────────────────────────────────┘
        │
        ▼ (Example: Diagnostic tool selected)
┌─────────────────────────────────────────────────────────┐
│ STEP 3: Select Intervention Approach                   │
├─────────────────────────────────────────────────────────┤
│ Recommended Pipeline:                                   │
│ 1. Pre-Processing: Data augmentation (oversample       │
│    underrepresented groups)                             │
│ 2. In-Processing: Fairness-constrained learning +      │
│    SAFETY CONSTRAINT (min sensitivity)                  │
│ 3. Validation: Clinical validation (sensitivity,       │
│    specificity by group), IRB approval                  │
│ 4. Deployment: Clinical trial (6-12 months)            │
│ 5. Monitoring: Continuous patient outcome tracking     │
└─────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────┐
│ CRIMINAL JUSTICE DOMAIN                               │
├───────────────────────────────────────────────────────┤
│ Regulations: Due Process, Equal Protection, 8th Amend│
│ Key Constraint: Transparency (explainability for      │
│                 judicial review), No direct use of    │
│                 race                                  │
│ Fairness Priority: Calibration (within-group         │
│                    accuracy), Equal Opportunity       │
│                    (false positive rate parity)       │
│ Validation: Calibration testing, legal review,       │
│             explainability audit                      │
│ Risk: Wrongful incarceration, discriminatory         │
│       sentencing, constitutional challenges           │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 2: What is your specific application?             │
├─────────────────────────────────────────────────────────┤
│ ☐ Recidivism prediction                                │
│ ☐ Pretrial risk assessment (bail decisions)            │
│ ☐ Sentencing recommendations                           │
│ ☐ Parole decisions                                     │
│ ☐ Resource allocation (probation supervision)          │
└─────────────────────────────────────────────────────────┘
        │
        ▼ (Example: Pretrial risk assessment selected)
┌─────────────────────────────────────────────────────────┐
│ STEP 3: Select Intervention Approach                   │
├─────────────────────────────────────────────────────────┤
│ Recommended Pipeline:                                   │
│ 1. Pre-Processing: Feature audit (remove biased        │
│    proxies like high-policing zip codes)                │
│ 2. In-Processing: Calibration-constrained learning     │
│    (NO DIRECT USE OF RACE) + Decision tree explainer   │
│ 3. Validation: Calibration test, equal opportunity     │
│    analysis, explainability audit, legal review        │
│ 4. Deployment: Pilot with judicial training (6 months) │
│ 5. Monitoring: Continuous fairness monitoring,         │
│    human override tracking                              │
└─────────────────────────────────────────────────────────┘
```

---

### Problem Type Selection Decision Tree

```
┌─────────────────────────────────────────────────────────┐
│ STEP 1: What type of prediction are you making?        │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
  ┌─────▼──────┐   ┌─────▼──────┐   ┌─────▼──────┐
  │CLASSIFICA- │   │REGRESSION  │   │RANKING     │
  │TION        │   │            │   │            │
  └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
        │                │                 │
        │                │                 │
┌───────▼────────────────────────────────────────────────┐
│ CLASSIFICATION (Binary or Multi-Class)                 │
├────────────────────────────────────────────────────────┤
│ Output: Discrete categories (approved/denied,          │
│         low/medium/high risk)                          │
│ Fairness Metrics: Demographic parity, equal           │
│                   opportunity, equalized odds,         │
│                   calibration                          │
│ Intervention Techniques: ALL (pre, in, post)          │
│ Post-Processing: Threshold optimization effective     │
└────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ Binary or Multi-Class?                                  │
├─────────────────────────────────────────────────────────┤
│ ☐ Binary (2 outcomes): Use standard fairness metrics   │
│ ☐ Multi-Class (3+ outcomes): Check fairness for ALL    │
│   classes, not just overall accuracy                    │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ Example Implementation (Binary Classification)          │
├─────────────────────────────────────────────────────────┤
│ python in_processing_adversarial_debiasing.py \         │
│   --data loan_data.csv \                                │
│   --protected race gender \                             │
│   --task binary_classification \                        │
│   --fairness_constraint demographic_parity \            │
│   --lambda 0.5 \                                        │
│   --output model.pkl                                    │
└─────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────┐
│ REGRESSION (Continuous Outcomes)                      │
├───────────────────────────────────────────────────────┤
│ Output: Continuous values (loan amount, dosage,       │
│         sentence length)                              │
│ Fairness Metrics: Mean prediction parity, error      │
│                   parity (MAE, RMSE), calibration     │
│ Intervention Techniques: Pre-processing, in-processing│
│ Post-Processing: LIMITED (no thresholds), calibration │
│                  corrections possible                 │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ Key Consideration: Define "Fair" Continuous

---



# Prediction

**Context**: In regression tasks, fairness is more nuanced than classification. There are no binary decisions, so we must consider:
- **Prediction parity**: Are average predictions similar across groups?
- **Error parity**: Are prediction errors (MAE, RMSE) similar across groups?
- **Calibration**: Are predictions well-calibrated for all groups?

**Example**: A salary prediction model might predict accurately for men (RMSE = $5,000) but poorly for women (RMSE = $12,000), indicating unfair performance.

---

## 📊 Regression Fairness Metrics

### 1. **Mean Prediction Parity**
- **Definition**: Average predictions should be similar across protected groups
- **Formula**: `|E[ŷ|A=0] - E[ŷ|A=1]| < ε`
- **Use Case**: Ensuring loan amounts, salary estimates, or credit limits aren't systematically different

### 2. **Error Rate Parity**
- **MAE Parity**: `|MAE(A=0) - MAE(A=1)| < ε`
- **RMSE Parity**: `|RMSE(A=0) - RMSE(A=1)| < ε`
- **Use Case**: Model should be equally accurate for all groups

### 3. **Calibration Fairness**
- **Definition**: Predicted values should match actual values across groups
- **Formula**: `E[y|ŷ=v, A=a] ≈ v` for all groups a
- **Use Case**: Risk scores, price estimates should be well-calibrated per group

### 4. **Conditional Statistical Parity**
- **Definition**: Predictions should be similar across groups given legitimate features
- **Formula**: `E[ŷ|X=x, A=0] ≈ E[ŷ|X=x, A=1]`

---

## 🛠️ Bias Mitigation Techniques for Regression

### **Pre-Processing**
1. **Reweighing**: Adjust sample weights to balance group representation
2. **Data Augmentation**: Oversample underrepresented groups
3. **Feature Engineering**: Remove or transform biased features

### **In-Processing**
1. **Fairness Constraints**: Add penalty terms to loss function
   ```python
   loss = MSE + λ * |E[ŷ|A=0] - E[ŷ|A=1]|
   ```
2. **Adversarial Debiasing**: Train model to be unable to predict protected attribute
3. **Fair Representation Learning**: Learn embeddings that are group-invariant

### **Post-Processing**
1. **Calibration Correction**: Adjust predictions per group to achieve calibration
2. **Residual Adjustment**: Apply group-specific corrections to reduce error disparities
3. **Quantile Matching**: Align prediction distributions across groups

**Note**: Post-processing in regression is more limited than classification since there are no decision thresholds to adjust.

---

## 💼 Real-World Example: Fair Housing Price Prediction

**Scenario**: A real estate platform predicts home values for mortgage lending decisions.

**Problem Identified**:
- Predictions in predominantly minority neighborhoods systematically undervalue homes by 8%
- RMSE for minority areas: $45,000
- RMSE for non-minority areas: $28,000

**Fairness Intervention**:
1. **Pre-processing**: Remove ZIP code features that encode racial composition
2. **In-processing**: Add fairness constraint:
   ```
   minimize: MSE + 0.3 * |MAE(minority) - MAE(non-minority)|
   ```
3. **Post-processing**: Apply calibration correction to align predictions with actual sale prices per neighborhood type

**Results**:
- RMSE gap reduced from $17,000 to $6,000
- Mean prediction parity improved from 8% to 2% difference
- Overall model accuracy maintained (R² = 0.87)

---

## ⚖️ Trade-offs and Considerations

### **Accuracy vs. Fairness**
- Enforcing strict fairness constraints may reduce overall model accuracy
- Must balance business objectives with fairness goals
- Document trade-off decisions explicitly

### **Choice of Fairness Metric**
- Different metrics may conflict (can't optimize all simultaneously)
- Stakeholder input crucial for defining "fairness" in context
- Legal and ethical requirements should guide metric selection

### **Group Definition**
- How are protected groups defined? (intersectionality matters)
- Small group sizes lead to high variance in fairness metrics
- Consider multiple levels of granularity

---

## 🔍 Monitoring and Auditing

### **Continuous Monitoring**
- Track fairness metrics alongside performance metrics
- Set up alerts for fairness metric degradation
- Regular audits (quarterly recommended)

### **Documentation Requirements**
- Model cards documenting fairness evaluations
- Bias testing results and mitigation steps
- Stakeholder sign-off on fairness definitions

### **Regulatory Compliance**
- Fair Housing Act (real estate)
- Equal Credit Opportunity Act (lending)
- GDPR Article 22 (automated decisions)
- Local regulations (e.g., NYC Local Law 144 for hiring)

---

## 🎯 Key Takeaways

✅ **Regression fairness is multifaceted**: No single metric captures all aspects  
✅ **Context matters**: Define fairness based on domain, stakeholders, and regulations  
✅ **Multiple intervention points**: Pre-, in-, and post-processing each have roles  
✅ **Trade-offs are inevitable**: Balance accuracy, fairness, and business objectives  
✅ **Continuous vigilance**: Fairness is not "one and done" - requires ongoing monitoring  

---

## 📚 Additional Resources

- **Fairlearn**: Microsoft's toolkit for fairness assessment and mitigation
- **AI Fairness 360**: IBM's comprehensive fairness toolkit
- **Google's What-If Tool**: Interactive fairness exploration
- **Research**: "Fairness in Machine Learning" (Barocas, Hardt, Narayanan)

---

*End of Module 2 Consolidation Document*