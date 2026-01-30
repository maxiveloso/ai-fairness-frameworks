# Adaptability Guidelines Synthesis

## **1. Adaptability Philosophy**

### **Core Principles:**

The Fairness Audit Playbook is designed to be a comprehensive and rigorous framework for evaluating and mitigating bias in AI systems. However, its effectiveness hinges on its ability to be adapted to the specific context in which it is applied. This adaptability is not a license to cut corners, but rather a necessary component of a meaningful and effective fairness audit. The core principles of the playbook's adaptability philosophy are:

*   **What must remain constant (non-negotiable methodological requirements):**
    *   **The Four-Component Structure:** Every audit must address all four components of the playbook: Historical Context, Fairness Definitions, Bias Sources, and Metrics. These components are interconnected and essential for a holistic understanding of fairness.
    *   **Intersectional Analysis:** Fairness cannot be assessed along a single axis. Every audit must consider the intersection of multiple protected attributes to uncover unique and compounded forms of bias.
    *   **Stakeholder Engagement:** A fairness audit is not a purely technical exercise. It must involve engagement with a diverse range of stakeholders, including those from affected communities, to understand the real-world impacts of the AI system.
    *   **Documentation:** All adaptation decisions, and the rationale behind them, must be thoroughly documented to ensure transparency and accountability.

*   **What should adapt based on context (flexible implementation details):**
    *   **The specific historical patterns investigated:** The historical context analysis should focus on the patterns of discrimination that are most relevant to the specific domain and application.
    *   **The fairness definitions prioritized:** The choice of which fairness definitions to prioritize will depend on the specific ethical, legal, and practical considerations of the application context.
    - **The bias sources investigated:** The playbook provides a comprehensive list of potential bias sources, but the audit should prioritize the investigation of those most likely to be present in the specific domain and problem type.
    *   **The metrics used for evaluation:** The specific metrics used to evaluate fairness should be tailored to the problem type and the prioritized fairness definitions.
    *   **The depth and breadth of the audit:** The level of detail and rigor of the audit will depend on the risk level of the application and the resources available.

*   **How to balance standardization with customization:**
    *   The playbook provides a standardized framework, but it is not a one-size-fits-all solution. The goal is to provide a common language and a shared set of principles for fairness auditing, while allowing for customization to the specific context.
    *   The balance between standardization and customization should be a conscious and deliberate decision, guided by the principles of methodological rigor, stakeholder engagement, and transparency.

### **Adaptation Decision Framework:**

*   **When to adapt vs. when to follow standard playbook:**
    *   Follow the standard playbook when the application is in a well-understood domain with clearly established fairness best practices.
    *   Adapt the playbook when the application is in a novel domain, when there are conflicting stakeholder priorities, or when the standard playbook does not adequately address the specific fairness challenges of the application.

*   **Who should make adaptation decisions (team, experts, stakeholders):**
    *   Adaptation decisions should be made by a cross-functional team with expertise in AI, fairness, the specific domain, and the legal and ethical implications of the application.
    *   The team should consult with a diverse range of stakeholders, including those from affected communities, to ensure that the adaptation decisions are informed by a broad range of perspectives.

*   **How to document adaptation rationales:**
    *   All adaptation decisions should be documented in a clear and transparent manner.
    *   The documentation should include the rationale for the adaptation, the potential risks and benefits, and the process by which the decision was made.

## **2. Domain-Specific Adaptations**

### **Healthcare Domain**

**Domain Characteristics:**

*   **Regulatory environment:** HIPAA, FDA, clinical trial requirements
*   **Key stakeholders:** Patients, providers, insurers, regulators
*   **Common AI applications:** Diagnosis, treatment recommendations, resource allocation
*   **Unique fairness considerations:** Life-and-death decisions, medical ethics

**Historical Context Adaptations:**

*   **Healthcare-specific historical discrimination patterns:** Investigate the history of medical experimentation on marginalized communities, the legacy of segregation in healthcare, and the historical underrepresentation of women and minorities in clinical trials.
*   **Medical data codification practices and their biases:** Examine how medical conditions have been defined and classified over time, and how these classifications may reflect historical biases.
*   **Relevant historical harms to investigate:** Focus on historical harms such as the Tuskegee Syphilis Study, the forced sterilization of women of color, and the disparate impact of a wide range of medical innovations.

**Fairness Definition Adaptations:**

*   **Clinical outcome fairness vs. access fairness:** Consider the trade-offs between ensuring that the AI system leads to equitable clinical outcomes and ensuring that it provides equitable access to care.
*   **Balancing individual patient benefit with population health:** Address the tension between optimizing for the best outcome for a single patient and optimizing for the best outcome for the population as a whole.
*   **Regulatory constraints on fairness definitions:** Ensure that the chosen fairness definitions are consistent with relevant healthcare regulations.

**Bias Source Adaptations:**

*   **Healthcare-specific bias sources:** Investigate bias sources such as the use of race as a biological proxy in clinical algorithms, the underrepresentation of certain populations in medical device testing, and the impact of social determinants of health on medical data.
*   **Measurement bias in health data:** Consider how differences in symptom reporting and access to care can lead to measurement bias in health data.
*   **Representation bias in clinical datasets:** Address the historical underrepresentation of women and minorities in clinical datasets and its impact on the performance of AI systems.

**Metrics Adaptations:**

*   **Clinical outcome metrics with fairness disaggregation:** Disaggregate clinical outcome metrics by race, ethnicity, gender, and other protected attributes to identify and mitigate fairness disparities.
*   **Safety-critical metrics specific to healthcare:** Use safety-critical metrics to evaluate the performance of AI systems in high-stakes healthcare applications.
*   **Regulatory compliance metrics:** Ensure that the AI system complies with all relevant regulatory requirements, such as those related to disparate impact in medical devices.

**Practical Considerations:**

*   **IRB approval processes for fairness audits:** Navigate the Institutional Review Board (IRB) approval process for fairness audits that involve human subjects research.
*   **Patient privacy constraints on fairness analysis:** Protect patient privacy while conducting fairness analysis, in compliance with HIPAA and other relevant regulations.
*   **Integration with clinical validation processes:** Integrate the fairness audit into the clinical validation process to ensure that the AI system is both safe and fair.

### **Financial Services Domain**

**Domain Characteristics:**

*   **Regulatory environment:** Equal Credit Opportunity Act, Fair Lending laws
*   **Key stakeholders:** Consumers, regulators, financial institutions
*   **Common AI applications:** Credit scoring, fraud detection, insurance underwriting
*   **Unique fairness considerations:** Legal definition of disparate impact

**Historical Context Adaptations:**

*   **Financial discrimination patterns:** Investigate the history of redlining, predatory lending, and other forms of financial discrimination.
*   **Credit reporting history and bias codification:** Examine how credit reporting agencies have historically collected and used data, and how this data may reflect historical biases.
*   **Relevant financial regulation history:** Understand the history of financial regulation and its impact on fairness in the financial services industry.

**Fairness Definition Adaptations:**

*   **Legal vs. technical fairness definitions:** Bridge the gap between legal and technical definitions of fairness, and ensure that the chosen fairness definitions are consistent with relevant legal requirements.
*   **Proxy discrimination concerns:** Address the risk of proxy discrimination, where seemingly neutral variables are used as proxies for protected attributes.
*   **Business necessity defenses vs. fairness requirements:** Navigate the tension between the business necessity defense and the requirement to use the least discriminatory alternative.

**Bias Source Adaptations:**

*   **Alternative data sources and their biases:** Evaluate the potential for bias in alternative data sources, such as social media data and utility payment history.
*   **Historical lending patterns embedded in training data:** Address the risk that historical lending patterns embedded in training data will perpetuate past discrimination.
*   **Measurement bias in financial data:** Consider how differences in income reporting and access to financial services can lead to measurement bias in financial data.

**Metrics Adaptations:**

*   **Disparate impact ratio (80% rule) compliance:** Ensure that the AI system complies with the 80% rule for disparate impact, as defined by the EEOC.
*   **Adverse action reason codes and fairness:** Evaluate the fairness of the adverse action reason codes that are provided to consumers who are denied credit.
*   **Profit/fairness trade-offs quantification:** Quantify the trade-offs between profit and fairness, and make explicit decisions about how to balance these competing objectives.

**Practical Considerations:**

*   **Fair lending testing requirements:** Comply with fair lending testing requirements, as mandated by regulators.
*   **Documentation for regulatory examinations:** Prepare comprehensive documentation for regulatory examinations, including the results of the fairness audit.
*   **Third-party model risk management integration:** Integrate the fairness audit into the third-party model risk management process.

### **Criminal Justice Domain**

**Domain Characteristics:**

*   **Constitutional constraints:** Due process, equal protection
*   **Key stakeholders:** Defendants, victims, courts, public defenders, prosecutors
*   **Common AI applications:** Risk assessment, recidivism prediction, resource allocation
*   **Unique fairness considerations:** Liberty deprivation, presumption of innocence

**Historical Context Adaptations:**

*   **Criminal justice discrimination history:** Investigate the history of over-policing, sentencing disparities, and other forms of discrimination in the criminal justice system.
*   **Data codification of criminogenic factors:** Examine how "criminogenic" factors have been defined and measured over time, and how these factors may reflect historical biases.
*   **Relevant case law and policy history:** Understand the history of case law and policy related to fairness in the criminal justice system.

**Fairness Definition Adaptations:**

*   **Calibration vs. balance in recidivism prediction:** Consider the trade-offs between calibration and balance in recidivism prediction, and make explicit decisions about which to prioritize.
*   **Pre-trial vs. sentencing vs. parole fairness:** Adapt the fairness definitions to the specific context of the criminal justice decision being made.
*   **Individual rights vs. public safety trade-offs:** Address the tension between individual rights and public safety, and make explicit decisions about how to balance these competing objectives.

**Bias Source Adaptations:**

*   **Feedback loops in arrest data:** Address the risk of feedback loops in arrest data, where police are dispatched to areas with high arrest rates, leading to even more arrests.
*   **Measurement bias in criminal history records:** Consider how measurement bias in criminal history records can lead to biased predictions.
*   **Geographic bias in law enforcement data:** Address the risk of geographic bias in law enforcement data, where certain neighborhoods are over-policed.

**Metrics Adaptations:**

*   **False positive vs. false negative trade-offs:** Consider the trade-offs between false positives and false negatives, and make explicit decisions about which to prioritize.
*   **Subgroup calibration requirements:** Ensure that the AI system is calibrated across all relevant subgroups.
*   **Transparency metrics for legal challenges:** Use transparency metrics to prepare for legal challenges to the AI system.

**Practical Considerations:**

*   **Discovery requirements in criminal cases:** Comply with discovery requirements in criminal cases, which may require the disclosure of information about the AI system.
*   **Expert witness testimony on fairness:** Prepare for expert witness testimony on the fairness of the AI system.
*   **Integration with judicial decision-making:** Integrate the fairness audit into the judicial decision-making process.

### **Employment/Hiring Domain**

**Domain Characteristics:**

*   **Regulatory environment:** EEOC guidelines, Title VII
*   **Key stakeholders:** Job seekers, hiring managers, HR, legal compliance
*   **Common AI applications:** Resume screening, interview scoring, performance prediction
*   **Unique fairness considerations:** Workplace diversity goals

**Historical Context Adaptations:**

*   **Employment discrimination history:** Investigate the history of employment discrimination, including the legacy of "separate but equal" and the history of occupational segregation.
*   **Occupational segregation patterns:** Examine how occupational segregation has led to the underrepresentation of women and minorities in certain fields.
*   **Relevant EEOC case law:** Understand the history of EEOC case law and its impact on fairness in employment.

**Fairness Definition Adaptations:**

*   **Four-fifths rule compliance:** Ensure that the AI system complies with the four-fifths rule for disparate impact, as defined by the EEOC.
*   **Qualified applicant pool definition:** Define the qualified applicant pool in a fair and objective manner.
*   **Business necessity vs. fairness constraints:** Navigate the tension between the business necessity defense and the requirement to use the least discriminatory alternative.

**Bias Source Adaptations:**

*   **Biased job descriptions and requirements:** Address the risk of biased job descriptions and requirements, which can deter qualified candidates from applying.
*   **Historical hiring patterns in training data:** Address the risk that historical hiring patterns embedded in training data will perpetuate past discrimination.
*   **Measurement bias in qualifications assessment:** Consider how measurement bias in qualifications assessment can lead to biased predictions.

**Metrics Adaptations:**

*   **Adverse impact analysis:** Conduct a thorough adverse impact analysis to identify and mitigate fairness disparities.
*   **Qualified candidate flow metrics:** Track the flow of qualified candidates through the hiring process to identify and address any bottlenecks.
*   **Diversity hiring effectiveness:** Evaluate the effectiveness of the AI system in achieving the organization's diversity and inclusion goals.

**Practical Considerations:**

*   **EEOC recordkeeping requirements:** Comply with EEOC recordkeeping requirements, which may require the disclosure of information about the AI system.
*   **Validation study integration:** Integrate the fairness audit into the validation study for the AI system.
*   **Reasonable accommodation considerations:** Ensure that the AI system can accommodate candidates with disabilities.

### **Education Domain**

**Domain Characteristics:**

*   **Regulatory environment:** Title IX, FERPA, disability accommodations
*   **Key stakeholders:** Students, parents, educators, administrators
*   **Common AI applications:** Admissions, grade prediction, resource allocation
*   **Unique fairness considerations:** Opportunity equity, educational outcomes

**Historical Context Adaptations:**

*   **Educational segregation and tracking history:** Investigate the history of educational segregation and tracking, and their impact on educational outcomes.
*   **Standardized testing bias history:** Examine the history of standardized testing and its disparate impact on marginalized communities.
*   **Access disparities in educational resources:** Address the historical disparities in access to educational resources, and their impact on educational outcomes.

**Fairness Definition Adaptations:**

*   **Predictive accuracy vs. opportunity provision:** Consider the trade-offs between predictive accuracy and opportunity provision, and make explicit decisions about which to prioritize.
*   **Standardized test score interpretation:** Interpret standardized test scores in a fair and objective manner, taking into account the historical biases of these tests.
*   **Disability accommodation in AI systems:** Ensure that the AI system can accommodate students with disabilities.

**Bias Source Adaptations:**

*   **Prior achievement data reflecting unequal educational access:** Address the risk that prior achievement data, which reflects unequal access to educational resources, will lead to biased predictions.
*   **Measurement bias in assessment tools:** Consider how measurement bias in assessment tools can lead to biased predictions.
*   **Representation bias in gifted/talented identification:** Address the underrepresentation of marginalized students in gifted and talented programs.

**Metrics Adaptations:**

*   **Educational outcome equity metrics:** Use educational outcome equity metrics to evaluate the fairness of the AI system.
*   **Opportunity gap measurement:** Measure the opportunity gap between different groups of students, and use the AI system to close this gap.
*   **Longitudinal fairness tracking:** Track the fairness of the AI system over time to ensure that it is not perpetuating historical inequities.

**Practical Considerations:**

*   **Student privacy protection (FERPA):** Protect student privacy while conducting fairness analysis, in compliance with FERPA and other relevant regulations.
*   **Disability accommodation validation:** Validate that the AI system can accommodate students with disabilities.
*   **Integration with individualized education plans:** Integrate the fairness audit into the individualized education plan (IEP) process.

## **3. Problem Type-Specific Adaptations**

### **Classification Problems**

**Applicability:**

*   Binary classification (approve/reject, high-risk/low-risk)
*   Multi-class classification (risk tiers, diagnosis categories)

**Fairness Definition Selection:**

*   **Which fairness definitions apply to classification:** Demographic parity, equal opportunity, equalized odds, and predictive parity are all applicable to classification problems.
*   **Trade-offs between TPR parity, FPR parity, predictive parity:** Consider the trade-offs between these different fairness definitions, and make explicit decisions about which to prioritize.
*   **Decision threshold optimization for fairness:** Optimize the decision threshold of the classifier to achieve the desired balance of fairness and accuracy.

**Bias Source Considerations:**

*   **Label bias in classification training data:** Address the risk of label bias in classification training data, where the labels themselves are a source of bias.
*   **Feature representation for classification:** Ensure that the feature representation for the classification problem is fair and objective.

**Metrics Adaptations:**

*   **Confusion matrix disaggregation by group:** Disaggregate the confusion matrix by group to identify and mitigate fairness disparities.
*   **ROC curves and fairness visualization:** Use ROC curves and other fairness visualization techniques to evaluate the fairness of the classifier.
*   **Classification fairness metrics checklist:** Use a checklist of classification fairness metrics to ensure that all relevant metrics are being considered.

**Examples:**

*   Credit approval, medical diagnosis, recidivism prediction

### **Regression Problems**

**Applicability:**

*   Continuous outcome prediction (salary, loan amount, treatment dosage)

**Fairness Definition Selection:**

*   **Adapting demographic parity to continuous outcomes:** Adapt the demographic parity definition to continuous outcomes by requiring that the mean prediction is the same across groups.
*   **Mean prediction parity vs. full distribution parity:** Consider the trade-offs between mean prediction parity and full distribution parity, and make explicit decisions about which to prioritize.
*   **Error parity considerations:** Ensure that the errors of the regression model are similar across groups.

**Bias Source Considerations:**

*   **Historical outcome bias in regression targets:** Address the risk of historical outcome bias in regression targets, where the target variable itself is a source of bias.
*   **Non-linear relationships masking discrimination:** Consider the risk that non-linear relationships in the data may be masking discrimination.

**Metrics Adaptations:**

*   **Residual analysis by protected group:** Analyze the residuals of the regression model by protected group to identify and mitigate fairness disparities.
*   **Prediction interval fairness:** Ensure that the prediction intervals of the regression model are similar across groups.
*   **Regression fairness metrics checklist:** Use a checklist of regression fairness metrics to ensure that all relevant metrics are being considered.

**Examples:**

*   Salary prediction, insurance pricing, resource allocation

### **Ranking Problems**

**Applicability:**

*   Ordered lists (search results, recommendation systems, college admissions)

**Fairness Definition Selection:**

*   **Position-based fairness (exposure in top-k):** Ensure that different groups are represented fairly in the top-k results of the ranking.
*   **Meritocratic ranking vs. fair representation:** Consider the trade-offs between meritocratic ranking and fair representation, and make explicit decisions about which to prioritize.
*   **Individual vs. group fairness in rankings:** Address a potential tension between individual and group fairness in rankings.

**Bias Source Considerations:**

*   **Query bias in ranking systems:** Address the risk of query bias in ranking systems, where the queries themselves are a source of bias.
*   **Historical engagement patterns embedding bias:** Address the risk that historical engagement patterns embedded in the training data will perpetuate past discrimination.
*   **Position bias in training data:** Address the risk of position bias in the training data, where items that are ranked higher are more likely to be clicked on, regardless of their relevance.

**Metrics Adaptations:**

*   **Exposure fairness metrics:** Use exposure fairness metrics to evaluate the fairness of the ranking.
*   **Normalized discounted cumulative gain (NDCG) by group:** Disaggregate the NDCG by group to identify and mitigate fairness disparities.
*   **Ranking fairness metrics checklist:** Use a checklist of ranking fairness metrics to ensure that all relevant metrics are being considered.

**Examples:**

*   Job candidate ranking, college admissions, search results

### **Recommendation Systems**

**Applicability:**

*   Personalized recommendations (content, products, connections)

**Fairness Definition Selection:**

*   **Provider fairness vs. consumer fairness:** Consider the trade-offs between provider fairness and consumer fairness, and make explicit decisions about which to prioritize.
*   **Filter bubble mitigation:** Mitigate the risk of filter bubbles, where users are only exposed to content that confirms their existing beliefs.
*   **Diversity vs. relevance trade-offs:** Address the tension between diversity and relevance in recommendations.

**Bias Source Considerations:**

*   **Historical interaction patterns embedding discrimination:** Address the risk that historical interaction patterns embedded in the training data will perpetuate past discrimination.
*   **Cold start problems for underrepresented groups:** Address the cold start problem for underrepresented groups, who may not have enough data to receive accurate recommendations.
*   **Feedback loops amplifying bias:** Address the risk of feedback loops amplifying bias in recommendation systems.

**Metrics Adaptations:**

*   **Recommendation diversity metrics:** Use recommendation diversity metrics to evaluate the fairness of the recommendation system.
*   **Fairness in multi-stakeholder recommendation:** Consider the fairness of the recommendation system for all stakeholders, including providers and consumers.
*   **Long-term fairness effects:** Evaluate the long-term fairness effects of the recommendation system.

**Examples:**

*   Job recommendations, content recommendations, social network suggestions

### **Clustering/Unsupervised Learning**

**Applicability:**

*   Customer segmentation, anomaly detection, representation learning

**Fairness Definition Selection:**

*   **Fairness without explicit protected attributes:** Define fairness in the absence of explicit protected attributes.
*   **Similarity metric fairness:** Ensure that the similarity metric used for clustering is fair and objective.
*   **Unsupervised bias detection:** Use unsupervised bias detection techniques to identify and mitigate bias in clustering.

**Bias Source Considerations:**

*   **Implicit protected attribute influence:** Address the risk that implicit protected attributes will influence the clustering.
*   **Geometric bias in embedding spaces:** Address the risk of geometric bias in the embedding spaces used for clustering.
*   **Cluster interpretability and fairness:** Ensure that the clusters are interpretable and that the interpretation is fair.

**Metrics Adaptations:**

*   **Cluster balance metrics:** Use cluster balance metrics to evaluate the fairness of the clustering.
*   **Embedding fairness metrics:** Use embedding fairness metrics to evaluate the fairness of the embedding spaces used for clustering.
*   **Unsupervised fairness assessment:** Use unsupervised fairness assessment techniques to evaluate the fairness of the clustering.

**Examples:**

*   Market segmentation, fraud detection, representation learning

## **4. Organizational Context Adaptations**

**Organization Size:**

*   **Startups vs. enterprises (resource constraints, speed vs. rigor):**
    *   Startups may have limited resources and may need to prioritize the most critical fairness risks.
    *   Enterprises may have more resources and may be able to conduct a more comprehensive fairness audit.

*   **Team expertise levels (when to bring in external experts):**
    *   Teams with limited expertise in fairness may need to bring in external experts to assist with the audit.
    *   Teams with more expertise may be able to conduct the audit in-house.

**Industry Maturity:**

*   **Early-stage AI adoption (building fairness foundations):**
    *   Organizations that are new to AI may need to focus on building a strong foundation for fairness, including establishing clear policies and procedures.
*   **Mature AI organizations (advanced fairness practices):**
    *   Organizations that are more mature in their use of AI may be able to implement more advanced fairness practices, such as fairness-aware machine learning.

**Regulatory Context:**

*   **Highly regulated industries (compliance-driven adaptations):**
    *   Organizations in highly regulated industries will need to adapt the playbook to ensure compliance with all relevant regulations.
*   **Unregulated industries (ethics-driven adaptations):**
    *   Organizations in unregulated industries may have more flexibility in how they adapt the playbook, and may choose to focus on ethics-driven adaptations.

**Risk Tolerance:**

*   **High-stakes decisions (conservative fairness approaches):**
    *   For high-stakes decisions, organizations should adopt a conservative approach to fairness, and may want to err on the side of caution.
*   **Low-stakes decisions (pragmatic trade-offs):**
    *   For low-stakes decisions, organizations may be able to make more pragmatic trade-offs between fairness and other objectives.

## **5. Adaptation Decision Trees**

### **Domain Selection Tree:**

*   "What domain am I working in?"
    *   Healthcare -> Go to Healthcare Domain Adaptations
    *   Finance -> Go to Financial Services Domain Adaptations
    *   Criminal Justice -> Go to Criminal Justice Domain Adaptations
    *   Hiring -> Go to Employment/Hiring Domain Adaptations
    *   Education -> Go to Education Domain Adaptations
    *   Other -> Consult with domain experts to adapt the playbook

### **Problem Type Selection Tree:**

*   "What ML problem type?"
    *   Classification -> Go to Classification Problem Type Adaptations
    *   Regression -> Go to Regression Problem Type Adaptations
    *   Ranking -> Go to Ranking Problem Type Adaptations
    *   Recommendation -> Go to Recommendation Systems Problem Type Adaptations
    *   Clustering -> Go to Clustering/Unsupervised Learning Problem Type Adaptations

### **Organizational Context Tree:**

*   Assess: Size, Expertise, Regulatory Constraints, Stakeholder Requirements
    *   Based on the assessment, consult the Organizational Context Adaptations section for guidance.

## **6. Adaptation Templates & Checklists**

### **Domain Adaptation Template:**

```
# Domain: [Name]

## Context
- Regulatory environment:
- Key stakeholders:
- Common applications:
- Unique fairness considerations:

## Historical Context Adaptations
- Historical patterns to investigate:
- Codification practices to examine:
- Relevant historical sources:

## Fairness Definition Adaptations
- Domain-specific fairness definitions:
- Stakeholder fairness priorities:
- Legal/regulatory constraints:

## Bias Source Adaptations
- Domain-specific bias sources:
- Measurement considerations:
- Data representation issues:

## Metrics Adaptations
- Domain-specific metrics:
- Regulatory compliance metrics:
- Interpretation guidance:

## Practical Considerations
- Timeline implications:
- Expertise requirements:
- Integration challenges:
```

### **Problem Type Adaptation Checklist:**

*   [ ] Identified applicable fairness definitions for this problem type
*   [ ] Adapted metrics to problem type characteristics
*   [ ] Considered problem-specific bias sources
*   [ ] Addressed problem-specific data challenges
*   [ ] Selected appropriate validation methods

## **7. Common Adaptation Pitfalls**

### **Over-Adaptation Risks:**

*   **Losing methodological rigor in pursuit of flexibility:** This can lead to a superficial or incomplete audit that fails to identify and mitigate significant fairness risks.
*   **Domain-specific justifications for cutting corners:** Teams may be tempted to use the unique characteristics of their domain as an excuse to cut corners on the fairness audit.
*   **"Our domain is unique" as excuse for incomplete audits:** While every domain has its unique challenges, the core principles of fairness are universal. Teams should not use the uniqueness of their domain as an excuse to conduct an incomplete audit.

### **Under-Adaptation Risks:**

*   **Applying generic approaches without domain context:** This can lead to an audit that is irrelevant or ineffective, as it may fail to address the specific fairness challenges of the domain.
*   **Missing critical domain-specific bias sources:** Different domains have different bias sources. A generic audit may miss critical domain-specific bias sources, leading to a false sense of security.
*   **Using inappropriate fairness definitions for context:** The choice of fairness definitions should be guided by the specific context of the application. Using inappropriate fairness definitions can lead to a misleading or counterproductive audit.

### **Guidance:**

*   **When to seek external domain expertise:** If the team lacks expertise in the specific domain, it is important to seek external expertise to ensure that the audit is relevant and effective.
*   **Red flags indicating inappropriate adaptations:** Red flags include cutting corners on methodological rigor, ignoring the perspectives of marginalized stakeholders, and making adaptation decisions without a clear and transparent rationale.
*   **Validation approaches for adaptation decisions:** Adaptation decisions should be validated through a process of stakeholder engagement, peer review, and empirical testing.

Last Updated: 2025-11-08T16:46:29.532-05:00

