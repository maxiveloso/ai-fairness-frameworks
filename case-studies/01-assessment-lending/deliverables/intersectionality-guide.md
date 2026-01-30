
## 1. Intersectionality Foundation

### Conceptual Framework

#### Definition
Intersectionality in the context of AI fairness is the analytical framework that recognizes that individuals' identities are constituted by multiple, intersecting social categories such as race, gender, age, and disability. It posits that the forms of discrimination and disadvantage experienced by individuals at these intersections are often unique and cannot be understood by simply adding up the biases associated with each individual identity. For instance, the discrimination faced by a Black woman is not merely the sum of the discrimination faced by Black men and white women; it is a distinct form of discrimination that arises from the unique intersection of her race and gender.

#### Core principles
*   **Multiplicative, not Additive:** Intersectional analysis considers the multiplicative effects of intersecting identities, where the whole is different from the sum of its parts.
*   **Context-Specific:** The salience and effects of different intersections depend on the specific social and historical context of the AI system's deployment.
*   **Power Dynamics:** Intersectionality is fundamentally about power and the way in which intersecting identities can lead to unique forms of marginalization and privilege.
*   **Beyond Demographics:** While often focused on protected attributes, intersectional analysis can also include other factors like socioeconomic status, geographic location, and language.

#### Why it matters
Ignoring intersectionality can lead to significant harms, including:
*   **Invisibility:** The unique experiences of intersectional groups can be rendered invisible, leading to their needs and concerns being overlooked in the design and evaluation of AI systems.
*   **Compound Discrimination:** AI systems can perpetuate and even amplify existing societal biases, resulting in compounded discrimination against individuals with multiple marginalized identities.
*   **Ineffective Interventions:** Fairness interventions that do not account for intersectionality may fail to address the specific harms faced by intersectional groups, and in some cases, may even exacerbate them.

#### Historical grounding
The concept of intersectionality was first coined by legal scholar Kimberlé Crenshaw in 1989. She used it to describe how the legal system's failure to account for the intersection of race and gender discrimination left Black women without legal recourse. The concept has since been widely adopted in a variety of fields, including feminist and critical race studies, to analyze how systems of power and oppression intersect to create unique experiences of discrimination.

### Common Misconceptions
*   **Intersectionality is NOT an additive model:** It is a common misconception that intersectional analysis is simply a matter of adding up the biases associated with each individual identity. In fact, intersectional analysis is about understanding the unique, non-additive effects that arise from the intersection of multiple identities.
*   **Intersectionality is NOT just subgroup analysis:** While subgroup analysis is a necessary component of intersectional analysis, it is not sufficient. A true intersectional analysis must also consider the power dynamics and historical context that shape the experiences of different intersectional groups.
*   **Intersectionality is NOT the same as demographic segmentation:** Demographic segmentation is a marketing technique that involves dividing a population into different groups based on their demographic characteristics. Intersectional analysis, on the other hand, is a critical framework for understanding how systems of power and oppression intersect to create unique experiences of discrimination.

## 2. Intersectionality in Each Component

### Component 1: Historical Context Assessment with Intersectional Lens

#### Intersectional Historical Analysis
*   **Researching historical discrimination patterns:** This involves going beyond single-axis historical narratives to research how historical discrimination has specifically impacted intersectional groups. For example, instead of just researching the history of housing discrimination against Black people, an intersectional analysis would also look at how this discrimination has been compounded for Black women, Black disabled people, or Black queer people.
*   **Examples of intersectional historical harms:**
    *   The forced sterilization of Black and Indigenous women in the 20th century.
    *   Employment discrimination against disabled immigrants.
    *   The exclusion of women of color from the mainstream feminist movement.
*   **Documentation requirements:** The historical context assessment should include a specific section on intersectional historical patterns, which documents the unique forms of discrimination faced by relevant intersectional groups.

#### Data Codification Practices
*   **Analyzing historical codification practices:** This involves examining how historical data collection and codification practices have created or erased intersectional categories. For example, the historical tendency to collect data on race and gender as separate, independent categories has made it difficult to study the experiences of intersectional groups.
*   "**Analyzing proxy variables:** This involves analyzing how historical data may contain proxy variables that capture intersectional effects. For example, a dataset that does not contain information on race may still contain information on neighborhood, which can serve as a proxy for race.
*   **Understanding visibility vs. invisibility:** This involves understanding how historical data collection practices have made some intersectional groups more visible than others. For example, the historical focus on the experiences of white women in the feminist movement has made their experiences more visible than those of women of color.

#### Practical Guidance
*   **Specific historical sources to consult:**
    *   Academic databases such as JSTOR, Project MUSE, and Google Scholar.
    *   Primary sources such as government documents, court records, and newspapers.
    *   Oral histories and community archives.
*   **Questions to ask:**
    *   "Which intersectional groups have been most affected by historical discrimination in this domain?"
    *   "What are the specific historical harms that these groups have experienced?"
    *   "How have historical data collection and codification practices created or erased intersectional categories?"
*   **Template for documenting intersectional historical findings:** The documentation should include a description of the historical context, the specific intersectional groups affected, the harms they have experienced, and the data sources used to support the analysis.

### Component 2: Fairness Definition Selection with Intersectionality

#### Intersectional Fairness Definitions
*   **How standard fairness definitions perform with intersectional groups:** Standard fairness definitions, such as demographic parity and equalized odds, can fail to detect discrimination against intersectional groups. For example, a model may satisfy demographic parity for both men and women and for both Black and white people, but still, discriminate against Black women.
*   **Limitations of single-axis fairness definitions:** Single-axis fairness definitions are limited in their ability to capture the unique, non-additive effects that arise from the intersection of multiple identities.
*   **When to use subgroup-specific fairness constraints:** Subgroup-specific fairness constraints should be used when there is reason to believe that a model may be discriminating against a specific intersectional group.

#### Stakeholder Engagement
*   **Ensuring intersectional community representation:** It is crucial to ensure that intersectional community members are represented in the stakeholder engagement process. This can be done by partnering with community organizations that work with intersectional groups.
*   **Eliciting intersectional fairness priorities:** It is important to elicit the fairness priorities of intersectional community members. This can be done through interviews, focus groups, and surveys.
*   **Addressing conflicts between single-axis and intersectional fairness goals:** It is important to have a process in place for addressing conflicts between single-axis and intersectional fairness goals. This may involve prioritizing the needs of the most marginalized groups.

#### Mathematical Considerations
*   **Sample size challenges:** Intersectional subgroups often have small sample sizes, which can make it difficult to reliably test for fairness.
*   **Statistical power:** Small sample sizes can also lead to low statistical power, which makes it difficult to detect discrimination even when it is present.
*   **Aggregation decisions:** When analyzing intersectional subgroups, it is important to make thoughtful decisions about when to aggregate data and when to analyze subgroups separately.

#### Practical Guidance
*   **Decision tree:** A decision tree can be used to help select the most appropriate fairness definitions for a given context. The decision tree should take into account the specific intersectional groups of concern, the fairness priorities of stakeholders, and the mathematical limitations of the data.
*   **Examples:** The Fairness Definition Selection Tool should include examples of how different fairness definitions can be applied to different intersectional contexts.
*   "**Documentation template:** The documentation for the Fairness Definition Selection Tool should include a template for documenting the choices made about which fairness definitions to use.

### Component 3: Bias Source Identification with Intersectional Analysis

#### Intersectional Bias Sources
*   **Compound bias:** Compound bias occurs when multiple sources of bias interact to create a unique, non-additive effect. For example, a model may be biased against both women and people of color, but the bias against women of color may be greater than the sum of the biases against women and people of color.
*   **Data representation bias:** Data representation bias can occur when intersectional groups are underrepresented or misrepresented in the training data. This can lead to a model that is less accurate for these groups.
*   **Historical bias:** Historical bias can be encoded in the training data, leading to a model that perpetuates and even amplifies existing societal biases.
*   **Measurement bias:** Measurement bias can occur when the tools and proxies used to measure a particular construct are more accurate for some groups than for others.

#### Interaction Effects
*   **Identifying non-additive bias patterns:** It is important to identify non-additive bias patterns, as these can be a sign of compound bias.
*   **Detecting when bias sources amplify each other:** It is also important to detect when bias sources amplify each other for specific intersections. This can be done by using statistical tests for interaction effects.
*   **Methods for testing interaction effects:** There are a number of statistical methods that can be used to test for interaction effects, such as logistic regression and ANOVA.

#### Technical Methods
*   **Statistical tests for intersectional bias:** There are a number of statistical tests that can be used to test for intersectional bias, such as the three-part analysis for disparate impact.
*   **Qualitative methods:** Qualitative methods, such as interviews and focus groups, can be used to supplement statistical methods and gain a deeper understanding of the experiences of intersectional community members.
*   **Case studies:** Case studies can be used to illustrate how intersectional bias can manifest in real-world AI systems.

#### Practical Guidance
*   **Intersectional bias source checklist:** An intersectional bias source checklist can be used to help identify potential sources of intersectional bias.
*   **Example bias source analyses:** The Bias Source Identification Tool should include example bias source analyses for different intersectional groups.
*   **Common pitfalls:** The Bias Source Identification Tool should also include a list of common pitfalls to avoid when identifying intersectional-specific bias sources.

### Component 4: Comprehensive Metrics with Intersectional Disaggregation

#### Intersectional Metrics Framework
*   **Disaggregation strategy:** The disaggregation strategy should be guided by the specific intersectional groups of concern. In general, it is best to disaggregate the data as much as possible, while still maintaining a sufficient sample size for reliable analysis.
*   **Handling statistical limitations:** There are a number of statistical techniques that can be used to handle the statistical limitations of small sample sizes, such as bootstrapping and permutation testing.
*   **Visualization:** There are a number of visualization techniques that can be used to present intersectional metrics effectively, such as heatmaps and treemaps.

#### Metric Selection
*   **Which metrics are most appropriate for intersectional analysis?**: There are a number of metrics that are particularly well-suited for intersectional analysis, such as the between-group generalization error and the intersectional disparate impact ratio.
*   **How to adapt single-axis metrics for intersectional contexts:** Single-axis metrics can be adapted for intersectional contexts by disaggregating the data by intersectional group.
*   **New metrics specifically designed for intersectional fairness:** There are a number of new metrics that have been specifically designed for intersectional fairness, such as the conditional demographic parity and the intersectional accuracy difference.

#### Interpretation Guidance
*   **Reading intersectional metrics:** When reading intersectional metrics, it is important to pay attention to the patterns of disparity across different intersectional groups.
*   **Comparing intersectional vs. single-axis results:** It is also important to compare the results of the intersectional analysis to the results of the single-axis analysis. This can help to identify cases where a model appears to be fair on the surface but is actually discriminating against a specific intersectional group.
*   **Prioritization:** When intersectional findings conflict with single-axis findings, it is important to have a process in place for prioritizing the needs of the most marginalized groups.

#### Practical Guidance
*   **Example intersectional metrics implementations:** The Comprehensive Metrics Tool should include example implementations of intersectional metrics.
*   **Code snippets for intersectional disaggregation:** The Comprehensive Metrics Tool should also include code snippets for intersectional disaggregation.
*   **Reporting template for intersectional metrics results:** The documentation for the Comprehensive Metrics Tool should include a template for reporting the results of the intersectional metrics analysis.

## 3. Cross-Cutting Intersectionality Practices

### Data Requirements
*   **Collecting data on multiple protected attributes:** In order to conduct an intersectional analysis, it is necessary to collect data on multiple protected attributes, such as race, gender, age, and disability.
*   **Privacy-preserving approaches to intersectional data:** There are a number of privacy-preserving approaches that can be used to collect and analyze intersectional data, such as differential privacy and federated learning.
*   **Dealing with missing or incomplete intersectional data:** There are a number of techniques that can be used to deal with missing or incomplete intersectional data, such as imputation and data augmentation.

### Organizational Readiness
*   **Building internal capacity for intersectional analysis:** It is important to build internal capacity for intersectional analysis. This can be done through training and by hiring staff with expertise in intersectional fairness.
*   **Training requirements for team members:** All team members should receive training on intersectional fairness. This training should cover the conceptual foundations of intersectionality, as well as the practical methods for integrating intersectional analysis into the fairness audit process.
*   **Establishing partnerships with intersectional advocacy groups:** It is important to establish partnerships with intersectional advocacy groups. These partnerships can provide valuable insights into the experiences of intersectional community members and can help to ensure that the fairness audit is responsive to their needs.

### Stakeholder Engagement Across Components
*   **Ensuring intersectional representation throughout the audit process:** It is crucial to ensure that intersectional community members are represented throughout the fairness audit process. This includes the planning, execution, and reporting phases of the audit.
*   **Compensating intersectional community members for participation:** It is important to compensate intersectional community members for their participation in the fairness audit. This will help to ensure that their voices are heard and that their contributions are valued.
*   **Building trust with historically marginalized intersectional communities:** It is important to build trust with historically marginalized intersectional communities. This can be done by being transparent about the fairness audit process, by being responsive to their concerns, and by taking action to address the harms that they have experienced.

## 4. Implementation Challenges & Solutions

### Common Challenges
*   **Sample size limitations with fine-grained intersectional analysis:** The number of individuals in each intersectional subgroup can be small, which can make it difficult to draw statistically significant conclusions.
*   **Computational complexity of intersectional metrics:** Calculating intersectional metrics can be computationally expensive, especially when there are many protected attributes.
*   **Organizational resistance to intersectional analysis:** Some organizations may be resistant to intersectional analysis due to a lack of understanding or a fear of what the analysis might reveal.
*   **Legal/regulatory constraints on collecting intersectional data:** In some jurisdictions, there may be legal or regulatory constraints on the collection of intersectional data.

### Practical Solutions
*   **Workarounds for small sample sizes:** There are a number of workarounds for small sample sizes, such as using hierarchical models, Bayesian methods, and data augmentation.
*   **Prioritization frameworks:** When it is not feasible to analyze all intersectional subgroups, a prioritization framework can be used to focus on the most important subgroups.
*   **Communication strategies for explaining intersectionality to stakeholders:** It is important to have a clear and concise communication strategy for explaining intersectionality to stakeholders. This strategy should tailor the message to the specific audience and should use examples to illustrate the concepts.
*   **Legal strategies for collecting intersectional data within constraints:** There are a number of legal strategies that can be used to collect intersectional data within constraints, such as obtaining consent from the individuals whose data is being collected and using privacy-preserving techniques.

## 5. Real-World Examples

**1. Hiring Algorithm**
*   **System/Domain:** A hiring algorithm used to screen job applicants for a tech company.
*   **Intersectionality Integration:** The audit team disaggregated the data by race and gender and found that the algorithm was significantly more likely to reject Black women than any other group, even though it appeared to be fair when looking at race and gender separately.
*   **Findings:** The intersectional analysis revealed a hidden bias against Black women that would have gone unnoticed with a single-axis analysis.
*   **Actions Taken:** The company retrained the algorithm on a more balanced dataset and implemented a new policy to ensure that all qualified Black women were given a fair opportunity to interview for the job.

**2. Loan Application AI**
*   **System/Domain:** An AI system used to approve or deny loan applications.
*   **Intersectionality Integration:** A fairness audit disaggregated performance data by race, gender, and zip code.
*   **Findings:** The analysis found that the AI system had a significantly higher false rejection rate for women of color living in low-income zip codes, even though the model appeared to be fair when evaluated along each of these axes individually.
*   **Actions Taken:** The financial institution revised its model to remove the impact of zip code as a proxy for socioeconomic status and implemented a manual review process for all applications from high-risk intersectional subgroups.

**3. Medical Diagnostic System**
*   **System/Domain:** A medical diagnostic system used to identify skin cancer.
*   **Intersectionality Integration:** The audit team disaggregated the data by skin tone and age and found that the system was significantly less accurate for older patients with dark skin tones.
*   **Findings:** The intersectional analysis revealed a hidden bias against older, dark-skinned patients that would have gone unnoticed with a single-axis analysis.
*   **Actions Taken:** The company retrained the model on a more diverse dataset and implemented a new policy to ensure that all patients, regardless of their skin tone or age, received a high-quality diagnosis.

## 6. Quality Checklist

*   [ ] The historical context assessment includes a specific section on intersectional discrimination patterns.
*   [ ] The fairness definition selection process includes a discussion of how standard fairness definitions perform with intersectional groups and when to use subgroup-specific fairness constraints.
*   [ ] The bias source identification process includes an analysis of compound bias, data representation bias, historical bias, and measurement bias.
*   [ ] The comprehensive metrics section includes a disaggregation strategy, a discussion of how to handle statistical limitations, and a plan for visualizing intersectional metrics.
*   [ ] Stakeholder engagement includes representatives from intersectional communities, and their fairness priorities are elicited and addressed.
*   [ ] The documentation for the fairness audit explicitly addresses intersectional considerations.
*   [ ] The findings of the fairness audit are interpreted with an intersectional lens.
*   [ ] The recommendations of the fairness audit specifically address intersectional harms.


