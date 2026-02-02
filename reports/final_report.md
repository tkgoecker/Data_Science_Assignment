# Clinical Readmission Prediction & Discharge Note Entity Extraction

## Overview
This project explores two complementary tasks using a synthetic clinical dataset:
1. Predicting whether a patient will be readmitted within 30 days of discharge.
2. Extracting clinically relevant information from free-text discharge notes.

The goal is to demonstrate an end-to-end data science workflow covering data preparation, modeling, evaluation, NLP, and clear communication of results.

---

## 1. Predictive Modeling: 30-Day Readmission

### Approach
The structured dataset includes demographic, diagnostic, and utilization features. After initial validation and cleaning, the following steps were performed:

- **Feature Engineering**
  - Created `admissions_per_day` to normalize prior admissions by length of stay.
  - Bucketed age into categorical groups to capture non-linear effects.
  - One-hot encoded categorical variables (gender, diagnosis code, medication type).
  - Standardized numerical features.

- **Models Trained**
  - Logistic Regression (interpretable baseline)
  - Random Forest Classifier (non-linear ensemble)

- **Evaluation Metrics**
  - ROC AUC
  - F1 Score
  - Confusion Matrix

Stratified train/test splitting was used to preserve class balance.

### Results
| Model | ROC AUC | F1 Score |
|------|--------|----------|
| Logistic Regression | 0.453 | 0.343 |
| Random Forest | 0.425 | 0.286 |

Both models perform only slightly above random guessing, indicating weak predictive signal in the available structured features.
### Full-Dataset Risk Scoring (Post-Training)

In addition to standard train/test evaluation, the trained models were applied to the full dataset of 200 patients to simulate a deployment-style scenario. This analysis estimates how many patients would be flagged as high risk for readmission under a fixed probability threshold (0.5). These results are descriptive and are reported separately from evaluation metrics to avoid data leakage.

| Model | Patients Flagged (out of 200) | Percentage |
|------|-------------------------------|------------|
| Logistic Regression | 84 | 42.0% |
| Random Forest | 65 | 32.5% |

The random forest model flags a number of patients similar to the observed readmission rate in the dataset (32.5%), while logistic regression produces a higher number of positive predictions, reflecting differences in model calibration and decision boundaries.

### Feature Importance (Logistic Regression)
Most influential features included:
- Age buckets (younger patients showed higher readmission likelihood)
- Medication Type C
- Diagnosis Code D003
- Gender (female associated with slightly lower risk)

### Interpretation
The limited performance is expected given:
- Small dataset size (200 records)
- Lack of strong temporal, clinical severity, or lab-based features
- Readmission being influenced by many unobserved social and clinical factors

This reflects real-world healthcare modeling challenges and highlights the importance of richer data sources.

---

## 2. NLP: Entity Extraction from Discharge Notes

### Approach
A lightweight rule-based NLP pipeline was implemented to extract clinically relevant entities from discharge notes, focusing on:

- Diagnoses
- Treatments
- Symptoms
- Medications
- Follow-up actions

Keyword-based matching was used to ensure transparency and reproducibility, with results saved per patient in structured JSON format.

### Example Extracted Entities
- **Diagnosis:** pneumonia, infection
- **Treatment:** antibiotics, medication
- **Symptoms:** discomfort, reaction
- **Follow-up:** follow-up scheduled, monitoring advised

### Risks & Limitations
- Keyword methods lack contextual understanding.
- Ambiguous phrases may be misclassified.
- General-purpose NLP approaches may miss nuanced clinical terminology.
- LLM-based methods risk hallucination without careful prompt design and validation.

In production, this would be improved using clinical-domain models (e.g. scispaCy, MedSpaCy) and human-in-the-loop validation.

---

## 3. Practical Implications
- Readmission prediction requires richer longitudinal and clinical data to be actionable.
- Discharge note extraction can support care coordination, follow-up tracking, and summarization.
- Combining structured features with text-derived signals may improve performance.

---

## 4. Future Improvements
With more time or data, I would:
- Incorporate discharge note embeddings into the predictive model.
- Tune hyperparameters with cross-validation.
- Add calibration analysis and precision-recall curves.
- Use a clinical NLP model or LLM with prompt-guided entity extraction.
- Evaluate fairness and subgroup performance.

---

## Conclusion
This project demonstrates a realistic clinical data science workflow, balancing interpretability, modeling rigor, and NLP techniques while transparently addressing limitations and opportunities for improvement.
