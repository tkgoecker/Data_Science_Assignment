# Data Science Assignment
# Overview
This repository contains a technical assessment demonstrating an end-to-end data science workflow applied to a synthetic clinical dataset. The project addresses two complementary tasks:

1. Predicting whether a patient will be readmitted to the hospital within 30 days  
2. Extracting clinically relevant entities from free-text discharge notes  

The solution emphasizes practical feature engineering, interpretable modeling, NLP techniques, and clear communication of results.

---

## Problems Addressed

### 1. Readmission Prediction (Binary Classification)
Using structured patient data (demographics, diagnosis codes, medication type, and utilization history), classification models are trained to predict 30-day hospital readmission.

### 2. Discharge Note Entity Extraction (NLP)
Free-text discharge notes are processed to extract key clinical information such as diagnoses, treatments, symptoms, medications, and follow-up actions.

---

## Project Structure
Data_Science_Assignment/
│
├── Data/
│ └── Assignment_Data.xlsx
│
├── src/
│ ├── train_model.py # Model training and evaluation
│ ├── nlp_entities.py # NLP entity extraction
│ ├── utils.py # Shared utilities
│ └── init.py
│
├── reports/
│ ├── metrics.json
│ ├── roc_logistic_regression.png
│ ├── roc_random_forest.png
│ ├── confusion_logistic_regression.png
│ ├── confusion_random_forest.png
│ ├── nlp_entities.json
│ └── final_report.md
│
├── requirements.txt
└── README.md

## Setup Instructions

## Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate

pip install -r requirements.txt
