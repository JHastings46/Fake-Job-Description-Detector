# Fake Job Description Detector (DistilBERT)

Real vs. fake job posting detection using a fine-tuned DistilBERT classifier.

## Overview
Fake job postings reduce trust in online job platforms and can mislead applicants. This project builds a supervised NLP classifier that predicts whether a job posting is **real (0)** or **fake (1)** using the Kaggle *Real Fake Job Posting* dataset.

I benchmarked a transformer-based approach (DistilBERT) against classic TF-IDF + linear models. DistilBERT delivered the strongest overall performance and the best fraud recall.

## Key Results (Hold-out Test Set)
**Best model: DistilBERT (Hugging Face)**
- Accuracy: **0.9424**
- F1-score: **0.9429**
- Recall: **0.9538** (strong at catching fraudulent posts)

**Baselines (TF-IDF)**
- Multinomial Naive Bayes: Accuracy 0.9222, F1 0.9213  
- Linear SVC: Accuracy 0.9222, F1 0.9213  
- Ridge Classifier: Accuracy 0.9164, F1 0.9150  
- SGD Classifier: Accuracy 0.9107, F1 0.9086  

## What’s inside this repo
Typical contents you’ll see in this project:
- `DataMiningproject.ipynb` — end-to-end notebook (EDA → preprocessing → modeling → evaluation)

## Data
- Dataset: **Kaggle – Real Fake Job Posting**
- Target label: `fraudulent`  
  - **0 = real**
  - **1 = fake**

## Method Summary
### 1) Exploratory Data Analysis (EDA)
- Checked label balance and missingness
- Explored relationships between categorical fields and the target
- Used Cramér’s V to identify categorical predictors with stronger association:
  - `industry`, `state`, `has_company_logo`

### 2) Feature Construction
- Tested text fields with a Logistic Regression baseline (F1 comparison)
- Best performing text fields included `company_profile` and `description`
- Combined selected text + metadata-as-text into a single model input:
  - `final_text`

### 3) Modeling
**Transformer model**
- `distilbert-base-uncased` (Hugging Face Transformers + PyTorch)
- Tokenization with a max sequence length primarily at **512**
- `DataCollatorWithPadding` for dynamic padding
- Hyperparameter tuning on learning rate / epochs / max length

**Baselines**
- TF-IDF vectorization
- Multinomial Naive Bayes, Linear SVC, Ridge, SGD, and other linear models

### 4) Evaluation
- Consistent hold-out test split for fair benchmarking
- Metrics reported:
  - Accuracy, Precision, Recall, F1-score
  - Confusion matrix

## Truncation & Practical Constraints
DistilBERT has a **512 token** limit. Since metadata + job text can exceed this, truncation risk was a real issue.

Token length percentiles for `final_text`:
- 50th: 307  
- 75th: 468  
- 90th: 653  
- 95th: 808.45  
- 99th: 1104.07  

I compared `max_length=512` (more context, slower) vs `max_length=385` (faster, more truncation).

### Validation Confusion Matrices (n=173)
Baseline DistilBERT (max_length≈512): `[[77, 7], [5, 84]]`  
- 7 false alarms, 5 misses

Tuned DistilBERT (max_length=385): `[[78, 6], [6, 83]]`  
- 6 false alarms, 6 misses

Both had the same accuracy on validation (0.9306), but the error pattern shifted slightly.

## How to Run
## Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UJIPU6JUJSE23gJumsFS1GDak7uNeguC?usp=sharing)
Click the badge to open the notebook, then go to **Runtime → Run all**.




