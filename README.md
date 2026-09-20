# Campas2Career — Job Candidate Success Predictor

> Predict whether a job candidate will succeed in a role using Machine Learning and Natural Language Processing.

---



## What This Project Does

A candidate fills in three things:
1. Their professional background (title, industry, experience, skills, description)
2. The job they are applying for (required title, industry, description, experience, skills)
3. Their soft skills (motivation, enthusiasm, communication — rated 1–10)

The AI model then:
- Calculates **text similarity** between the candidate profile and the job description (using cosine similarity, approximating BERT embeddings)
- Applies the **experience score formula** from the research paper (sigmoid function)
- Computes **skill coverage** (percentage of required skills the candidate has)
- Runs a **weighted XGBoost-inspired prediction** to output a success probability
- Returns a verdict, confidence score, 8-factor breakdown, and personalised tips

**Model accuracy: 93%** (as reported in the original paper for the New Model with NLP features)

---

## Project Structure

```
hireiq-project/
│
├── website/                        ← Public-facing web app (no backend needed)
│   ├── index.html                  ← Main page structure
│   ├── style.css                   ← All styling
│   └── app.js                      ← Prediction engine (JavaScript)
│
├── notebook/
│   └── Job_Candidate_Selection_Prediction.ipynb   ← Full Google Colab notebook
│
└── README.md                       ← This file
```

---

### Key findings from the paper

| Model | Accuracy | Precision (Success) | Recall (Success) |
|-------|----------|---------------------|-----------------|
| Old Model (classic features only) | 88% | 0.97 | 0.88 |
| **New Model (classic + NLP)** | **93%** | **0.98** | **0.93** |

Adding NLP text features (BERT embeddings + cosine similarity) improved accuracy by **5 percentage points** over classic recruitment methods. The ML model also outperformed human recruiters who achieved only **84% accuracy** manually.

---

## Technologies Used

### Website
- Pure HTML, CSS, JavaScript — zero dependencies, zero frameworks
- Runs entirely in the browser (no server, no backend, no API key)
- TF-cosine similarity engine (approximates BERT text similarity)
- Sigmoid experience scoring formula (exact from paper)

### Google Colab Notebook
- **Python 3**
- `transformers` — BERT (bert-base-uncased) for real text embeddings
- `xgboost` — XGBoost classifier with 10-fold cross-validation
- `scikit-learn` — metrics, confusion matrix, cross-validation
- `pandas`, `numpy` — data handling
- `matplotlib`, `seaborn` — visualisation

---


## Challenges & Limitations

1. **Data** — The website uses a local text similarity engine. The Colab notebook uses real BERT embeddings for higher accuracy.
2. **Bias** — Any ML recruitment tool can reflect biases present in training data. Results should be used as guidance, not final decisions.
3. **Definition of success** — The paper defines success as passing a 6-month probationary period, which may not match every organisation's definition.

---

## Author

Built as an academic ML/NLP project based on research.
Authors: Sayan Jana, 2026.
