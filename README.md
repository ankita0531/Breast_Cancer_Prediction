# Breast Cancer Clinical Data Analysis

Machine learning project to classify breast tumors as **Benign** or **Malignant** using Logistic Regression and Random Forest.

---

## Dataset

**Source:** Wisconsin Breast Cancer Dataset (Kaggle / UCI ML Repository)

| Property | Value |
|---|---|
| Samples | 569 |
| Features | 30 numerical |
| Target | Benign (0) / Malignant (1) |
| Class Split | 62.7% Benign · 37.3% Malignant |

Features describe cell nuclei characteristics from FNA biopsy images — radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension — each measured as mean, standard error, and worst value.

---

## Project Structure

```
CLINICAL-DATA-ANALYSIS/
│
├── data/
│   └── data.csv
│
├── notebooks/
│   └── breast_cancer_eda.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── model.py
│
├── Outputs/
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── ROC_curve.png
│   ├── Precision-Recall_curve.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   ├── cm_logistic.png
│   └── cm_rf.png
│
├── main.py
├── model.pkl
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/your-username/breast-cancer-analysis.git
cd breast-cancer-analysis
pip install -r requirements.txt
python main.py
```

> Place `data.csv` inside the `data/` folder before running.

---

## Results

| Model | Accuracy | AUC-ROC | CV Accuracy (5-Fold) |
|---|---|---|---|
| Logistic Regression | 98.25% | 0.9977 | 97.14% ± 1.49% |
| Random Forest | 97.37% | 0.9950 | 96.04% ± 2.04% |

---

## Outputs Generated

| File | Description |
|---|---|
| `class_distribution.png` | Count of Benign vs Malignant samples |
| `correlation_heatmap.png` | Correlation among top 10 features |
| `ROC_curve.png` | ROC curves for both models |
| `Precision-Recall_curve.png` | PR curves for both models |
| `feature_importance.png` | Top 10 features from Random Forest |
| `model_comparison.png` | Accuracy, Precision, Recall bar chart |
| `cm_logistic.png` | Confusion matrix — Logistic Regression |
| `cm_rf.png` | Confusion matrix — Random Forest |

---

## Tech Stack

Python · scikit-learn · pandas · NumPy · Matplotlib · Seaborn

---

## Future Work

- Hyperparameter tuning with GridSearchCV
- Add SVM and XGBoost for comparison
- SHAP values for clinical explainability
- Deploy as a Streamlit web app

---

## Author

**Ankita Rout**
B.Tech — Computer Science and Engineering
Institute of Technical Education and Research, Siksha 'O' Anusandhan University.

[LinkedIn](https://www.linkedin.com/in/ankita-rout0531/)

[GitHub](https://github.com/ankita0531)