from src.preprocess import load_data, preprocess_data, split_and_scale
from src.train import train_logistic, train_random_forest
from src.evaluate import evaluate_model
from src.model import save_model

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
     roc_curve, auc, precision_recall_curve,
     confusion_matrix, average_precision_score,
     accuracy_score, precision_score, recall_score)
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load data
df = load_data("data/data.csv")

# Preprocess
X, y = preprocess_data(df)

#EDA

plt.figure(figsize=(5,4))
sns.countplot(x=y)
plt.title("Class Distribution (0 = Benign, 1 = Malignant)")
plt.savefig("outputs/class_distribution.png")
plt.close()

# Split & Scale
X_train, X_test, y_train, y_test, scalar = split_and_scale(X, y)

# Train models
lr_model = train_logistic(X_train, y_train)
rf_model = train_random_forest(X_train, y_train)

# Evaluate Logistic Regression
lr_acc, lr_cm, lr_report = evaluate_model(lr_model, X_test, y_test)

print("\nLogistic Regression")
print("Accuracy:", lr_acc)
print(lr_cm)
print(lr_report)

# Evaluate Random Forest
rf_acc, rf_cm, rf_report = evaluate_model(rf_model, X_test, y_test)

print("\nRandom Forest")
print("Accuracy:", rf_acc)
print(rf_cm)
print(rf_report)

# Save best model
if rf_acc > lr_acc:
    save_model(rf_model)
    print("\nSaved Random Forest Model")
else:
    save_model(lr_model)
    print("\nSaved Logistic Regression Model")

# ROC for Logistic Regression
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# ROC for Random Forest
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Plot ROC Curve
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {roc_auc_lr:.4f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_rf:.4f})")

plt.plot([0, 1], [0, 1], linestyle='--')  # baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("outputs/ROC_curve.png") 
plt.close()    

# Logistic Regression PR
precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_prob_lr)
ap_lr = average_precision_score(y_test, y_prob_lr)

# Random Forest PR
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_prob_rf)
ap_rf = average_precision_score(y_test, y_prob_rf)

plt.plot(recall_lr, precision_lr, label=f"Logistic Regression (AP = {ap_lr:.2f})")
plt.plot(recall_rf, precision_rf, label=f"Random Forest (AP = {ap_rf:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig("outputs/Precision-Recall_curve.png") 
plt.close()


feature_names = X.columns
importances = rf_model.feature_importances_

fi_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Important Features:\n")
print(fi_df.head(10))

# Plot
plt.figure(figsize=(8,5))
plt.barh(fi_df['Feature'][:10], fi_df['Importance'][:10])
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.savefig("outputs/feature_importance.png")
plt.close()

top_features = fi_df['Feature'].head(10)
corr_matrix = df[top_features].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, cmap='RdBu')
plt.title("Top 10 Feature Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png")
plt.close()

y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

metrics = {
    "Metric": ["Accuracy", "Precision", "Recall"],
    "Logistic Regression": [
        accuracy_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_lr)
    ],
    "Random Forest": [
        accuracy_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_rf)
    ]
}

metrics_df = pd.DataFrame(metrics)

# Plot
metrics_df.set_index("Metric").plot(kind="bar")
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.savefig("outputs/model_comparison.png")
plt.close()

# Logistic Regression CM
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/cm_logistic.png")
plt.close()

# Random Forest CM
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/cm_rf.png")
plt.close()


# Proper pipelines for CV
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=5000, random_state=42))
])

rf_pipeline = Pipeline([
    ('model', RandomForestClassifier(random_state=42))
])

# Cross Validation
lr_cv_scores = cross_val_score(lr_pipeline, X, y, cv=5)
rf_cv_scores = cross_val_score(rf_pipeline, X, y, cv=5)

print("\nCross Validation Scores:")
print("Logistic Regression:", lr_cv_scores)
print("Mean LR Accuracy:", lr_cv_scores.mean())

print("Random Forest:", rf_cv_scores)
print("Mean RF Accuracy:", rf_cv_scores.mean())