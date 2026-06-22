"""
Predictive Modeling Using Machine Learning

Features:
- Data preprocessing
- Train/Test split
- Logistic Regression
- Decision Tree
- Random Forest
- Accuracy comparison
- Confusion Matrix
- ROC Curve

Author: Your Name
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)


# =========================
# Load Dataset
# =========================

# Replace with your dataset path
DATASET_PATH = "dataset.csv"

df = pd.read_csv(DATASET_PATH)

print("\nDataset Preview:")
print(df.head())

print("\nDataset Shape:", df.shape)

# =========================
# Data Preprocessing
# =========================

# Remove missing values
df.dropna(inplace=True)

# Assume last column is target variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encode categorical target if needed
if y.dtype == "object":
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

# Convert categorical features
X = pd.get_dummies(X, drop_first=True)

# =========================
# Train-Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# =========================
# Model Training
# =========================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = {
        "model": model,
        "accuracy": accuracy
    }

    print(f"\n{'='*50}")
    print(name)
    print(f"{'='*50}")

    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

# =========================
# Accuracy Comparison
# =========================

accuracy_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": [results[m]["accuracy"] for m in results]
})

plt.figure(figsize=(8, 5))

sns.barplot(
    data=accuracy_df,
    x="Model",
    y="Accuracy",
    palette="viridis"
)

plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# =========================
# Best Model Selection
# =========================

best_model_name = max(
    results,
    key=lambda x: results[x]["accuracy"]
)

best_model = results[best_model_name]["model"]

print(f"\nBest Model: {best_model_name}")

# =========================
# Confusion Matrix
# =========================

best_predictions = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(f"Confusion Matrix ({best_model_name})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# =========================
# ROC Curve (Binary Only)
# =========================

if len(set(y)) == 2:

    probabilities = best_model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, probabilities)

    auc_score = roc_auc_score(y_test, probabilities)

    plt.figure(figsize=(7, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {auc_score:.3f}",
        linewidth=2
    )

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({best_model_name})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"\nROC-AUC Score: {auc_score:.4f}")

else:
    print("\nROC Curve skipped (multiclass dataset detected).")

print("\nProject Completed Successfully!")
