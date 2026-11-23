# 8A Work

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

RANDOM_STATE = 42

feature_names = [
    "attendance", "study_time", "engagement",
    "assessment_score", "help_seeking"
]

df = pd.read_csv("C:\\Users\\abadj\\Documents\\GitHub\\CSC1171\\Week 1\\week8-DataIntro\\data\\week8_evaluation_synt_data.csv")
X = df[feature_names]
y = df["needs_support"]

print(X.head())
print("Class counts:")
print(y.value_counts().sort_index().rename(index={0: "0 = no", 1: "1 = yes"}))

print(y.value_counts(normalize=True).sort_index())

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)

print(f"Train:      {len(X_train):4d} rows ({len(X_train)/len(X):.0%})")
print(f"Validation: {len(X_valid):4d} rows ({len(X_valid)/len(X):.0%})")
print(f"Test:       {len(X_test):4d} rows ({len(X_test)/len(X):.0%})")

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ]),
    "KNN (k=15)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=15))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, random_state=RANDOM_STATE
    )
}

validation_rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    validation_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_valid, pred),
        "Precision": precision_score(y_valid, pred, zero_division=0),
        "Recall": recall_score(y_valid, pred, zero_division=0),
        "F1": f1_score(y_valid, pred, zero_division=0)
    })

validation_results = pd.DataFrame(validation_rows).set_index("Model").sort_values("F1", ascending=False)
validation_results.round(3)

# Choose the model with the highest validation F1 score.
best_name = validation_results.index[0]
best_model = models[best_name]
valid_pred = best_model.predict(X_valid)

ConfusionMatrixDisplay.from_predictions(
    y_valid, valid_pred, labels=[0, 1],
    display_labels=["No support (0)", "Support (1)"],
    cmap="Blues"
)
plt.title(f"Validation confusion matrix: {best_name}")
plt.show()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_rows = []
for name, model in models.items():
    scores = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring=["accuracy", "precision", "recall", "f1"],
        return_train_score=False
    )
    cv_rows.append({
        "Model": name,
        "Accuracy mean": scores["test_accuracy"].mean(),
        "Accuracy SD": scores["test_accuracy"].std(),
        "Precision mean": scores["test_precision"].mean(),
        "Recall mean": scores["test_recall"].mean(),
        "F1 mean": scores["test_f1"].mean(),
        "F1 SD": scores["test_f1"].std()
    })
cv_results = pd.DataFrame(cv_rows).set_index("Model").sort_values("F1 mean", ascending=False)
cv_results.round(3)
safe_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])

safe_scores = cross_validate(
    safe_model, X_train, y_train, cv=cv,
    scoring=["accuracy", "precision", "recall", "f1"]
)

X_trainval = pd.concat([X_train, X_valid])
y_trainval = pd.concat([y_train, y_valid])

# Re-create the selected model with the same settings, then fit on train + validation.
final_models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ]),
    "KNN (k=15)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=15))
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
}
final_model = final_models[best_name]
final_model.fit(X_trainval, y_trainval)
test_pred = final_model.predict(X_test)

print("Final selected model:", best_name)
ConfusionMatrixDisplay.from_predictions(
    y_test, test_pred,
    display_labels=["No support (0)", "Support (1)"],
    cmap="Blues"
)
plt.title(f"Final test results: {best_name}")
plt.show()
