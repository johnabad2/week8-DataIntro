# 8B Work

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

RANDOM_STATE = 42  # fixes the randomness

feature_names = ["attendance", "study_time", "engagement", "assessment_score", "help_seeking"]

data_path = Path(__file__).resolve().parent.parent / "data" / "week8_evaluation_synt_data.csv"
df = pd.read_csv(data_path)
X = df[feature_names]
y = df["needs_support"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print("Training rows:", len(X_train), "| Test rows (kept aside):", len(X_test))
y_train.value_counts(normalize=True).rename("training class proportion")

forest = RandomForestClassifier(
    n_estimators=200,     
    max_features="sqrt", 
    bootstrap=True,         
    random_state=RANDOM_STATE,
    n_jobs=-1              
)
forest.fit(X_train, y_train)
print("Number of trees in the forest:", len(forest.estimators_))

votes = pd.DataFrame(
    {f"tree {i}": tree.predict(X_train.to_numpy()).astype(int)
     for i, tree in enumerate(forest.estimators_[:5], start=1)},
    index=X_train.index
)
votes["forest (all 200 trees)"] = forest.predict(X_train)

votes[votes.iloc[:, :5].nunique(axis=1) > 1].head()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)  # same class mix in every fold
metrics = ["accuracy", "precision", "recall", "f1"]

scores = cross_validate(forest, X_train, y_train, cv=cv, scoring=metrics)

cv_summary = pd.DataFrame({
    "mean": [scores[f"test_{m}"].mean() for m in metrics],
    "SD":   [scores[f"test_{m}"].std() for m in metrics],
}, index=metrics)
cv_summary.round(3)

tree_counts = [1, 5, 10, 50, 100, 200]
f1_by_size = []

for n in tree_counts:
    model = RandomForestClassifier(n_estimators=n, max_features="sqrt",
                                   random_state=RANDOM_STATE, n_jobs=-1)
    f1 = cross_validate(model, X_train, y_train, cv=cv, scoring="f1")["test_score"]
    f1_by_size.append(f1.mean())

plt.plot(tree_counts, f1_by_size, marker="o")
plt.xscale("log")
plt.xticks(tree_counts, tree_counts)
plt.xlabel("Number of trees (log scale)")
plt.ylabel("Mean cross-validated F1")
plt.title("Adding trees helps at first, then levels off")
plt.show()

importances = pd.Series(forest.feature_importances_, index=feature_names).sort_values()

importances.plot.barh()
plt.xlabel("Importance (sums to 1)")
plt.title("Random Forest feature importances")
plt.show()

test_pred = forest.predict(X_test)

ConfusionMatrixDisplay.from_predictions(y_test, test_pred, cmap="Blues")
plt.title("Random Forest: test set confusion matrix")
plt.show()
