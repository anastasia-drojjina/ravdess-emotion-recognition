"""
Trains sklearn baseline models (SVM, Random Forest) and saves results.
Equivalent to notebook 02_baseline.ipynb.
Run: python run_baseline.py
"""
import os, sys, io
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
_c_pkgs = r"C:\Users\Anastasia\ravdess-pkgs"
_site = _c_pkgs if os.path.isdir(_c_pkgs) else os.path.join(os.path.dirname(os.path.abspath(__file__)), "site-packages")
if os.path.isdir(_site) and _site not in sys.path:
    sys.path.insert(0, _site)

import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

os.makedirs("results", exist_ok=True)
EMOTION_NAMES = ["neutral","calm","happy","sad","angry","fearful"]

if not os.path.exists("results/mfcc_features.npz"):
    print("ERROR: run extract_representations/extract_features.py first!")
    sys.exit(1)

data = np.load("results/mfcc_features.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_test,  y_test  = data["X_test"],  data["y_test"]
print(f"Train: {X_train.shape}  Test: {X_test.shape}")

classifiers = {
    "SVM (RBF)":    Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"))]),
    "Random Forest": Pipeline([("sc", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))]),
    "k-NN (k=5)":   Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
print("\nCross-validation:")
for name, clf in classifiers.items():
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:20s}  F1-macro: {scores.mean():.4f} +/- {scores.std():.4f}")

# Best model on test set
best_name = max(cv_results, key=lambda n: cv_results[n].mean())
print(f"\nBest: {best_name}")
best_clf = classifiers[best_name]
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro",    zero_division=0)
f1w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
print(f"Accuracy: {acc:.4f}  F1 macro: {f1m:.4f}  F1 weighted: {f1w:.4f}")
print(classification_report(y_test, y_pred, target_names=EMOTION_NAMES, zero_division=0))

json.dump({"accuracy": acc, "f1_macro": f1m, "f1_weighted": f1w},
          open("results/metrics_baseline.json", "w"), indent=2)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_n = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES, ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("results/cm_baseline.png", dpi=150)
plt.close()

# CV comparison chart
fig, ax = plt.subplots(figsize=(8, 4))
names = list(cv_results.keys())
means = [cv_results[n].mean() for n in names]
stds  = [cv_results[n].std()  for n in names]
bars = ax.bar(names, means, yerr=stds, capsize=5,
              color=sns.color_palette("Set2", len(names)), edgecolor="black")
ax.set_ylim(0, 1); ax.set_ylabel("F1 Macro (5-fold CV)")
ax.set_title("Baseline Classifiers")
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f"{m:.3f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("results/baseline_cv.png", dpi=150)
plt.close()

print("Saved -> results/metrics_baseline.json, cm_baseline.png, baseline_cv.png")
