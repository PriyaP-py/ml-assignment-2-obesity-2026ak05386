"""
Train 5 classifiers on a binary or multi-class CSV, evaluate them, and save:
  model/*.joblib          -> full pipelines (preprocessing + model)
  model/label_encoder.joblib
  test_data.csv           -> held-out test split (features + target) for the Streamlit app
  metrics_table.csv       -> comparison table for the README

Usage:  python train_models.py
Edit DATA_PATH and TARGET_COL below for your dataset.
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

# ---------------- CONFIG: change these for your dataset ----------------
DATA_PATH = "obesity.csv"   # created by prepare_data.py
TARGET_COL = "NObeyesdad"    # 7 obesity levels (multi-class)
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ROWS = None              # e.g. 10000 to subsample very large data (SVM/KNN get slow)
# ------------------------------------------------------------------------

os.makedirs("model", exist_ok=True)

df = pd.read_csv(DATA_PATH).drop_duplicates()
if MAX_ROWS and len(df) > MAX_ROWS:
    df = df.sample(MAX_ROWS, random_state=RANDOM_STATE)
df = df.dropna(subset=[TARGET_COL])

X = df.drop(columns=[TARGET_COL])
y_raw = df[TARGET_COL]
n_classes = y_raw.nunique()
assert n_classes >= 2, "Target needs at least 2 classes."
MULTI = n_classes > 2
assert X.shape[1] >= 12 and len(df) >= 500, "Assignment needs >=12 features and >=500 rows."

le = LabelEncoder()
y = pd.Series(le.fit_transform(y_raw), index=df.index)
joblib.dump(le, "model/label_encoder.joblib")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

# Save the test split with the ORIGINAL label values so the app can upload it directly
test_out = X_test.copy()
test_out[TARGET_COL] = y_raw.loc[X_test.index]
test_out.to_csv("test_data.csv", index=False)

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]


def make_ohe():
    """OneHotEncoder that works on old (<1.2) and new scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", make_ohe())]), cat_cols),
    ])


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=RANDOM_STATE),
}

rows = []
for name, clf in models.items():
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)
    avg = "weighted" if MULTI else "binary"
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": (roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
                if MULTI else roc_auc_score(y_test, proba[:, 1])),
        "Precision": precision_score(y_test, pred, average=avg, zero_division=0),
        "Recall": recall_score(y_test, pred, average=avg, zero_division=0),
        "F1": f1_score(y_test, pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    })
    fname = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipe, os.path.join("model", fname))
    print(f"saved model/{fname}")

results = pd.DataFrame(rows).round(4)
results.to_csv("metrics_table.csv", index=False)
print("\n", results.to_string(index=False))
print("\nMarkdown table for README:\n")
try:
    print(results.to_markdown(index=False))
except ImportError:   # tabulate not installed
    print(results.to_string(index=False))