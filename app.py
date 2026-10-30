"""
Obesity Level Classification - Model Evaluation Dashboard (Streamlit)
- Loads pre-trained pipelines from model/ (no retraining)
- Upload a test CSV (features + target), choose a model, view metrics,
  confusion matrix / classification report, and compare all five models.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
TARGET_COL = "NObeyesdad"
GREEN, TEAL, AMBER = "#2E7D32", "#00897B", "#F9A825"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
METRIC_COLORS = {"Accuracy": "#2E7D32", "AUC": "#00897B", "Precision": "#1565C0",
                 "Recall": "#6A1B9A", "F1": "#EF6C00", "MCC": "#C62828"}

# Inline SVG icons: decorative graphics that need no image files
ICONS = {
    "apple": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="{s}" height="{s}"><circle cx="24" cy="38" r="17" fill="#EF5350"/><circle cx="40" cy="38" r="17" fill="#EF5350"/><rect x="30" y="14" width="4" height="12" rx="2" fill="#6D4C41"/><ellipse cx="42" cy="15" rx="10" ry="5" transform="rotate(-25 42 15)" fill="#A5D6A7"/></svg>',
    "scale": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="{s}" height="{s}"><rect x="8" y="10" width="48" height="44" rx="9" fill="#FFFFFF"/><path d="M16 34 A16 16 0 0 1 48 34 Z" fill="#E8F5E9" stroke="#2E7D32" stroke-width="3"/><line x1="32" y1="34" x2="40" y2="24" stroke="#C62828" stroke-width="3" stroke-linecap="round"/><circle cx="32" cy="34" r="3" fill="#2E7D32"/><rect x="20" y="42" width="24" height="6" rx="3" fill="#B0BEC5"/></svg>',
    "dumbbell": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="{s}" height="{s}"><rect x="6" y="22" width="8" height="20" rx="3" fill="#FFB74D"/><rect x="14" y="16" width="8" height="32" rx="3" fill="#FFB74D"/><rect x="22" y="29" width="20" height="6" fill="#ECEFF1"/><rect x="42" y="16" width="8" height="32" rx="3" fill="#FFB74D"/><rect x="50" y="22" width="8" height="20" rx="3" fill="#FFB74D"/></svg>',
    "heart": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="{s}" height="{s}"><path d="M32 56 C8 40 4 26 10 17 C16 9 27 10 32 19 C37 10 48 9 54 17 C60 26 56 40 32 56 Z" fill="#EC407A"/><polyline points="14,32 24,32 28,24 34,42 38,32 50,32" fill="none" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>',
}

# Feature groups shown as cards on the Overview tab (only columns present are listed)
FEATURE_GROUPS = [
    ("Eating habits", "apple",
     {"FAVC": "high-calorie food", "FCVC": "vegetable intake", "NCP": "main meals per day",
      "CAEC": "snacking", "CH2O": "water intake", "CALC": "alcohol"}),
    ("Physical activity", "dumbbell",
     {"FAF": "activity frequency", "TUE": "technology use time", "MTRANS": "transport used",
      "SCC": "calorie monitoring", "SMOKE": "smoking"}),
    ("Body and background", "scale",
     {"Gender": "gender", "Age": "age", "Height": "height", "Weight": "weight",
      "family_history_with_overweight": "family history"}),
]


def icon(name, size=64):
    return ICONS[name].replace("{s}", str(size))


def feature_card(title, icon_name, items):
    lis = "".join(f"<li><b>{k}</b> - {v}</li>" for k, v in items.items())
    return (f'<div class="fcard"><div class="ficon">{icon(icon_name, 44)}</div>'
            f'<div><div class="ftitle">{title}</div><ul>{lis}</ul></div></div>')


def find_photos(limit=3):
    """Optional photos: put .jpg/.png files in an 'images' folder next to app.py."""
    img_dir = BASE_DIR / "images"
    if not img_dir.exists():
        return []
    files = sorted(p for p in img_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
                   and p.stem.lower() != "logo")
    return files[:limit]


# st.cache_resource needs Streamlit >= 1.18; fall back to st.cache on older versions
_cache = getattr(st, "cache_resource", None) or st.cache(allow_output_mutation=True)


@_cache
def load_artifacts():
    models = {name: joblib.load(MODEL_DIR / f) for name, f in MODEL_FILES.items()}
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    return models, label_encoder


def compute_metrics(model, X, y_true, n_classes):
    """Return metrics dict and predictions. Weighted averaging for multi-class."""
    pred = model.predict(X)
    proba = model.predict_proba(X)
    multi = n_classes > 2
    avg = "weighted" if multi else "binary"
    try:
        auc = (roc_auc_score(y_true, proba, multi_class="ovr", average="weighted",
                             labels=np.arange(n_classes))
               if multi else roc_auc_score(y_true, proba[:, 1]))
    except ValueError:   # e.g. a class missing from the uploaded file
        auc = float("nan")
    metrics = {
        "Accuracy": accuracy_score(y_true, pred),
        "AUC": auc,
        "Precision": precision_score(y_true, pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, pred),
    }
    return metrics, pred


def metric_card(label, value):
    txt = "n/a" if value is None or np.isnan(value) else f"{value:.4f}"
    color = METRIC_COLORS.get(label, GREEN)
    return (f'<div class="mcard" style="border-left-color:{color}">'
            f'<div class="mlabel">{label}</div><div class="mvalue">{txt}</div></div>')


def info_card(label, value):
    return (f'<div class="mcard" style="border-left-color:{AMBER}">'
            f'<div class="mlabel">{label}</div><div class="mvalue">{value}</div></div>')


def plot_confusion_matrix(y_true, pred, class_names, normalize=False):
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, pred, labels=labels)
    if normalize:
        sums = cm.sum(axis=1, keepdims=True)
        cm_show = np.divide(cm, sums, out=np.zeros(cm.shape, dtype=float), where=sums != 0)
    else:
        cm_show = cm
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_show, cmap="Greens")
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    thresh = cm_show.max() / 2 if cm_show.max() else 0
    for i in range(cm_show.shape[0]):
        for j in range(cm_show.shape[1]):
            txt = f"{cm_show[i, j]:.2f}" if normalize else str(int(cm_show[i, j]))
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if cm_show[i, j] > thresh else "black", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_class_distribution(series):
    counts = series.value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(counts.index, counts.values, color=TEAL)
    ax.set_xlabel("Number of rows")
    for i, v in enumerate(counts.values):
        ax.text(v, i, f" {v}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_comparison(comp):
    comp = comp.fillna(0)
    names = list(comp.index)
    metrics = list(comp.columns)
    x = np.arange(len(names))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for k, m in enumerate(metrics):
        ax.bar(x + k * width, comp[m].values, width, label=m,
               color=METRIC_COLORS.get(m, GREEN))
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend(ncol=len(metrics), loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False)
    fig.tight_layout()
    return fig


# ----------------------------- UI -----------------------------
st.set_page_config(page_title="Obesity Level Classifier", layout="wide")

st.markdown("""
<style>
.hero {background: linear-gradient(120deg, #2E7D32, #00897B); padding: 28px 32px;
       border-radius: 14px; margin-bottom: 18px; display: flex; flex-wrap: wrap;
       align-items: center; justify-content: space-between; gap: 14px;}
.hero-icons svg {margin-left: 10px;}
.fcard {display: flex; gap: 14px; background: #FFFFFF; border-radius: 12px; padding: 16px;
        margin-bottom: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.12);}
.ficon {background: #2E7D32; border-radius: 12px; padding: 8px; height: 44px;}
.ftitle {color: #1F2933; font-weight: 700; font-size: 1.05rem; margin-bottom: 4px;}
.fcard ul {color: #3E4C59; margin: 0; padding-left: 18px; font-size: 0.88rem;}
.hero h1 {color: #FFFFFF; margin: 0; font-size: 2rem;}
.hero p {color: #E8F5E9; margin: 6px 0 0 0; font-size: 1.02rem;}
.mcard {background: #FFFFFF; border-left: 6px solid #2E7D32; border-radius: 10px;
        padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.12);}
.mlabel {color: #52606D; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em;}
.mvalue {color: #1F2933; font-size: 1.7rem; font-weight: 700;}
.badge {display: inline-block; background: #2E7D32; color: #FFFFFF; padding: 6px 14px;
        border-radius: 20px; font-weight: 600;}
.footer {color: #7B8794; font-size: 0.85rem; text-align: center; margin-top: 30px;}
</style>
""", unsafe_allow_html=True)

hero_icons = "".join(icon(n, 56) for n in ("apple", "dumbbell", "scale", "heart"))
st.markdown(
    '<div class="hero"><div><h1>Obesity Level Classifier</h1>'
    '<p>Evaluate five trained machine-learning models on your own test data, '
    'inspect their errors, and find the best performer.</p></div>'
    f'<div class="hero-icons">{hero_icons}</div></div>',
    unsafe_allow_html=True)

try:
    models, label_encoder = load_artifacts()
except Exception as e:
    stamp = MODEL_DIR / "versions.json"
    saved_with = (json.loads(stamp.read_text()).get("scikit-learn", "unknown")
                  if stamp.exists() else "unknown (no versions.json: the model files are old)")
    st.error("Could not load saved models from the 'model' folder.\n\n"
             f"- scikit-learn installed in this app: {sklearn.__version__}\n"
             f"- scikit-learn that saved the models: {saved_with}\n\n"
             "If these differ, retrain with train_models.py, commit the new model/ files, "
             f"and pin scikit-learn=={sklearn.__version__} in requirements.txt.\n\n"
             f"Details: {e}")
    st.stop()

class_names = list(label_encoder.classes_)

# ---- sidebar ----
logo_path = BASE_DIR / "images" / "logo.png"
if logo_path.exists():
    st.sidebar.image(str(logo_path), width=120)
st.sidebar.title("Controls")
uploaded = st.sidebar.file_uploader("1. Upload test data (CSV)", type=["csv"])
model_name = st.sidebar.selectbox("2. Choose a model", list(MODEL_FILES.keys()))
normalize = st.sidebar.checkbox("Normalize confusion matrix (row %)", value=False)

sample_path = BASE_DIR / "test_data.csv"
if sample_path.exists():
    st.sidebar.download_button("Download sample test_data.csv",
                               data=sample_path.read_bytes(),
                               file_name="test_data.csv", mime="text/csv")
st.sidebar.markdown("---")
st.sidebar.caption(f"Models are pre-trained; nothing is retrained in this app. (scikit-learn {sklearn.__version__})")

with st.expander("About this project"):
    st.write(
        "**Task:** multi-class classification of obesity level from eating habits and "
        "physical condition (UCI: Estimation of Obesity Levels, 2,111 records, 16 features, "
        f"{len(class_names)} classes).")
    st.write("**Models:** Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM.")
    st.write("**Metrics:** Accuracy, AUC (one-vs-rest, weighted), Precision, Recall, F1 "
             "(weighted) and Matthews Correlation Coefficient.")
    st.write("**Classes:** " + ", ".join(class_names))

if uploaded is None:
    st.info("Start by uploading a test CSV in the sidebar. It must contain every feature "
            f"column plus the target column `{TARGET_COL}`. You can download a ready-made "
            "sample from the sidebar.")
    st.stop()

# ---- read and validate the uploaded file ----
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read the CSV: {e}")
    st.stop()

df.columns = [c.strip() for c in df.columns]
if TARGET_COL not in df.columns:
    st.error(f"The uploaded file must contain the target column '{TARGET_COL}'. "
             f"Found columns: {list(df.columns)}")
    st.stop()

df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
unknown = set(df[TARGET_COL]) - set(class_names)
if unknown:
    st.error(f"Unknown labels in '{TARGET_COL}': {sorted(unknown)}. Expected only: {class_names}")
    st.stop()

X = df.drop(columns=[TARGET_COL])
y_true = label_encoder.transform(df[TARGET_COL])

expected = getattr(models[model_name].named_steps["prep"], "feature_names_in_", None)
if expected is not None:
    missing = [c for c in expected if c not in X.columns]
    if missing:
        st.error(f"Missing required feature columns: {missing}")
        st.stop()
    X = X[list(expected)]

try:
    metrics, pred = compute_metrics(models[model_name], X, y_true, len(class_names))
    rows = []
    for name, mdl in models.items():
        m, _ = compute_metrics(mdl, X, y_true, len(class_names))
        rows.append({"ML Model Name": name, **m})
except Exception as e:
    st.error("Prediction failed. Check that the columns and values match the training "
             f"data. Details: {e}")
    st.stop()

comparison = pd.DataFrame(rows).set_index("ML Model Name").round(4)

# tabs (fallback to plain containers on very old Streamlit)
labels = ["Overview", "Selected Model", "Compare Models"]
tabs = st.tabs(labels) if hasattr(st, "tabs") else [st.container() for _ in labels]

with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.markdown(info_card("Rows", df.shape[0]), unsafe_allow_html=True)
    c2.markdown(info_card("Features", X.shape[1]), unsafe_allow_html=True)
    c3.markdown(info_card("Classes", len(class_names)), unsafe_allow_html=True)

    groups = [(t, i, {k: v for k, v in items.items() if k in X.columns})
              for t, i, items in FEATURE_GROUPS]
    groups = [g for g in groups if g[2]]
    if groups:
        st.subheader("What the model looks at")
        for col, (t, i, items) in zip(st.columns(len(groups)), groups):
            col.markdown(feature_card(t, i, items), unsafe_allow_html=True)

    photos = find_photos()
    if photos:
        st.subheader("Gallery")
        for col, p in zip(st.columns(len(photos)), photos):
            col.image(str(p), caption=p.stem.replace("_", " ").title(), width=300)

    left, right = st.columns(2)
    with left:
        st.subheader("Class distribution")
        st.pyplot(plot_class_distribution(df[TARGET_COL]))
    with right:
        st.subheader("Data preview")
        st.dataframe(df.head(10))

with tabs[1]:
    st.subheader(f"Results for: {model_name}")
    items = list(metrics.items())
    for start in (0, 3):
        cols = st.columns(3)
        for col, (k, v) in zip(cols, items[start:start + 3]):
            col.markdown(metric_card(k, v), unsafe_allow_html=True)
    if len(class_names) > 2:
        st.caption("Multi-class: precision, recall, F1 and AUC use weighted averaging "
                   "(AUC is one-vs-rest).")
    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        st.pyplot(plot_confusion_matrix(y_true, pred, class_names, normalize))
    with right:
        st.subheader("Classification report")
        report = classification_report(y_true, pred, labels=np.arange(len(class_names)),
                                       target_names=class_names, output_dict=True,
                                       zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3))

with tabs[2]:
    st.subheader("All models on the uploaded data")
    best = comparison["MCC"].idxmax()
    st.markdown(f'Best model by MCC: <span class="badge">{best}</span>',
                unsafe_allow_html=True)
    st.write("")
    try:
        st.dataframe(comparison.style.highlight_max(axis=0, color="#C8E6C9"))
    except Exception:
        st.dataframe(comparison)
    st.pyplot(plot_comparison(comparison))
    st.download_button("Download comparison table (CSV)",
                       data=comparison.to_csv().encode("utf-8"),
                       file_name="model_comparison.csv", mime="text/csv")

st.markdown('<div class="footer">Machine Learning - Assignment 2</div>',
            unsafe_allow_html=True)