# Obesity Level Classification: Machine Learning Assignment 2

**Course:** Machine Learning, M.Tech (AIML), BITS Pilani WILP
**Author:** [Your Full Name], [Your BITS ID]
**GitHub username used for commits:** Pps08 (see the [authorship note](#9-note-on-authorship-and-github-accounts))

---

## 1. Problem Statement

Obesity is influenced by a person's eating habits, physical activity and body measurements. The goal of this project is to build and compare five machine-learning classifiers that **predict a person's obesity level** (one of seven categories) from 16 lifestyle and physical attributes.

The project covers the full workflow: data preparation, training and evaluation of five models, saving the trained models, and an interactive **Streamlit web app** that loads the saved models, accepts an uploaded test CSV, and shows evaluation metrics, a confusion matrix, a classification report, and a comparison of all models. No model is retrained inside the app.

---

## 2. Dataset Description

| Item | Details |
|---|---|
| Name | Estimation of Obesity Levels Based on Eating Habits and Physical Condition |
| Source | UCI Machine Learning Repository (ID 544); also available on Kaggle |
| Reference | Palechor and De la Hoz Manotas (2019), *Data in Brief* |
| Origin | Individuals from Mexico, Peru and Colombia, aged 14 to 61 |
| Task | Multi-class classification (7 classes) |
| Raw size | 2,111 rows, 16 features + 1 target column |
| After cleaning | 2,087 unique rows (24 exact duplicate rows removed) |
| Train / test split | 80% / 20%, stratified, `random_state=42`: 1,669 training rows, 418 test rows |

**Target column:** `NObeyesdad` with seven classes: Insufficient_Weight, Normal_Weight, Overweight_Level_I, Overweight_Level_II, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III. The classes are close to balanced (about 270 to 350 rows each in the raw data).

**Features (16):**

| Group | Columns |
|---|---|
| Eating habits | `FAVC` (frequent high-calorie food), `FCVC` (vegetable consumption), `NCP` (number of main meals), `CAEC` (eating between meals), `CH2O` (daily water intake), `CALC` (alcohol consumption) |
| Physical activity and lifestyle | `FAF` (physical activity frequency), `TUE` (time using technology devices), `MTRANS` (transport used), `SCC` (calorie monitoring), `SMOKE` (smoking) |
| Body and background | `Gender`, `Age`, `Height`, `Weight`, `family_history_with_overweight` |

The data has 8 numeric and 8 categorical columns.

**Important notes about this dataset**
- About **77% of the records are synthetic**, generated with the SMOTE technique by the dataset authors. Only about 23% were collected directly from people through a web survey. This is a likely reason why scores are high overall.
- The labels are derived from body mass index (BMI), so `Height` and `Weight` carry most of the information about the class.

**Preprocessing (inside each saved pipeline)**
1. Exact duplicate rows are dropped.
2. Numeric columns: median imputation, then standardisation (`StandardScaler`).
3. Categorical columns: most-frequent imputation, then one-hot encoding.
4. The target is label-encoded (`LabelEncoder`, saved as `model/label_encoder.joblib`).

Preprocessing and model are saved together as one scikit-learn `Pipeline` per classifier. Because of this, the app can take a raw uploaded CSV and predict without any separate preprocessing step.

---

## 3. GitHub Repository Link

https://github.com/PriyaP-py/ml-assignment-2-obesity-2026ak05386

**Repository contents**

```
ml-assignment-2-obesity-2026ak05386/
|-- app.py                     Streamlit application
|-- requirements.txt           Python dependencies (pinned versions)
|-- README.md                  This file
|-- test_data.csv              Held-out test split (418 rows, features + target)
|-- train_models.py            Trains the 5 models, computes metrics, saves pipelines
|-- prepare_data.py            Reads the downloaded dataset and creates obesity.csv
|-- model/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- support_vector_machine.joblib
|   `-- label_encoder.joblib
|-- .streamlit/config.toml     Optional app theme
|-- images/                    Optional pictures used by the app
`-- .gitattributes             Keeps LF line endings
```

**Live Streamlit app:** https://YOUR-APP-NAME.streamlit.app

---

## 4. Models Used

All five models were trained on the same training split and evaluated on the same 418-row test split.

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=1000` |
| Decision Tree | `max_depth=6` |
| K-Nearest Neighbors | `n_neighbors=7` |
| Naive Bayes | `GaussianNB` (default settings) |
| Support Vector Machine | `SVC` with RBF kernel, `probability=True` |

**How the metrics are computed.** The problem is multi-class, so Precision, Recall and F1 use **weighted averaging** (each class is weighted by its number of test samples). AUC is computed **one-vs-rest with weighted averaging** from the predicted class probabilities. MCC is the multi-class Matthews Correlation Coefficient.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:--------------|---------:|-------:|----------:|-------:|-------:|-------:|
| Logistic Regression | 0.8947 | 0.9861 | 0.8927 | 0.8947 | 0.8931 | 0.8772 |
| Decision Tree | 0.8947 | 0.9814 | 0.9077 | 0.8947 | 0.8968 | 0.8789 |
| KNN | 0.8038 | 0.9671 | 0.7951 | 0.8038 | 0.7823 | 0.7750 |
| Naive Bayes | 0.5144 | 0.8391 | 0.5252 | 0.5144 | 0.4650 | 0.4566 |
| Support Vector Machine | 0.9115 | 0.9940 | 0.9121 | 0.9115 | 0.9114 | 0.8967 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|:--------------|:------------------------------------|
| Logistic Regression | Performs strongly for a linear model (accuracy 0.8947, AUC 0.9861, MCC 0.8772). This suggests the classes are largely separable by combinations of the scaled numeric features, mainly height and weight. It is fast and easy to interpret, but its linear boundaries leave it slightly behind the SVM. |
| Decision Tree | Matches Logistic Regression in accuracy (0.8947) and has the second-best precision (0.9077) and a marginally higher MCC (0.8789). Its AUC (0.9814) is a little lower, which is expected because a depth-limited tree gives coarse probability estimates. It is easy to explain with rules, but a single tree is more sensitive to the training split. |
| KNN | Clearly weaker (accuracy 0.8038, MCC 0.7750). Distance-based voting is affected by the many one-hot encoded and discrete columns, which add distance that does not help separate the classes. Its high AUC (0.9671) shows that it still ranks the correct class well, but its final hard predictions are less accurate. |
| Naive Bayes | The weakest model by a large margin (accuracy 0.5144, F1 0.4650, MCC 0.4566). Gaussian Naive Bayes assumes independent, normally distributed features. Here, height, weight and the one-hot columns are strongly related and not Gaussian, so the assumption does not hold. The gap between its recall (0.5144) and F1 (0.4650) suggests it performs unevenly across classes. |
| Support Vector Machine | Best on every metric (accuracy 0.9115, AUC 0.9940, precision 0.9121, recall 0.9115, F1 0.9114, MCC 0.8967). The RBF kernel can draw non-linear boundaries between neighbouring obesity levels, which a linear model cannot. The cost is slower training and no direct interpretability. |
| **Overall winner for this dataset** | **Support Vector Machine.** It has the highest score on all six metrics. Logistic Regression and Decision Tree are close behind, KNN is moderate, and Naive Bayes is not suitable for this data without changes. |

**Caveats.** The test set has 418 rows, so each prediction is worth about 0.24 percentage points of accuracy. Differences of less than one point (for example Logistic Regression versus Decision Tree) are within normal variation and should not be over-interpreted. Because most of the data is synthetic, scores are probably higher than they would be on purely real-world data.

---

## 5. Streamlit App: What Is Where

Open the live app, then follow the steps below. The app opens on the **Overview** tab, and the main evaluation results are on the other tabs.

| Required feature | Where to find it in the app |
|---|---|
| Dataset upload (CSV) | **Sidebar, "1. Upload test data (CSV)"**. Upload `test_data.csv` from this repository (a download button for it is in the sidebar). |
| Model selection dropdown | **Sidebar, "2. Choose a model"**. Lists all five models. |
| Display of evaluation metrics | **"Selected Model" tab, top section.** Six colored cards show Accuracy, AUC, Precision, Recall, F1 and MCC for the chosen model. |
| **Confusion matrix** | **"Selected Model" tab, left column, under the heading "Confusion matrix".** The sidebar checkbox "Normalize confusion matrix (row %)" switches between counts and row percentages. |
| Classification report | **"Selected Model" tab, right column, under the heading "Classification report"** (precision, recall, F1 and support for every class). |
| Results of different models on the test data | **"Compare Models" tab.** Shows all five models side by side in one table (best value per column highlighted), a "best model by MCC" badge, a grouped bar chart, and a button to download the table as CSV. |
| Dataset summary | **"Overview" tab.** Row, feature and class counts, class distribution chart, groups of features, and a preview of the uploaded data. |

**Quick walkthrough for evaluators**
1. Open the app link. The page shows a header and an instruction to upload a file.
2. In the sidebar, click **Browse files** and upload `test_data.csv` (or use the sidebar button to download the sample first).
3. Pick a model from the dropdown. The results update immediately.
4. Click the **Selected Model** tab to see the metric cards, the confusion matrix and the classification report.
5. Click the **Compare Models** tab to see all five models on the same uploaded data.

**Notes**
- The uploaded CSV must contain all 16 feature columns and the target column `NObeyesdad`. If something is missing or a label is unknown, the app shows a clear error message instead of crashing.
- The models are loaded from the `model/` folder. Nothing is retrained when a file is uploaded.
- If the app is opened with a Streamlit version that does not support tabs, the sections are shown one below another on a single page.

---

## 6. How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the Obesity Levels CSV from UCI (id 544) or Kaggle into this folder,
#    then create obesity.csv
python prepare_data.py

# 3. Train the five models, save them in model/, and create test_data.csv
python train_models.py

# 4. Start the app
python -m streamlit run app.py
```

`train_models.py` prints the comparison table, saves `metrics_table.csv`, writes the pipelines to `model/`, and creates `test_data.csv`.

---

## 7. Technologies

Python, scikit-learn, pandas, NumPy, matplotlib, joblib, Streamlit.

---

## 8. Execution on BITS Virtual Lab

A screenshot of the assignment execution on the BITS Virtual Lab is included in the submitted PDF, as required.

---

## 9. Note on Authorship and GitHub Accounts

The commits in this repository appear under the GitHub username **Pps08**, while the repository itself is hosted under the username **PriyaP-py**. **Both accounts belong to the same person, [Your Full Name] ([Your BITS ID]), who is the sole author of this assignment.**

The two usernames appear because the Git installation on my computer was signed in to the Pps08 account, while the repository was created under PriyaP-py. No other person contributed to the code, the models, or the app. The commit history shows the work being done in stages: data preparation, model training, saved models and test data, the Streamlit app, requirements, and finally this README.
