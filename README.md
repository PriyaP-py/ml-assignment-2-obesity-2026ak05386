# Obesity Level Classification: Machine Learning Assignment 2

**Course:** Machine Learning, M.Tech (AIML), BITS Pilani WILP
**Author:** [Your Full Name], [Your BITS ID]
**GitHub username used for commits:** Pps08 (see the [authorship note](#9-note-on-authorship-and-github-accounts)) and also Priya
Streamlit app link : https://ml-assignment-2-obesity-2026ak05386-luyz2xk6xzjbhhkruunfrj.streamlit.app/ 
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

**Live Streamlit app:** https://ml-assignment-2-obesity-2026ak05386-luyz2xk6xzjbhhkruunfrj.streamlit.app/

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

## 5. Live Streamlit Application: Access and User Guide

### 5.1 Open the deployed application

**Live application:**  
[Obesity Level Classifier – Open Streamlit App](https://ml-assignment-2-obesity-2026ak05386-luyz2xk6xzjbhhkruunfrj.streamlit.app/)

Open the link in a web browser. The application is deployed using Streamlit and provides an interactive interface for evaluating the five pre-trained machine-learning classifiers. The app loads the saved pipelines from the repository; it does not train or retrain models when a user uploads a CSV.

If the link does not open immediately, refresh the page and allow the Streamlit application a little time to start. The deployed app may be temporarily unavailable while the hosting service is starting or updating.

### 5.2 First-time use: upload test data

The application expects a **CSV file containing all 16 input features and the target column `NObeyesdad`**.

1. Open the live app using the link above.
2. In the left-hand **Controls** sidebar, find **“1. Upload test data (CSV)”**.
3. Click the upload area and select `test_data.csv`. This sample test file is included in the repository. The sidebar also provides a **“Download sample test_data.csv”** button when that file is available to the deployed app.
4. After the file is uploaded and validated, the app calculates predictions and evaluation metrics for the selected model and all five models.
5. Use **“2. Choose a model”** to select the classifier whose detailed results you want to inspect.

The uploaded CSV must include the feature columns used during training and the target column `NObeyesdad`. The target values must match the seven known class labels. The app checks for the target column and unknown target labels, and reports missing model input features or prediction errors rather than silently evaluating an incomplete file.

**Target classes**
- `Insufficient_Weight`
- `Normal_Weight`
- `Overweight_Level_I`
- `Overweight_Level_II`
- `Obesity_Type_I`
- `Obesity_Type_II`
- `Obesity_Type_III`

### 5.3 Sidebar controls

| Control | Purpose |
|---|---|
| **Upload test data (CSV)** | Uploads the test dataset to be evaluated. The file must contain the input features and `NObeyesdad`. |
| **Choose a model** | Selects one of the five saved classifiers for the detailed **Selected Model** view. Changing the selection updates the selected-model results. |
| **Normalize confusion matrix (row %)** | Changes the confusion matrix from raw counts to row-wise proportions. This helps compare class-wise performance when class counts differ. |
| **Download sample test_data.csv** | Downloads the sample test dataset provided with the project, if present in the app deployment. |

The app also displays the installed scikit-learn version in the sidebar. This is useful because the saved model pipelines were created with scikit-learn and can depend on compatible package versions.

### 5.4 “About this project” section

Expand **“About this project”** below the banner to see a short description of the classification task, the dataset, the five models, the evaluation metrics and the seven target classes. This section provides context before examining the results.

### 5.5 Overview tab

The **Overview** tab gives a summary of the uploaded test dataset and the features used by the classifier.

| Feature | What it shows |
|---|---|
| **Rows** | Number of records in the uploaded CSV. |
| **Features** | Number of input columns passed to the model after the target column is separated. |
| **Classes** | Number of obesity-level categories known to the saved label encoder. |
| **What the model looks at** | Feature cards grouped into Eating habits, Physical activity, and Body and background. Only feature descriptions for columns present in the uploaded data are displayed. |
| **Class distribution** | A bar chart showing the number of uploaded records in each `NObeyesdad` category. |
| **Data preview** | A table showing the first 10 rows of the uploaded dataset. |
| **Gallery** | Optional images from an `images/` folder, if image files are included in the deployment. |

**How to interpret the class distribution:** the chart shows how many records of each obesity category are present in the uploaded test file. It describes the test data; it is not a model-accuracy chart.

**Feature groups shown in the app**
- **Eating habits:** frequent high-calorie food (`FAVC`), vegetable consumption (`FCVC`), number of main meals (`NCP`), eating between meals (`CAEC`), water intake (`CH2O`) and alcohol consumption (`CALC`).
- **Physical activity:** physical activity frequency (`FAF`), technology-use time (`TUE`), transport (`MTRANS`), calorie monitoring (`SCC`) and smoking (`SMOKE`).
- **Body and background:** gender, age, height, weight and family history of overweight.

### 5.6 Selected Model tab

The **Selected Model** tab presents the detailed evaluation of the model currently selected in the sidebar.

**A. Six evaluation metric cards**

The top of the tab displays:
- **Accuracy:** proportion of all test records classified correctly.
- **AUC:** area under the ROC curve. For this seven-class problem, the app calculates one-vs-rest AUC and uses weighted averaging based on class support.
- **Precision:** among the records predicted as a class, how many were correct. The app reports weighted precision.
- **Recall:** among the actual records in a class, how many were identified correctly. The app reports weighted recall.
- **F1-score:** harmonic mean of precision and recall. The app reports weighted F1.
- **MCC (Matthews Correlation Coefficient):** a balanced measure of the agreement between true and predicted labels. It accounts for all classes and is useful when assessing multi-class classification.

For this multi-class task, precision, recall and F1 use **weighted averaging**. A class contributes according to its number of records in the test data. AUC is computed using a one-vs-rest approach with weighted averaging. If AUC cannot be calculated (for example, when a class is absent from the uploaded test data), the app displays it as unavailable (`n/a`).

**B. Confusion matrix**

The confusion matrix compares actual classes with predicted classes:
- Rows represent the **actual** obesity categories.
- Columns represent the **predicted** obesity categories.
- Diagonal cells represent correct predictions.
- Off-diagonal cells represent misclassifications and show which categories the model confused.

Use the sidebar checkbox **“Normalize confusion matrix (row %)”** to switch from counts to row-normalized proportions. In the normalized view, each row indicates how the model's predictions are distributed for records belonging to that actual class. A strong model should generally have larger values along the diagonal.

**C. Classification report**

The classification report provides per-class:
- **Precision:** reliability of predictions for that class.
- **Recall:** proportion of actual records of that class correctly identified.
- **F1-score:** balance between precision and recall.
- **Support:** number of actual test records belonging to that class.

This report is useful for identifying categories where the model performs less well, even if its overall accuracy is high.

### 5.7 Compare Models tab

The **Compare Models** tab evaluates all five saved classifiers on the same uploaded test data.

| Element | Explanation |
|---|---|
| **Best model by MCC** | Displays the model with the highest Matthews Correlation Coefficient in the current comparison. |
| **Comparison table** | Shows Accuracy, AUC, Precision, Recall, F1 and MCC for all five models. |
| **Highlighted values** | The highest value in each metric column is highlighted to make comparisons easier. |
| **Grouped bar chart** | Visualizes the six metrics for each model. Scores are displayed on a 0 to 1 scale. |
| **Download comparison table (CSV)** | Downloads the current model-comparison metrics as `model_comparison.csv`. |

All five models are evaluated against the same uploaded records, which makes their scores directly comparable for that test file. The model with the highest MCC is identified as the best model by the app. The best value may differ by metric, so inspect the full table rather than relying on only one score.

### 5.8 Models included in the application

The application loads five pre-trained scikit-learn pipelines. Each pipeline includes the required preprocessing and classifier, so the uploaded raw feature columns can be passed to the pipeline.

| Model | Configuration used in this assignment | How it works | Main strengths and considerations |
|---|---|---|---|
| **Logistic Regression** | `max_iter=1000` | Learns class decision scores from a linear combination of the input features and converts them into class probabilities. It is a linear classification method. | Fast and relatively interpretable; provides a useful baseline. It may not capture complex non-linear boundaries between obesity categories. |
| **Decision Tree** | `max_depth=6` | Learns a sequence of feature-based decision rules. Each split divides records into groups, and a leaf produces the predicted class. | Easy to explain as rules and can model non-linear relationships. A single tree can be sensitive to the training data; limiting depth helps control complexity. |
| **K-Nearest Neighbors (KNN)** | `n_neighbors=7` | Finds the seven training records closest to a new record in feature space and predicts based on their class votes. | Simple, instance-based method that can model local patterns. Its performance depends on feature scaling and the distance representation; one-hot encoded categorical features can affect distances. |
| **Gaussian Naive Bayes** | `GaussianNB` with default settings | Uses Bayes’ theorem and assumes features are conditionally independent given the class, with numeric feature distributions modelled using Gaussian distributions. | Very fast and simple. Its independence and distribution assumptions may not fit correlated lifestyle and body-measurement features. |
| **Support Vector Machine (SVM)** | RBF kernel, `probability=True` | Finds decision boundaries that separate classes. The radial basis function (RBF) kernel allows non-linear boundaries by representing similarity between records. Probability estimates are enabled for AUC evaluation. | Can model complex non-linear class boundaries and performed best on the reported test split. Training can be slower and the resulting model is less directly interpretable. |

### 5.9 Reported model results

The following values are the results documented for the assignment's 418-row held-out test split. When you upload a different CSV, the application recalculates the metrics, so the displayed values may differ.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8947 | 0.9861 | 0.8927 | 0.8947 | 0.8931 | 0.8772 |
| Decision Tree | 0.8947 | 0.9814 | 0.9077 | 0.8947 | 0.8968 | 0.8789 |
| KNN | 0.8038 | 0.9671 | 0.7951 | 0.8038 | 0.7823 | 0.7750 |
| Naive Bayes | 0.5144 | 0.8391 | 0.5252 | 0.5144 | 0.4650 | 0.4566 |
| Support Vector Machine | 0.9115 | 0.9940 | 0.9121 | 0.9115 | 0.9114 | 0.8967 |

**Result interpretation:** on the documented test split, the Support Vector Machine achieved the highest value for all six reported metrics and is the overall best performer for this experiment. Logistic Regression and Decision Tree were close behind, KNN had moderate performance, and Gaussian Naive Bayes had the lowest scores. These results describe this dataset and split; they should not be treated as a guarantee of performance on other populations or data.

### 5.10 Recommended walkthrough for an evaluator

To explore the complete application:

1. Open the [live Obesity Level Classifier app](https://ml-assignment-2-obesity-2026ak05386-luyz2xk6xzjbhhkruunfrj.streamlit.app/).
2. Expand **About this project** to read the task, dataset, models and metrics summary.
3. In the sidebar, upload the repository's `test_data.csv` using **Upload test data (CSV)**. If needed, use **Download sample test_data.csv** first.
4. Review the **Overview** tab for row, feature and class counts, feature groups, class distribution and the data preview.
5. In the sidebar, select **Logistic Regression** and open **Selected Model** to inspect its metric cards, confusion matrix and classification report.
6. Repeat Step 5 for **Decision Tree**, **KNN**, **Naive Bayes** and **Support Vector Machine** to inspect each classifier.
7. Use **Normalize confusion matrix (row %)** to compare the confusion matrix in count and percentage form.
8. Open **Compare Models** to review all five models together, identify the best model by MCC, compare the metric columns and view the grouped bar chart.
9. Select **Download comparison table (CSV)** to save the comparison results.

This walkthrough demonstrates the upload workflow, model selection, dataset overview, per-model evaluation, error inspection and cross-model comparison.

### 5.11 Important application notes

- The app uses saved model files from the `model/` directory; it does not retrain models during evaluation.
- The uploaded CSV must include the expected feature columns and the target column `NObeyesdad`.
- The app checks for missing target columns, unknown target labels and missing required feature columns.
- The app's metric values are calculated from the uploaded data. The values in Section 5.9 are the previously recorded results for the assignment's held-out test set.
- The app uses the scikit-learn pipelines and label encoder saved with the project. Compatible dependency versions are important when loading serialized models.
- The app uses Streamlit tabs when supported by the installed Streamlit version. On older versions without tab support, the sections are displayed one after another.

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

## 8. Execution on Pragyoshala Lab

The assignment was executed in the **Pragyoshala Lab**. Screenshots showing the execution and the application are attached in the submitted PDF for reference.

---

## 9. Note on Authorship and GitHub Accounts

The commits in this repository appear under the GitHub username **Pps08**, while the repository itself is hosted under the username **PriyaP-py**. **Both accounts belong to the same person, [Your Full Name] ([Your BITS ID]), who is the sole author of this assignment.**

The two usernames appear because the Git installation on my computer was signed in to the Pps08 account, while the repository was created under PriyaP-py. No other person contributed to the code, the models, or the app. The commit history shows the work being done in stages: data preparation, model training, saved models and test data, the Streamlit app, requirements, and finally this README.
