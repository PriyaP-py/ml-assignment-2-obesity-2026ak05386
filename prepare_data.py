"""
prepare_data.py
Reads the Obesity Levels CSV you downloaded from Kaggle/UCI and saves obesity.csv
with the target column named 'NObeyesdad'.

Put the downloaded CSV in this folder (file name must contain 'obesity'), then run:
    python prepare_data.py
"""
import glob
import pandas as pd

TARGET = "NObeyesdad"

files = [f for f in glob.glob("*.csv")
         if "obesity" in f.lower() and f.lower() != "obesity.csv"]
if not files:
    raise SystemExit("No CSV with 'obesity' in its name found in this folder. "
                     "Download the dataset and rename/upload it here.")

print("Reading:", files[0])
df = pd.read_csv(files[0], encoding="utf-8-sig")
df.columns = [c.strip() for c in df.columns]

# Some Kaggle copies rename the target column; detect it and standardise the name
if TARGET not in df.columns:
    options = ["NObesity", "Obesity", "Obesity_level", "ObesityLevel", "obesity_level", "Label", "Class"]
    found = next((c for c in options if c in df.columns), df.columns[-1])
    print(f"Target column '{TARGET}' not found; using '{found}' instead.")
    df = df.rename(columns={found: TARGET})

df[TARGET] = df[TARGET].astype(str).str.strip()
df.to_csv("obesity.csv", index=False)

print("Saved obesity.csv")
print("Shape:", df.shape, "-> features:", df.shape[1] - 1)
print(df[TARGET].value_counts())