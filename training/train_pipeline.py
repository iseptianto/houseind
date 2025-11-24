
# training/train_pipeline.py
import argparse, pandas as pd, numpy as np, joblib, os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

try:
    import mlflow, mlflow.sklearn  # optional
except Exception:
    mlflow = None

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to final.csv (with columns LB,LT,KM,KT,Provinsi,Kota/Kab,Type,Price)")
    p.add_argument("--out", default="models/trained/model_pipeline.pkl", help="Output path for model artifact")
    p.add_argument("--mlflow-uri", default="", help="MLflow tracking URI (empty to disable)")
    return p.parse_args()

def price_to_float(s):
    if pd.isna(s): return np.nan
    t = str(s).replace('.', '').replace(',', '')
    t = ''.join(ch for ch in t if ch.isdigit())
    return float(t) if t else np.nan

def room_to_int(s):
    if pd.isna(s): return np.nan
    t = str(s).strip()
    if t == '>10':
        return 11  # treat >10 as 11
    try:
        return int(t)
    except ValueError:
        return np.nan

if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(args.csv)

    needed = ["LB","LT","KM","KT","Kota/Kab","Provinsi","Type","Price"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}")

    df = df.copy()
    df["Price"] = df["Price"].apply(price_to_float)
    df["KM"] = df["KM"].apply(room_to_int)
    df["KT"] = df["KT"].apply(room_to_int)
    df = df.dropna(subset=["Price"])

    X = df[["LB","LT","KM","KT","Kota/Kab","Provinsi","Type"]]
    y = df["Price"]

    num_cols = ["LB","LT","KM","KT"]
    cat_cols = ["Kota/Kab","Provinsi","Type"]

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ],
        remainder="drop"
    )

    reg = GradientBoostingRegressor(random_state=42)
    pipe = Pipeline([("pre", pre), ("reg", reg)])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(Xtr, ytr)
    yhat = pipe.predict(Xte)

    mae = float(mean_absolute_error(yte, yhat))
    r2  = float(r2_score(yte, yhat))
    print(f"MAE={mae:,.2f}  R2={r2:.4f}  n={len(df)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out)

    if args.mlflow_uri and mlflow:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("house_price_model")
        with mlflow.start_run(run_name="pipeline_gbr"):
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(pipe, "model")
            mlflow.log_param("algo", "GradientBoostingRegressor")
            mlflow.log_param("features", "LB,LT,KM,KT,Kota/Kab,Provinsi,Type")
