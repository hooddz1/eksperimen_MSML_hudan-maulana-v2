import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import argparse
import os
import sys
import shutil

# 1. SETUP ARGUMEN (Wajib ada agar 'mlflow run' dengan parameter bisa jalan)
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=10)
args = parser.parse_args()

# 2. SETUP AUTH DAGSHUB
try:
    user = os.environ.get('DAGSHUB_USERNAME', 'hooddz1')
    token = os.environ.get('DAGSHUB_TOKEN', '')
    os.environ['MLFLOW_TRACKING_USERNAME'] = user
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    
    DAGSHUB_REPO_NAME = "eksperimen_MSML_hudan-maulana-v2"
    uri = f"https://dagshub.com/{user}/{DAGSHUB_REPO_NAME}.mlflow"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("Eksperimen_CI_Automatic_Run")
except Exception as e:
    print(f"Warning Auth: {e}")

def run():
    print("--- Memulai Training via MLflow Project ---")
    
    # 3. Load Data
    csv_path = "processed_clv.csv"
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' tidak ditemukan!")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    X = df.drop(columns=['customer_id', 'monetary', 'order_purchase_timestamp'], errors='ignore')
    y = df['monetary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Training
    with mlflow.start_run() as run:
        print(f"Active Run ID: {run.info.run_id}")
        
        model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth)
        model.fit(X_train, y_train)
        
        mae = mean_absolute_error(y_test, model.predict(X_test))
        print(f"MAE: {mae}")
        
        # Log ke DagsHub (Sesuai Syarat Modul)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_metric("mae", mae)
        
        print("Mencoba upload log ke DagsHub...")
        mlflow.sklearn.log_model(model, "model")

        # --- SIMPAN LOKAL (KUNCI SUKSES DOCKER) ---
        # Kita simpan di folder 'model_output' agar Docker tinggal ambil
        print("Menyimpan artifact lokal untuk Docker...")
        if os.path.exists("model_output"):
            shutil.rmtree("model_output")
        mlflow.sklearn.save_model(model, "model_output")
        print("Model lokal siap!")

if __name__ == "__main__":
    run()