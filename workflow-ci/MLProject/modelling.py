import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse
import os
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=10)
args = parser.parse_args()


try:
    print("Mengatur konfigurasi DagsHub...")
    # Pastikan variabel ini ada di os.environ (dari GitHub Secrets)
    if "DAGSHUB_USERNAME" not in os.environ or "DAGSHUB_TOKEN" not in os.environ:
        print("PERINGATAN: Username/Token tidak ditemukan di environment variables!")
    
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.environ.get('DAGSHUB_USERNAME', 'hooddz1')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ.get('DAGSHUB_TOKEN', '')
    
    # Set URI
    DAGSHUB_USERNAME = os.environ.get('DAGSHUB_USERNAME', 'hooddz1')
    DAGSHUB_REPO_NAME = "eksperimen_MSML_hudan-maulana-v2"
    uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"
    mlflow.set_tracking_uri(uri)
    print(f"Tracking URI diset ke: {uri}")
    
except Exception as e:
    print(f"Error setup auth: {e}")

# Nama eksperimen
mlflow.set_experiment("Eksperimen_CI_Automatic_Run")

def run():
    print("--- Memulai Training (Direct Python) ---")
    
    # 3. Load Data
    csv_path = "processed_clv.csv"
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' tidak ditemukan di {os.getcwd()}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # 4. Preprocessing
    X = df.drop(columns=['customer_id', 'monetary', 'order_purchase_timestamp'], errors='ignore')
    y = df['monetary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Training & Logging
    with mlflow.start_run() as run:
        print(f"Active Run ID: {run.info.run_id}")
        
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluasi
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        print(f"MAE: {mae}")

        # Logging Metrics
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_metric("mae", mae)

        # LOG MODEL (Ini langkah paling krusial yang tadi gagal)
        print("Sedang mengupload model ke DagsHub...")
        mlflow.sklearn.log_model(model, "model")
        print("Upload model selesai.")

if __name__ == "__main__":
    run()