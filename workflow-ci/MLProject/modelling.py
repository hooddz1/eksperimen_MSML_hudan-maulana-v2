import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse
import os

# -------------------------------------------------------------------------
# 1. SETUP ARGUMEN (Agar parameter bisa diatur lewat file MLProject/Command Line)
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=10)
args = parser.parse_args()


DAGSHUB_USERNAME = "hooddz1"
DAGSHUB_REPO_NAME = "eksperimen_MSML_hudan-maulana-v2"
mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow")


mlflow.set_experiment("Eksperimen_CI_Automatic_Run")

def run():
    print("--- Memulai Training Otomatis (Workflow CI) ---")
    print(f"Menggunakan Parameter -> n_estimators: {args.n_estimators}, max_depth: {args.max_depth}")
    csv_path = "processed_clv.csv"

    if not os.path.exists(csv_path):
        print(f"Error Fatal: File '{csv_path}' tidak ditemukan di folder MLProject.")
        return

    df = pd.read_csv(csv_path)

    # 4. Preprocessing Sederhana
    # (Pastikan kolom target 'monetary' ada)
    X = df.drop(columns=['customer_id', 'monetary', 'order_purchase_timestamp'], errors='ignore')
    y = df['monetary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Training & Logging
    with mlflow.start_run():
        
        # Inisialisasi Model dengan parameter dari input Argumen
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        
        # Training
        model.fit(X_train, y_train)

        # Evaluasi
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)

        print(f"Hasil Evaluasi -> MAE: {mae:.2f}, MSE: {mse:.2f}")

        # LOGGING KE DAGSHUB
        # Log Parameter yang dipakai
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)

        # Log Metrics Hasil
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)

        # Log Model (Wajib untuk pembuatan Docker Image nanti)
        mlflow.sklearn.log_model(model, "model")

        print("Sukses! Model dan metrics tersimpan di DagsHub.")

if __name__ == "__main__":
    run()