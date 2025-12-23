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
    
    # Ambil credentials dari Environment Variables (disiapkan oleh GitHub Actions)
    user = os.environ.get('DAGSHUB_USERNAME')
    token = os.environ.get('DAGSHUB_TOKEN')
    
    if not user or not token:
        print("PERINGATAN: Username/Token tidak ditemukan di environment variables!")
        
        user = "hooddz1" 
        
    
    # Set Environment Variables untuk MLflow internal
    os.environ['MLFLOW_TRACKING_USERNAME'] = user
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    
    # Set URI
    DAGSHUB_REPO_NAME = "eksperimen_MSML_hudan-maulana-v2"
    uri = f"https://dagshub.com/{user}/{DAGSHUB_REPO_NAME}.mlflow"
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

    # ... (kode atas biarkan sama) ...

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

        # Logging Metrics ke DagsHub (Untuk Nilai/Grading)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_metric("mae", mae)

        # A. LOG KE DAGSHUB (Usaha Upload)
        print("Mencoba upload ke DagsHub...")
        mlflow.sklearn.log_model(model, "model")
        
        # B. SIMPAN LOKAL (Jalur Penyelamat untuk Docker)
        print("Menyimpan model ke folder lokal 'model_output'...")
        # Hapus folder lama jika ada (biar bersih)
        import shutil
        if os.path.exists("model_output"):
            shutil.rmtree("model_output")
            
        # Simpan fisik
        mlflow.sklearn.save_model(model, "model_output")
        print("Model berhasil disimpan secara lokal!")

if __name__ == "__main__":
    run()