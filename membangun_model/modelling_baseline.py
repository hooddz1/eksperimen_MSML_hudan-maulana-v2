import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# 1. Konfigurasi MLflow Lokal
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Eksperimen_CLV_Hudan_v2")

def run_baseline():
    print("--- Memulai Training Baseline Model (dengan Autolog) ---")
    
    # 2. Load Data
    data_path = 'preprocessing/clv_dataset_preprocessing/processed_clv.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: File '{data_path}' tidak ditemukan.")
        print("Tip: Jalankan 'python membangun_model/preprocess.py' terlebih dahulu.")
        return

    df = pd.read_csv(data_path)
    
    # 3. Persiapan Data
    X = df.drop(columns=['customer_id', 'monetary', 'order_purchase_timestamp'], errors='ignore')
    y = df['monetary']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Aktifkan Autologging MLflow
    mlflow.sklearn.autolog()

    # 5. Training Baseline
    with mlflow.start_run(run_name="Baseline_Model"):
        
        # Inisialisasi Model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        # Saat .fit() dijalankan, Autolog akan bekerja otomatis mencatat semuanya ke MLflow
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        
        print(f"Baseline MAE: {mae}")
        print(f"Baseline MSE: {mse}")
        
       
        mlflow.log_param("model_type", "RandomForest_Baseline") # Info tambahan yang berguna
        
        
        print("Sukses! Baseline model telah dilatih. Cek MLflow untuk melihat parameter lengkap (Autolog).")

if __name__ == "__main__":
    run_baseline()