import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import joblib 


DAGSHUB_USERNAME = "hooddz1"  
DAGSHUB_TOKEN = "38a174de66a7e1a21f4adf7062d4cfed09c67231"        
DAGSHUB_REPO_NAME = "eksperimen_MSML_hudan-maulana-v2" 

# Set Environment Variables untuk Autentikasi
os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN

# Set Tracking URI ke DagsHub

mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow")

mlflow.set_experiment("Eksperimen_CLV_Hudan_Advanced")
# ----------------------------------------------------

def run_tuning_manual():
    print(f"--- Memulai Hyperparameter Tuning ke DagsHub ({DAGSHUB_REPO_NAME}) ---")
    
    # Load Data (Sama seperti sebelumnya)
    data_path = 'preprocessing/clv_dataset_preprocessing/processed_clv.csv'
    if not os.path.exists(data_path):
        print(f"Error: File '{data_path}' tidak ditemukan.")
        return

    df = pd.read_csv(data_path)
    X = df.drop(columns=['customer_id', 'monetary', 'order_purchase_timestamp'], errors='ignore')
    y = df['monetary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Mulai Run
    with mlflow.start_run(run_name="Tuning_Manual_DagsHub_Advanced"):
        
        # A. HYPERPARAMETER TUNING
        print("Sedang melakukan Grid Search...")
        rf = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
        }
        
        grid = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        
        # B. EVALUASI METRICS
        predictions = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        print(f"Metrics -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
        
        # C. LOGGING MANUAL (Syarat Advanced: Tanpa Autolog)
        mlflow.log_params(grid.best_params_)
        
        mlflow.log_metric("mean_absolute_error", mae)
        mlflow.log_metric("mean_squared_error", mse)
        mlflow.log_metric("root_mean_squared_error", rmse)
        mlflow.log_metric("r2_score", r2)
        
        # Log Model
        mlflow.sklearn.log_model(best_model, "model_best_tuning")
        
        # D. LOGGING 3 ARTIFACTS TAMBAHAN (Syarat Minimal 2)
        print("Mengupload visualisasi artifacts ke DagsHub...")
        
        # Artifact 1: Feature Importance
        plt.figure(figsize=(10, 6))
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = X.columns
        sns.barplot(x=importances[indices], y=features[indices], hue=features[indices], palette="viridis", legend=False)
        plt.title("Feature Importance - Random Forest")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()
        
        # Artifact 2: Actual vs Predicted
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, predictions, alpha=0.5, color='blue')
        max_val = max(max(y_test), max(predictions))
        plt.plot([0, max_val], [0, max_val], color='red', linestyle='--')
        plt.title("Actual vs Predicted CLV")
        plt.savefig("actual_vs_predicted.png")
        mlflow.log_artifact("actual_vs_predicted.png")
        plt.close()

        # Artifact 3: Residuals Plot
        residuals = y_test - predictions
        plt.figure(figsize=(10, 6))
        plt.scatter(predictions, residuals, alpha=0.5, color='purple')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title("Residuals Plot")
        plt.savefig("residuals_plot.png")
        mlflow.log_artifact("residuals_plot.png")
        plt.close()
        
        print("Sukses! Cek DagsHub kamu sekarang.")

if __name__ == "__main__":
    run_tuning_manual()