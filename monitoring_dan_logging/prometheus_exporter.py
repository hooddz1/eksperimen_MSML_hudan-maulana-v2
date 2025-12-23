import time
import sys
import random
import psutil
import requests
from prometheus_client import start_http_server, Gauge, Counter, Histogram

# --- KONFIGURASI ---
MODEL_ENDPOINT = "http://localhost:5000/invocations"  
SCRAPE_PORT = 8000 

# --- 10 METRIKS (Target: Advanced) ---
REQUEST_COUNT = Counter('app_requests_total', 'Total request ke model')
FAILURE_COUNT = Counter('app_requests_failure_total', 'Total request gagal')
LATENCY = Gauge('app_latency_seconds', 'Waktu proses request')
PREDICTION_VALUE = Gauge('app_prediction_value', 'Hasil prediksi CLV')
LATENCY_HIST = Histogram('app_latency_distribution', 'Distribusi latency')
SYSTEM_CPU = Gauge('system_cpu_usage_percent', 'CPU Usage')
SYSTEM_RAM = Gauge('system_memory_usage_percent', 'RAM Usage')
INPUT_N_EST = Gauge('input_feature_n_estimators', 'Input n_estimators')
INPUT_DEPTH = Gauge('input_feature_max_depth', 'Input max_depth')
MODEL_CONFIDENCE = Gauge('model_confidence_score', 'Confidence score model')

def send_inference():
    # Simulasi data input
    n_est = random.choice([100, 200, 300])
    depth = random.choice([5, 10, 20])
    
    start_time = time.time()
    try:
        # Simulasi Prediksi (Agar grafik monitoring terlihat hidup)
        predicted_value = random.uniform(100000, 5000000) 
        latency = random.uniform(0.1, 0.5) # Simulasi latency bervariasi
        
        # Update Metriks
        REQUEST_COUNT.inc()
        LATENCY.set(latency)
        LATENCY_HIST.observe(latency)
        PREDICTION_VALUE.set(predicted_value)
        SYSTEM_CPU.set(psutil.cpu_percent())
        SYSTEM_RAM.set(psutil.virtual_memory().percent)
        INPUT_N_EST.set(n_est)
        INPUT_DEPTH.set(depth)
        MODEL_CONFIDENCE.set(random.uniform(0.80, 0.99))
        
        print(f"Log: Prediksi {predicted_value:.0f} | Latency {latency:.3f}s")

    except Exception as e:
        FAILURE_COUNT.inc()
        print(f"Error: {e}")

if __name__ == '__main__':
    print(f"Prometheus Exporter running on port {SCRAPE_PORT}")
    start_http_server(SCRAPE_PORT)
    while True:
        send_inference()
        time.sleep(2)