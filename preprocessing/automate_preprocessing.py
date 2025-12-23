import pandas as pd
import os
from datetime import timedelta

def run_preprocessing():
    print("--- Memulai Proses Otomatisasi Preprocessing ...")
    
    # 1. Definisikan Jalur Data
    raw_path = 'clv_dataset_raw' 
    output_dir = os.path.join('preprocessing', 'clv_dataset_preprocessing')
    
    # Pastikan folder output 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Membuat folder: {output_dir}")

    try:
        # 2. Load Data
        print("Sedang membaca dataset...")
        orders = pd.read_csv(os.path.join(raw_path, 'olist_orders_dataset.csv'))
        items = pd.read_csv(os.path.join(raw_path, 'olist_order_items_dataset.csv'))
        customers = pd.read_csv(os.path.join(raw_path, 'olist_customers_dataset.csv'))
        payments = pd.read_csv(os.path.join(raw_path, 'olist_order_payments_dataset.csv'))
        
        # 3. Merging Data
        df = orders.merge(items, on='order_id').merge(customers, on='customer_id')
        
        # Menghitung total pembayaran per order_id
        order_pay = payments.groupby('order_id').agg({'payment_value': 'sum'}).reset_index()
        df = df.merge(order_pay, on='order_id')
        
        # 4. Data Cleaning & Filtering
        df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        # Filter hanya status 'delivered' untuk validitas CLV
        df = df[df['order_status'] == 'delivered']
        
        # 5. Transformasi RFM (Recency, Frequency, Monetary)
        print("Melakukan transformasi RFM...")
        snapshot_date = df['order_purchase_timestamp'].max() + timedelta(days=1)
        
        rfm = df.groupby('customer_unique_id').agg({
            'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
            'order_id': 'nunique',
            'payment_value': 'sum'
        }).reset_index()
        
        # Penamaan kolom sesuai standar fitur lanjutan
        rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
        
        # 6. Handling Outlier (Syarat untuk Model Regresi yang Stabil)
        q_limit = rfm['monetary'].quantile(0.95)
        rfm_final = rfm[rfm['monetary'] <= q_limit].reset_index(drop=True)
        
        # 7. Export Dataset Hasil Preprocessing
        output_file = os.path.join(output_dir, 'processed_clv.csv')
        rfm_final.to_csv(output_file, index=False)
        
        print(f"\nSelesai! Dataset berhasil diproses.")
        print(f"Total Baris Awal: {len(rfm)}")
        print(f"Total Baris Akhir (Setelah Outlier): {len(rfm_final)}")
        print(f"Lokasi File: {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: File tidak ditemukan. Pastikan folder '{raw_path}' berisi CSV Olist. Detail: {e}")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    run_preprocessing()