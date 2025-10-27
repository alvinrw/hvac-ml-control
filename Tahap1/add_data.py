# data_prep.py
# BAGIAN A: SIMULASI & PERSIAPAN DATASET
# Hasil: dataset.csv, scaler_X.pkl, scaler_Y.pkl
# ====================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# Konfigurasi Output File
DATASET_PATH = 'dataset.csv'
SCALER_X_PATH = 'scaler_X.pkl'
SCALER_Y_PATH = 'scaler_Y.pkl'

# ------------------------------------------------------------
# Membuat data simulasi (6 input: 4 sensor + 2 aktuator, 2 output target)
# ------------------------------------------------------------
np.random.seed(42)
N = 2000  # Jumlah sampel ditingkatkan
print(f"Membuat {N} data simulasi...")

data = {
    # Input sensor dan kondisi lingkungan (State pada waktu k)
    'T_k': np.random.uniform(20, 35, N),     # Suhu (°C)
    'H_k': np.random.uniform(40, 70, N),     # Kelembapan (%RH)
    'L_k': np.random.uniform(50, 800, N),    # Intensitas cahaya (LDR)
    'M_k': np.random.randint(0, 2, N),       # Deteksi gerakan (0/1)

    # Input aktuator (Aksi kontrol pada waktu k)
    'P_Kipas_k': np.random.randint(0, 2, N), # Kipas DC (0/1)
    'P_Pompa_k': np.random.randint(0, 2, N)  # Pompa air (0/1)
}

df = pd.DataFrame(data)

# ------------------------------------------------------------
# Simulasi output (State pada waktu k+1) - Model Sederhana Non-Linear
# ------------------------------------------------------------
# T_k+1: Dipengaruhi T_k, Kipas (pendingin), Gerakan (panas), dan noise
df['T_k+1'] = (
    0.95 * df['T_k'] +
    np.where(df['P_Kipas_k'] == 1, -1.5, 0.5) + # Kipas mendinginkan lebih kuat
    0.3 * df['M_k'] +
    np.random.normal(0, 0.15, N)
)

# H_k+1: Dipengaruhi H_k, Pompa (menaikkan kelembaban), Suhu (menurunkan kelembaban), dan noise
df['H_k+1'] = (
    0.98 * df['H_k'] +
    np.where(df['P_Pompa_k'] == 1, 2.0, -0.6) - # Pompa menaikkan kelembaban lebih kuat
    0.15 * df['T_k'] +
    np.random.normal(0, 0.3, N)
)

# Simpan data mentah ke CSV
df.to_csv(DATASET_PATH, index=False)
print(f"Data simulasi mentah berhasil disimpan ke: {DATASET_PATH}\n")


# ------------------------------------------------------------
# Normalisasi dan Penyimpanan Scaler
# ------------------------------------------------------------
# Menentukan input (X) dan target (Y)
X = df[['T_k', 'H_k', 'L_k', 'M_k', 'P_Kipas_k', 'P_Pompa_k']].values
Y = df[['T_k+1', 'H_k+1']].values

# Normalisasi data
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

# Simpan scaler agar dapat digunakan kembali saat prediksi
with open(SCALER_X_PATH, 'wb') as f:
    pickle.dump(scaler_X, f)
    
with open(SCALER_Y_PATH, 'wb') as f:
    pickle.dump(scaler_Y, f)

print(f"Scaler X berhasil disimpan ke: {SCALER_X_PATH}")
print(f"Scaler Y berhasil disimpan ke: {SCALER_Y_PATH}")

# ------------------------------------------------------------
# Tampilkan contoh data
# ------------------------------------------------------------
print("\n" + "=" * 68)
print(f"Data berhasil dipersiapkan. Total sampel: {N} | Input (X) shape: {X.shape} | Output (Y) shape: {Y.shape}")
print("=" * 68)

print("\nContoh Input (X_scaled) 3 baris pertama:")
print(X_scaled[:3])

print("\nContoh Output (Y_scaled) 3 baris pertama:")
print(Y_scaled[:3])