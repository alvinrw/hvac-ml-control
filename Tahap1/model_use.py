# model_use.py
# BAGIAN C: PENGGUNAAN MODEL
# Hasil: Prediksi state berikutnya (T_k+1, H_k+1)
# ====================================================================

import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
# Import Metrics dan Losses yang digunakan saat kompilasi model
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError

# Konfigurasi Input File
MODEL_PATH = 'hfac_nn_model.h5'
SCALER_X_PATH = 'scaler_X.pkl'
SCALER_Y_PATH = 'scaler_Y.pkl'

# Definisikan custom_objects
# Meskipun 'mse' dan 'mae' adalah standar, Keras 3/TF modern sering
# membutuhkan definisi eksplisit saat memuat model H5 lama.
custom_objects = {
    'mse': MeanSquaredError(), # Menggunakan class dari losses
    'mae': MeanAbsoluteError() # Menggunakan class dari metrics
}


# ------------------------------------------------------------
# Muat Model dan Scaler
# ------------------------------------------------------------
try:
    # --- PERBAIKAN DILAKUKAN DI SINI ---
    model = load_model(MODEL_PATH, custom_objects=custom_objects)
    # ------------------------------------
    
    with open(SCALER_X_PATH, 'rb') as f:
        scaler_X = pickle.load(f)
    with open(SCALER_Y_PATH, 'rb') as f:
        scaler_Y = pickle.load(f)
except FileNotFoundError as e:
    print(f"ERROR: Satu atau lebih file tidak ditemukan. Pastikan Anda telah menjalankan data_prep.py dan model_train.py.\nDetail: {e}")
    exit()

print("Model dan Scaler berhasil dimuat.\n")


# ------------------------------------------------------------
# Input Data Baru (Contoh Kasus)
# ------------------------------------------------------------
# Kasus 1: Kondisi Panas, Kelembaban Sedang, Kipas ON, Pompa OFF
new_data_raw_1 = np.array([[
    32.0,   # T_k: Suhu tinggi
    55.0,   # H_k: Kelembaban sedang
    600.0,  # L_k: Cahaya terang
    1,      # M_k: Ada gerakan
    1,      # P_Kipas_k: Kipas ON (Aksi pendinginan)
    0       # P_Pompa_k: Pompa OFF (Tidak ada penambahan kelembaban)
]])

# Kasus 2: Kondisi Dingin, Kelembaban Rendah, Kipas OFF, Pompa ON
new_data_raw_2 = np.array([[
    21.0,   # T_k: Suhu rendah
    42.0,   # H_k: Kelembaban rendah
    100.0,  # L_k: Cahaya redup
    0,      # M_k: Tidak ada gerakan
    0,      # P_Kipas_k: Kipas OFF
    1       # P_Pompa_k: Pompa ON (Aksi penambahan kelembaban)
]])

# Kita gunakan Kasus 1 untuk demonstrasi
new_data_raw = new_data_raw_1
print("=" * 68)
print("Data Input Mentah (T_k, H_k, L_k, M_k, P_Kipas_k, P_Pompa_k):")
print(new_data_raw)
print("=" * 68)


# ------------------------------------------------------------
# Prediksi
# ------------------------------------------------------------

# 1. Normalisasi data input
X_new_scaled = scaler_X.transform(new_data_raw)
print("\nInput setelah Normalisasi (X_scaled):")
print(X_new_scaled)

# 2. Prediksi menggunakan model NN
Y_pred_scaled = model.predict(X_new_scaled, verbose=0)
print("\nOutput Prediksi Scaled (Y_pred_scaled):")
print(Y_pred_scaled)

# 3. Denormalisasi hasil prediksi
Y_pred_raw = scaler_Y.inverse_transform(Y_pred_scaled)

# ------------------------------------------------------------
# Tampilkan Hasil
# ------------------------------------------------------------
T_k_plus_1_pred = Y_pred_raw[0, 0]
H_k_plus_1_pred = Y_pred_raw[0, 1]

print("\n" + "=" * 68)
print("HASIL PREDIKSI STATE BERIKUTNYA (k+1)")
print("-" * 68)
print(f"Input: T_k={new_data_raw[0,0]:.1f}°C, H_k={new_data_raw[0,1]:.1f}%RH")
print(f"Aksi: Kipas={new_data_raw[0,4]}, Pompa={new_data_raw[0,5]}")
print(f"Prediksi Suhu Berikutnya (T_k+1): {T_k_plus_1_pred:.2f} °C")
print(f"Prediksi Kelembaban Berikutnya (H_k+1): {H_k_plus_1_pred:.2f} %RH")
print("=" * 68)