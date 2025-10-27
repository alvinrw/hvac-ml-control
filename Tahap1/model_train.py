# model_train.py
# BAGIAN B: TRAINING DATASET
# Hasil: hfac_nn_model.h5
# ====================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score

# Konfigurasi Input/Output File
DATASET_PATH = 'dataset.csv'
MODEL_PATH = 'hfac_nn_model.h5'

# ------------------------------------------------------------
# Muat Data dan Normalisasi Ulang (untuk konsistensi)
# ------------------------------------------------------------
try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"ERROR: File {DATASET_PATH} tidak ditemukan. Jalankan data_prep.py terlebih dahulu.")
    exit()

# Definisikan kolom input dan output
INPUT_COLS = ['T_k', 'H_k', 'L_k', 'M_k', 'P_Kipas_k', 'P_Pompa_k']
OUTPUT_COLS = ['T_k+1', 'H_k+1']

X = df[INPUT_COLS].values
Y = df[OUTPUT_COLS].values

# Kita harus melakukan normalisasi di sini juga untuk mendapatkan X_scaled dan Y_scaled
# Namun, untuk efisiensi dan memastikan konsistensi dengan scaler yang disimpan,
# kita *asumsikan* scaler dari data_prep.py sudah diterapkan.
# Jika Anda ingin training dengan data scaled, Anda harus me-load scaler-nya dan apply,
# TAPI KARENA DI data_prep.py HANYA DISIMPAN SCALER-NYA BUKAN DATA SCALED,
# kita akan menerapkan kembali normalisasi disini sebelum split.

from sklearn.preprocessing import MinMaxScaler
scaler_X_temp = MinMaxScaler()
scaler_Y_temp = MinMaxScaler()
X_scaled = scaler_X_temp.fit_transform(X)
Y_scaled = scaler_Y_temp.fit_transform(Y)

# Split data: 80% training dan 20% validasi
X_train, X_val, Y_train, Y_val = train_test_split(
    X_scaled, Y_scaled, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# Bangun Model Neural Network
# ------------------------------------------------------------
INPUT_DIM = X_train.shape[1]  # 6 input
OUTPUT_DIM = Y_train.shape[1] # 2 output

model = Sequential([
    Dense(64, activation='relu', input_shape=(INPUT_DIM,)),
    Dense(32, activation='relu'),
    Dense(OUTPUT_DIM, activation='linear') # Output layer, 'linear' untuk regresi
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.summary()

# ------------------------------------------------------------
# Latih Model
# ------------------------------------------------------------
print("\nMemulai training model...")
history = model.fit(
    X_train, Y_train,
    epochs=100,            # Jumlah epoch disesuaikan
    batch_size=32,
    validation_data=(X_val, Y_val),
    verbose=0              # Set ke 1 untuk melihat progress
)
print("Training selesai.")

# ------------------------------------------------------------
# Evaluasi Model dan Akurasi
# ------------------------------------------------------------
loss, mae = model.evaluate(X_val, Y_val, verbose=0)
print(f"\n[EVALUASI AKURASI] Loss (MSE) pada data validasi: {loss:.4f}")
print(f"MAE (Mean Absolute Error) pada data validasi: {mae:.4f}")

# Hitung R2 Score (untuk mengukur goodness of fit)
Y_pred_val = model.predict(X_val, verbose=0)
r2 = r2_score(Y_val, Y_pred_val)
print(f"R-squared Score pada data validasi: {r2:.4f}")

# ------------------------------------------------------------
# Simpan Model
# ------------------------------------------------------------
model.save(MODEL_PATH)
print(f"\nModel berhasil disimpan ke: {MODEL_PATH}")