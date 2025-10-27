# gui_hfac_predictor.py
# BAGIAN D: GUI SEDERHANA UNTUK PENGGUNAAN MODEL (MODEL HFAC PREDICTOR)
# ====================================================================

import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
import tkinter as tk
from tkinter import messagebox

# Konfigurasi Input File
MODEL_PATH = 'hfac_nn_model.h5'
SCALER_X_PATH = 'scaler_X.pkl'
SCALER_Y_PATH = 'scaler_Y.pkl'

# Definisikan custom_objects untuk mengatasi error loading model
CUSTOM_OBJECTS = {
    'mse': MeanSquaredError(),
    'mae': MeanAbsoluteError()
}

# Variabel Global untuk Model dan Scaler
model = None
scaler_X = None
scaler_Y = None

def load_resources():
    """Memuat model dan scaler dari disk."""
    global model, scaler_X, scaler_Y
    try:
        model = load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS)
        with open(SCALER_X_PATH, 'rb') as f:
            scaler_X = pickle.load(f)
        with open(SCALER_Y_PATH, 'rb') as f:
            scaler_Y = pickle.load(f)
        return True
    except FileNotFoundError as e:
        messagebox.showerror("Error", f"Gagal memuat file. Pastikan {MODEL_PATH}, {SCALER_X_PATH}, dan {SCALER_Y_PATH} ada.\nDetail: {e}")
        return False
    except Exception as e:
        messagebox.showerror("Error", f"Gagal memuat model/scaler. Pastikan library TensorFlow dan Keras sudah terinstal dengan benar.\nDetail: {e}")
        return False

def predict_hfac():
    """Mengambil input, melakukan prediksi, dan menampilkan hasilnya."""
    if not model or not scaler_X or not scaler_Y:
        messagebox.showerror("Error", "Model belum dimuat. Coba restart aplikasi.")
        return

    # 1. Ambil Input dari Entry Fields
    input_values = []
    input_labels = ['T_k', 'H_k', 'L_k', 'M_k (0/1)', 'P_Kipas (0/1)', 'P_Pompa (0/1)']
    
    try:
        # Ambil nilai dan konversi ke float (atau int untuk aktuator/boolean)
        T_k = float(entries[0].get())
        H_k = float(entries[1].get())
        L_k = float(entries[2].get())
        M_k = int(entries[3].get())
        P_Kipas_k = int(entries[4].get())
        P_Pompa_k = int(entries[5].get())

        new_data_raw = np.array([[T_k, H_k, L_k, M_k, P_Kipas_k, P_Pompa_k]])

    except ValueError:
        messagebox.showerror("Error Input", "Pastikan semua input diisi dengan angka yang benar (M_k, P_Kipas, P_Pompa harus 0 atau 1).")
        return

    # 2. Normalisasi
    X_new_scaled = scaler_X.transform(new_data_raw)

    # 3. Prediksi
    Y_pred_scaled = model.predict(X_new_scaled, verbose=0)

    # 4. Denormalisasi
    Y_pred_raw = scaler_Y.inverse_transform(Y_pred_scaled)
    
    T_k_plus_1_pred = Y_pred_raw[0, 0]
    H_k_plus_1_pred = Y_pred_raw[0, 1]

    # 5. Tampilkan Hasil di GUI
    result_text = f"Prediksi Suhu Selanjutnya (T_k+1): {T_k_plus_1_pred:.2f} °C\n"
    result_text += f"Prediksi Kelembaban Selanjutnya (H_k+1): {H_k_plus_1_pred:.2f} %RH"
    
    result_label.config(text=result_text, fg="blue", font=("Arial", 11, "bold"))


# --- Inisialisasi GUI ---

root = tk.Tk()
root.title("HFAC State Predictor (NN)")

if not load_resources():
    root.destroy()
    exit()

# List untuk menyimpan Entry widgets
entries = []
input_labels_text = ['T_k (Suhu °C)', 'H_k (Kelembaban %RH)', 'L_k (Cahaya)', 'M_k (Gerakan 0/1)', 'P_Kipas (Aksi 0/1)', 'P_Pompa (Aksi 0/1)']
default_values = ['30.0', '60.0', '500.0', '1', '1', '0'] # Contoh default

# Membuat Input Fields
for i, label_text in enumerate(input_labels_text):
    # Label
    label = tk.Label(root, text=label_text, anchor='w')
    label.grid(row=i, column=0, padx=10, pady=5, sticky='w')
    
    # Entry Field
    entry = tk.Entry(root, width=20)
    entry.insert(0, default_values[i]) # Set nilai default
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)

# Tombol Prediksi
predict_button = tk.Button(root, text="PREDIKSI STATE BERIKUTNYA (k+1)", 
                           command=predict_hfac, 
                           bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
predict_button.grid(row=len(input_labels_text), column=0, columnspan=2, pady=15)

# Label Hasil
result_label = tk.Label(root, text="Masukkan data dan tekan tombol prediksi.", 
                        justify=tk.LEFT, fg="darkred")
result_label.grid(row=len(input_labels_text) + 1, column=0, columnspan=2, pady=10)

# Jalankan Loop Utama GUI
root.mainloop()