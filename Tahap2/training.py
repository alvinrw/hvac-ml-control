import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

print("🚀 HFAC Neural Network Training")
print("="*60)

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\n📂 Loading dataset...")
df = pd.read_csv('hfac_greenhouse_dataset.csv')
print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

# ==========================================
# 2. PREPARE DATA
# ==========================================

print("\n🔧 Preparing data...")

# Features (Input sensors)
X = df[['temperature', 'humidity', 'light_intensity', 'motion']].values

# Targets (Output PWM actuators)
y = df[['fan_cooling_pwm', 'fan_circulation_pwm', 'water_pump_pwm', 'grow_light_pwm']].values

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalization
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

print(f"✅ Training set: {X_train.shape[0]} samples")
print(f"✅ Testing set: {X_test.shape[0]} samples")

# ==========================================
# 3. BUILD NEURAL NETWORK MODEL
# ==========================================

print("\n🧠 Building Neural Network...")

model = keras.Sequential([
    layers.Input(shape=(4,)),  # 4 input features
    
    # Hidden layers
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.1),
    
    # Output layer: 4 PWM values (0-100%)
    layers.Dense(4, activation='linear')
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print(model.summary())

# ==========================================
# 4. TRAINING
# ==========================================

print("\n🎯 Training model...")

# Callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-7
)

# Train
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ==========================================
# 5. EVALUATION
# ==========================================

print("\n📊 Evaluating model...")

# Predictions
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Clip predictions to valid PWM range (0-100)
y_pred = np.clip(y_pred, 0, 100)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*60)
print("🎯 MODEL PERFORMANCE")
print("="*60)
print(f"RMSE (Root Mean Square Error): {rmse:.4f}")
print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Accuracy (R² * 100): {r2*100:.2f}%")

# Per-actuator metrics
actuator_names = ['Fan Cooling', 'Fan Circulation', 'Water Pump', 'Grow Light']
print("\n📈 Per-Actuator Performance:")
print("-"*60)
for i, name in enumerate(actuator_names):
    r2_i = r2_score(y_test[:, i], y_pred[:, i])
    mae_i = mean_absolute_error(y_test[:, i], y_pred[:, i])
    print(f"{name:20s} | R²: {r2_i:.4f} | MAE: {mae_i:.4f}")

# ==========================================
# 6. VISUALIZATIONS
# ==========================================

print("\n📊 Creating visualizations...")

# Plot 1: Training History
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Training & Validation Loss', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MAE', fontsize=12)
plt.title('Training & Validation MAE', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("✅ Saved: training_history.png")

# Plot 2: Prediction vs Actual
plt.figure(figsize=(16, 10))

for i, name in enumerate(actuator_names):
    plt.subplot(2, 2, i+1)
    
    # Sample 200 points for clarity
    sample_size = min(200, len(y_test))
    indices = np.random.choice(len(y_test), sample_size, replace=False)
    
    plt.scatter(y_test[indices, i], y_pred[indices, i], alpha=0.5, s=30)
    plt.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual PWM (%)', fontsize=11)
    plt.ylabel('Predicted PWM (%)', fontsize=11)
    plt.title(f'{name} - Prediction vs Actual', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)

plt.tight_layout()
plt.savefig('prediction_vs_actual.png', dpi=300, bbox_inches='tight')
print("✅ Saved: prediction_vs_actual.png")

# ==========================================
# 7. SAVE MODEL & SCALERS
# ==========================================

print("\n💾 Saving model and scalers...")

model.save('hfac_model.h5')
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

print("✅ Model saved: hfac_model.h5")
print("✅ Scaler X saved: scaler_X.pkl")
print("✅ Scaler y saved: scaler_y.pkl")

print("\n" + "="*60)
print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print("\n📌 Next step: Run the GUI application to use the model!")
print("📌 Required files: hfac_model.h5, scaler_X.pkl, scaler_y.pkl")