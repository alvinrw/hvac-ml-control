import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Set random seed untuk reproducibility
np.random.seed(42)

# Jumlah data
n_samples = 10000

print("🌱 Generating HFAC Greenhouse Dataset...")

# ==========================================
# GENERATE SENSOR DATA
# ==========================================

# Temperature (15-40°C) - distribusi normal dengan mean 27°C
temperature = np.random.normal(27, 5, n_samples)
temperature = np.clip(temperature, 15, 40)

# Humidity (30-90% RH) - distribusi normal dengan mean 60%
humidity = np.random.normal(60, 15, n_samples)
humidity = np.clip(humidity, 30, 90)

# Light intensity (0-100%) - mix antara siang dan malam
# 60% data siang (high light), 40% malam (low light)
light_day = np.random.uniform(60, 100, int(n_samples * 0.6))
light_night = np.random.uniform(0, 40, int(n_samples * 0.4))
light_intensity = np.concatenate([light_day, light_night])
np.random.shuffle(light_intensity)

# Motion detected (0 atau 1) - 30% ada motion
motion = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

# ==========================================
# GENERATE ACTUATOR PWM (0-100%)
# Berdasarkan logika greenhouse control
# ==========================================

# Inisialisasi arrays
fan_cooling_pwm = np.zeros(n_samples)
fan_circulation_pwm = np.zeros(n_samples)
water_pump_pwm = np.zeros(n_samples)
grow_light_pwm = np.zeros(n_samples)

# Target ideal greenhouse:
TARGET_TEMP = 25  # °C
TARGET_HUMIDITY = 65  # %
TARGET_LIGHT = 70  # %

for i in range(n_samples):
    temp = temperature[i]
    hum = humidity[i]
    light = light_intensity[i]
    has_motion = motion[i]
    
    # ========== FAN COOLING PWM ==========
    # Nyala kalau suhu > target, makin panas makin kencang
    if temp > TARGET_TEMP:
        temp_diff = temp - TARGET_TEMP
        fan_cooling_pwm[i] = min(100, (temp_diff / 15) * 100)  # Scale ke 0-100%
    else:
        fan_cooling_pwm[i] = 0
    
    # ========== FAN CIRCULATION PWM ==========
    # Selalu jalan sedikit untuk sirkulasi, lebih kencang kalau ada motion
    base_circulation = 20  # Base 20% untuk sirkulasi minimal
    if has_motion:
        fan_circulation_pwm[i] = min(100, base_circulation + 30)
    else:
        fan_circulation_pwm[i] = base_circulation
    
    # Tambah sirkulasi kalau humidity tinggi
    if hum > 75:
        fan_circulation_pwm[i] = min(100, fan_circulation_pwm[i] + 20)
    
    # ========== WATER PUMP PWM ==========
    # Nyala kalau humidity < target
    if hum < TARGET_HUMIDITY:
        hum_diff = TARGET_HUMIDITY - hum
        water_pump_pwm[i] = min(100, (hum_diff / 35) * 100)  # Scale ke 0-100%
    else:
        water_pump_pwm[i] = 0
    
    # ========== GROW LIGHT PWM ==========
    # Nyala kalau intensitas cahaya < target
    if light < TARGET_LIGHT:
        light_diff = TARGET_LIGHT - light
        grow_light_pwm[i] = min(100, (light_diff / 70) * 100)  # Scale ke 0-100%
    else:
        grow_light_pwm[i] = 0

# ==========================================
# TAMBAH NOISE & VARIASI REALISTIS
# ==========================================

# Tambah noise kecil ke PWM (±5%) untuk simulasi kondisi real
fan_cooling_pwm += np.random.uniform(-5, 5, n_samples)
fan_circulation_pwm += np.random.uniform(-5, 5, n_samples)
water_pump_pwm += np.random.uniform(-5, 5, n_samples)
grow_light_pwm += np.random.uniform(-5, 5, n_samples)

# Clip ke range 0-100
fan_cooling_pwm = np.clip(fan_cooling_pwm, 0, 100)
fan_circulation_pwm = np.clip(fan_circulation_pwm, 0, 100)
water_pump_pwm = np.clip(water_pump_pwm, 0, 100)
grow_light_pwm = np.clip(grow_light_pwm, 0, 100)

# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame({
    'temperature': temperature,
    'humidity': humidity,
    'light_intensity': light_intensity,
    'motion': motion,
    'fan_cooling_pwm': fan_cooling_pwm,
    'fan_circulation_pwm': fan_circulation_pwm,
    'water_pump_pwm': water_pump_pwm,
    'grow_light_pwm': grow_light_pwm
})

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ==========================================
# SAVE TO CSV
# ==========================================

csv_filename = 'hfac_greenhouse_dataset.csv'
df.to_csv(csv_filename, index=False)

print(f"✅ Dataset generated successfully!")
print(f"📊 Total samples: {n_samples}")
print(f"💾 Saved to: {csv_filename}")
print("\n📈 Dataset Statistics:")
print("="*60)
print(df.describe())
print("\n🔍 Sample data (first 5 rows):")
print("="*60)
print(df.head())
print("\n✨ Dataset ready for training!")