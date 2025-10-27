import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tensorflow as tf
import joblib
import os 

class HFACControlSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("🌱 HFAC Greenhouse Control System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # --- File Paths (Relative paths - MUST be run from the folder containing the files) ---
        MODEL_PATH = 'hfac_model.h5'
        SCALER_X_PATH = 'scaler_X.pkl'
        SCALER_Y_PATH = 'scaler_y.pkl'

        # Load model and scalers with enhanced error handling
        try:
            # 1. Check for file existence 
            missing_files = []
            for path in [MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH]:
                if not os.path.exists(path):
                    missing_files.append(path)

            if missing_files:
                raise FileNotFoundError(
                    f"The following essential files are missing: {', '.join(missing_files)}. "
                    "Please ensure they are in the same directory as the script, or check your Current Working Directory (CWD)."
                )
            
            # 2. Load model (SOLUSI TERBARU UNTUK 'mean_squared_error' error)
            self.model = tf.keras.models.load_model(
                MODEL_PATH, 
                # Mengganti metrik dengan losses untuk kompatibilitas API TensorFlow terbaru
                custom_objects={'mse': tf.keras.losses.MeanSquaredError}
            )
            
            # 3. Load scalers
            self.scaler_X = joblib.load(SCALER_X_PATH)
            self.scaler_y = joblib.load(SCALER_Y_PATH)
            
            print("✅ Model and scalers loaded successfully!")
        
        except FileNotFoundError as fnf_e:
            messagebox.showerror("CRITICAL ERROR: Files Not Found", str(fnf_e))
            self.root.destroy()
            return
        except Exception as e:
            messagebox.showerror("CRITICAL ERROR: Load Failure", f"Failed to load model or scalers. Details: {e}")
            self.root.destroy()
            return
        
        # Variables to store data
        self.current_conditions = {}
        self.target_conditions = {}
        self.actuator_outputs = {}
        
        self.setup_ui()
    
    def setup_ui(self):
        # ==========================================
        # HEADER
        # ==========================================
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill='x', side='top')
        
        title_label = tk.Label(
            header_frame, 
            text="🌱 HFAC Greenhouse Control System", 
            font=('Arial', 24, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=20)
        
        # ==========================================
        # MAIN CONTAINER
        # ==========================================
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Left Panel: Input & Control
        left_panel = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        left_panel.pack(side='left', fill='both', expand=False, padx=(0, 10))
        
        # Right Panel: Visualization
        right_panel = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        right_panel.pack(side='right', fill='both', expand=True)
        
        # ==========================================
        # LEFT PANEL CONTENT
        # ==========================================
        
        # === CURRENT CONDITIONS ===
        current_frame = tk.LabelFrame(
            left_panel, 
            text="📊 Current Conditions", 
            font=('Arial', 14, 'bold'),
            bg='white',
            fg='#34495e',
            padx=15,
            pady=15
        )
        current_frame.pack(fill='x', padx=15, pady=10)
        
        self.create_input_fields(current_frame, is_current=True)
        
        # === TARGET CONDITIONS ===
        target_frame = tk.LabelFrame(
            left_panel, 
            text="🎯 Target Conditions", 
            font=('Arial', 14, 'bold'),
            bg='white',
            fg='#34495e',
            padx=15,
            pady=15
        )
        target_frame.pack(fill='x', padx=15, pady=10)
        
        self.create_input_fields(target_frame, is_current=False)
        
        # === CONTROL BUTTONS ===
        button_frame = tk.Frame(left_panel, bg='white')
        button_frame.pack(fill='x', padx=15, pady=15)
        
        predict_btn = tk.Button(
            button_frame,
            text="🔮 Predict Actuators",
            font=('Arial', 12, 'bold'),
            bg='#3498db',
            fg='white',
            command=self.predict_actuators,
            height=2,
            cursor='hand2'
        )
        predict_btn.pack(fill='x', pady=5)
        
        simulate_btn = tk.Button(
            button_frame,
            text="📈 Simulate Path",
            font=('Arial', 12, 'bold'),
            bg='#2ecc71',
            fg='white',
            command=self.simulate_path,
            height=2,
            cursor='hand2'
        )
        simulate_btn.pack(fill='x', pady=5)
        
        reset_btn = tk.Button(
            button_frame,
            text="🔄 Reset",
            font=('Arial', 12, 'bold'),
            bg='#95a5a6',
            fg='white',
            command=self.reset_fields,
            height=2,
            cursor='hand2'
        )
        reset_btn.pack(fill='x', pady=5)
        
        # === ACTUATOR OUTPUT DISPLAY ===
        output_frame = tk.LabelFrame(
            left_panel, 
            text="⚡ Actuator Outputs (PWM %)", 
            font=('Arial', 14, 'bold'),
            bg='white',
            fg='#34495e',
            padx=15,
            pady=15
        )
        output_frame.pack(fill='x', padx=15, pady=10)
        
        self.create_output_display(output_frame)
        
        # ==========================================
        # RIGHT PANEL CONTENT - VISUALIZATION
        # ==========================================
        
        viz_title = tk.Label(
            right_panel,
            text="📊 Actuator Control Visualization",
            font=('Arial', 16, 'bold'),
            bg='white',
            fg='#2c3e50'
        )
        viz_title.pack(pady=15)
        
        # Matplotlib Figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=15, pady=15)
        
        # Initial empty plot
        self.plot_empty_state()
    
    def create_input_fields(self, parent, is_current=True):
        """Create input fields for sensor values"""
        
        fields = [
            ("🌡️ Temperature (°C):", "temperature", 15, 40, 25),
            ("💧 Humidity (%RH):", "humidity", 30, 90, 65),
            ("☀️ Light Intensity (%):", "light_intensity", 0, 100, 70),
            ("👤 Motion Detected:", "motion", 0, 1, 0)
        ]
        
        storage = self.current_conditions if is_current else self.target_conditions
        
        for i, (label, key, min_val, max_val, default) in enumerate(fields):
            # Label
            lbl = tk.Label(
                parent, 
                text=label, 
                font=('Arial', 11),
                bg='white',
                anchor='w'
            )
            lbl.grid(row=i, column=0, sticky='w', pady=8)
            
            if key == "motion":
                # Motion as checkbox
                var = tk.IntVar(value=default)
                chk = tk.Checkbutton(
                    parent,
                    variable=var,
                    bg='white',
                    font=('Arial', 11)
                )
                chk.grid(row=i, column=1, sticky='ew', padx=(10, 0))
                storage[key] = var
            else:
                # Numeric input with Scale
                frame = tk.Frame(parent, bg='white')
                frame.grid(row=i, column=1, sticky='ew', padx=(10, 0))
                
                var = tk.DoubleVar(value=default)
                
                scale = tk.Scale(
                    frame,
                    from_=min_val,
                    to=max_val,
                    orient='horizontal',
                    variable=var,
                    bg='white',
                    font=('Arial', 10),
                    length=200,
                    resolution=0.1 if key != "motion" else 1
                )
                scale.pack(side='left', fill='x', expand=True)
                
                entry = tk.Entry(
                    frame,
                    textvariable=var,
                    font=('Arial', 10),
                    width=8,
                    justify='center'
                )
                entry.pack(side='right', padx=(5, 0))
                
                storage[key] = var
        
        parent.columnconfigure(1, weight=1)
    
    def create_output_display(self, parent):
        """Create display for actuator PWM outputs"""
        
        actuators = [
            ("🌬️ Fan Cooling:", "fan_cooling"),
            ("💨 Fan Circulation:", "fan_circulation"),
            ("💦 Water Pump:", "water_pump"),
            ("💡 Grow Light:", "grow_light")
        ]
        
        for i, (label, key) in enumerate(actuators):
            lbl = tk.Label(
                parent,
                text=label,
                font=('Arial', 11, 'bold'),
                bg='white',
                anchor='w'
            )
            lbl.grid(row=i, column=0, sticky='w', pady=8)
            
            var = tk.StringVar(value="-- %")
            value_lbl = tk.Label(
                parent,
                textvariable=var,
                font=('Arial', 12, 'bold'),
                bg='#ecf0f1',
                fg='#e74c3c',
                anchor='center',
                relief='sunken',
                bd=2,
                width=10
            )
            value_lbl.grid(row=i, column=1, sticky='ew', padx=(10, 0))
            
            self.actuator_outputs[key] = var
        
        parent.columnconfigure(1, weight=1)
    
    def predict_actuators(self):
        """Predict actuator PWM values from current conditions"""
        try:
            # Get current sensor values
            temp = self.current_conditions['temperature'].get()
            hum = self.current_conditions['humidity'].get()
            light = self.current_conditions['light_intensity'].get()
            motion = self.current_conditions['motion'].get()
            
            # Prepare input
            X_input = np.array([[temp, hum, light, motion]])
            
            # Validation check
            if X_input.ndim != 2 or X_input.shape[1] != 4:
                raise ValueError("Input data shape is incorrect.")

            X_scaled = self.scaler_X.transform(X_input)
            
            # Predict
            y_pred_scaled = self.model.predict(X_scaled, verbose=0)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
            y_pred = np.clip(y_pred[0], 0, 100) # Clamp PWM values between 0 and 100
            
            # Update display
            actuator_keys = ['fan_cooling', 'fan_circulation', 'water_pump', 'grow_light']
            for i, key in enumerate(actuator_keys):
                self.actuator_outputs[key].set(f"{y_pred[i]:.1f} %")
            
            # Plot single prediction
            self.plot_single_prediction(y_pred)
            
            messagebox.showinfo("Success", "✅ Actuator prediction completed!")
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Prediction failed: {str(e)}")
    
    def simulate_path(self):
        """Simulate transition from current to target conditions"""
        try:
            # Get current and target values
            current = {
                'temp': self.current_conditions['temperature'].get(),
                'hum': self.current_conditions['humidity'].get(),
                'light': self.current_conditions['light_intensity'].get(),
                'motion': self.current_conditions['motion'].get()
            }
            
            target = {
                'temp': self.target_conditions['temperature'].get(),
                'hum': self.target_conditions['humidity'].get(),
                'light': self.target_conditions['light_intensity'].get(),
                'motion': self.target_conditions['motion'].get()
            }
            
            # Generate path (50 steps)
            n_steps = 50
            temps = np.linspace(current['temp'], target['temp'], n_steps)
            hums = np.linspace(current['hum'], target['hum'], n_steps)
            lights = np.linspace(current['light'], target['light'], n_steps)
            
            # Prepare input array for all steps
            X_path = np.column_stack((temps, hums, lights, np.full(n_steps, current['motion'])))

            # Scale and Predict
            X_scaled = self.scaler_X.transform(X_path)
            y_pred_scaled = self.model.predict(X_scaled, verbose=0)
            predictions = self.scaler_y.inverse_transform(y_pred_scaled)
            predictions = np.clip(predictions, 0, 100) # Clamp PWM values

            # Plot simulation
            self.plot_simulation(predictions, current, target)
            
            messagebox.showinfo("Success", "✅ Path simulation completed!")
            
        except Exception as e:
            messagebox.showerror("Simulation Error", f"Simulation failed: {str(e)}")
    
    def plot_empty_state(self):
        """Plot empty state message"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(
            0.5, 0.5, 
            "👆 Click 'Predict' or 'Simulate' to see visualization",
            ha='center', va='center',
            fontsize=14, color='#7f8c8d',
            transform=ax.transAxes
        )
        ax.axis('off')
        self.canvas.draw()
    
    def plot_single_prediction(self, pwm_values):
        """Plot bar chart of single prediction"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        actuators = ['Fan\nCooling', 'Fan\nCirculation', 'Water\nPump', 'Grow\nLight']
        colors = ['#e74c3c', '#3498db', '#1abc9c', '#f39c12']
        
        bars = ax.bar(actuators, pwm_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, pwm_values)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., height + 2,
                f'{val:.1f}%',
                ha='center', va='bottom',
                fontsize=12, fontweight='bold'
            )
        
        ax.set_ylabel('PWM (%)', fontsize=12, fontweight='bold')
        ax.set_title('🎯 Current Actuator Outputs', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def plot_simulation(self, predictions, current, target):
        """Plot simulation path with multiple subplots"""
        self.fig.clear()
        
        actuators = ['Fan Cooling', 'Fan Circulation', 'Water Pump', 'Grow Light']
        colors = ['#e74c3c', '#3498db', '#1abc9c', '#f39c12']
        
        # Create 2x2 subplot
        for i in range(4):
            ax = self.fig.add_subplot(2, 2, i+1)
            
            steps = np.arange(len(predictions))
            pwm_values = predictions[:, i]
            
            # Plot line
            ax.plot(steps, pwm_values, color=colors[i], linewidth=2.5, label=actuators[i])
            ax.fill_between(steps, 0, pwm_values, color=colors[i], alpha=0.2)
            
            # Mark start and end
            ax.scatter([0], [pwm_values[0]], color=colors[i], s=100, zorder=5, 
                        edgecolors='black', linewidths=2, label='Start')
            ax.scatter([len(steps)-1], [pwm_values[-1]], color=colors[i], s=100, 
                        zorder=5, marker='s', edgecolors='black', linewidths=2, label='End')
            
            ax.set_xlabel('Steps', fontsize=10)
            ax.set_ylabel('PWM (%)', fontsize=10)
            ax.set_title(f'{actuators[i]}', fontsize=11, fontweight='bold')
            ax.set_ylim(-5, 105)
            ax.grid(True, alpha=0.3, linestyle='--')
            
        # Add overall title
        self.fig.suptitle(
            f'📈 Path Simulation: Current → Target\n'
            f'Temp: {current["temp"]:.1f}°C → {target["temp"]:.1f}°C | '
            f'Humidity: {current["hum"]:.1f}% → {target["hum"]:.1f}% | '
            f'Light: {current["light"]:.1f}% → {target["light"]:.1f}%',
            fontsize=11,
            fontweight='bold',
            y=0.98
        )
        
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
        self.canvas.draw()
    
    def reset_fields(self):
        """Reset all input fields to default values"""
        defaults = {
            'temperature': 25.0,
            'humidity': 65.0,
            'light_intensity': 70.0,
            'motion': 0
        }
        
        for key, val in defaults.items():
            self.current_conditions[key].set(val)
            self.target_conditions[key].set(val)
        
        # Reset outputs
        for key in self.actuator_outputs:
            self.actuator_outputs[key].set("-- %")
        
        # Reset plot
        self.plot_empty_state()
        
        messagebox.showinfo("Reset", "✅ All fields reset to default values!")

def main():
    # Pastikan Anda telah menginstal semua library ini:
    # pip install tensorflow numpy matplotlib scikit-learn joblib
    
    root = tk.Tk()
    app = HFACControlSystem(root)
    root.mainloop()

if __name__ == "__main__":
    main()