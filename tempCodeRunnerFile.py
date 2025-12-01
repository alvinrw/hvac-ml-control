import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tensorflow as tf
import joblib
import os
from datetime import datetime

# ============================================
# MODEL SELECTION
# ============================================
SELECTED_MODEL = "NEURAL_NETWORK"  # Options: "NEURAL_NETWORK", "MPC", "QLEARNING"
# ============================================

class HFACControlSystemV2:
    def __init__(self, root):
        self.root = root
        self.root.title(f"HFAC Greenhouse Control System V2 - {SELECTED_MODEL}")
        self.root.geometry("1700x950")
        self.root.configure(bg='#ecf0f1')
        
        self.selected_model = SELECTED_MODEL
        
        # Load models
        self.load_models()
        
        # Variables
        self.current_conditions = {}
        self.target_conditions = {}
        self.actuator_outputs = {}
        
        self.setup_ui()
    
    def load_models(self):
        """Load all available models"""
        try:
            if self.selected_model == "NEURAL_NETWORK":
                print("Loading Neural Network...")
                self.nn_model = tf.keras.models.load_model(
                    'models/hfac_model.h5',
                    custom_objects={'mse': tf.keras.losses.MeanSquaredError}
                )
                self.scaler_X = joblib.load('models/scaler_X.pkl')
                self.scaler_y = joblib.load('models/scaler_y.pkl')
                print("[OK] Neural Network loaded")
                
            elif self.selected_model == "MPC":
                print("[OK] MPC initialized")
                self.nn_model = None
                
            elif self.selected_model == "QLEARNING":
                print("[OK] Q-Learning initialized")
                self.nn_model = None
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.root.destroy()
    
    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg='#2c3e50', height=70)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="HFAC Greenhouse Control System V2",
            font=('Segoe UI', 18, 'bold'),
            bg='#2c3e50',
            fg='white'
        ).pack(side='left', padx=30, pady=15)
        
        # Model indicator
        model_colors = {
            "NEURAL_NETWORK": "#3498db",
            "MPC": "#e74c3c",
            "QLEARNING": "#2ecc71"
        }
        model_names = {
            "NEURAL_NETWORK": "Neural Network",
            "MPC": "MPC",
            "QLEARNING": "Q-Learning"
        }
        
        tk.Label(
            header,
            text=f"Active: {model_names[self.selected_model]}",
            font=('Segoe UI', 11, 'bold'),
            bg=model_colors[self.selected_model],
            fg='white',
            padx=20,
            pady=8
        ).pack(side='right', padx=30)
        
        # Main container
        main = tk.Frame(self.root, bg='#ecf0f1')
        main.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Left panel - Controls
        left = tk.Frame(main, bg='white', relief='solid', bd=1, width=450)
        left.pack(side='left', fill='y', padx=(0, 10))
        left.pack_propagate(False)
        
        # Right panel - Visualization
        right = tk.Frame(main, bg='white', relief='solid', bd=1)
        right.pack(side='right', fill='both', expand=True)
        
        self.setup_controls(left)
        self.setup_visualization(right)
    
    def setup_controls(self, parent):
        """Setup control panel"""
        # Scrollable frame
        canvas = tk.Canvas(parent, bg='white', highlightthickness=0)
        scrollbar = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Title
        tk.Label(
            scrollable_frame,
            text="Control Panel",
            font=('Segoe UI', 14, 'bold'),
            bg='white',
            fg='#2c3e50'
        ).pack(pady=15)
        
        # Current conditions
        current_frame = tk.LabelFrame(
            scrollable_frame,
            text="Current Conditions",
            font=('Segoe UI', 11, 'bold'),
            bg='white',
            fg='#34495e',
            padx=20,
            pady=15
        )
        current_frame.pack(fill='x', padx=15, pady=10)
        
        self.create_sensor_inputs(current_frame, self.current_conditions, is_current=True)
        
        # Target conditions
        target_frame = tk.LabelFrame(
            scrollable_frame,
            text="Target Setpoints",
            font=('Segoe UI', 11, 'bold'),
            bg='white',
            fg='#34495e',
            padx=20,
            pady=15
        )
        target_frame.pack(fill='x', padx=15, pady=10)
        
        self.create_sensor_inputs(target_frame, self.target_conditions, is_current=False)
        
        # Buttons
        btn_frame = tk.Frame(scrollable_frame, bg='white')
        btn_frame.pack(fill='x', padx=15, pady=15)
        
        tk.Button(
            btn_frame,
            text="PREDICT",
            font=('Segoe UI', 11, 'bold'),
            bg='#3498db',
            fg='white',
            command=self.predict,
            height=2,
            cursor='hand2',
            relief='flat'
        ).pack(fill='x', pady=5)
        
        tk.Button(
            btn_frame,
            text="SIMULATE TRANSITION",
            font=('Segoe UI', 11, 'bold'),
            bg='#9b59b6',
            fg='white',
            command=self.simulate_transition,
            height=2,
            cursor='hand2',
            relief='flat'
        ).pack(fill='x', pady=5)
        
        tk.Button(
            btn_frame,
            text="COMPARE ALL MODELS",
            font=('Segoe UI', 11, 'bold'),
            bg='#e67e22',
            fg='white',
            command=self.compare_models,
            height=2,
            cursor='hand2',
            relief='flat'
        ).pack(fill='x', pady=5)
        
        tk.Button(
            btn_frame,
            text="SAVE RESULTS",
            font=('Segoe UI', 10, 'bold'),
            bg='#16a085',
            fg='white',
            command=self.save_results,
            height=2,
            cursor='hand2',
            relief='flat'
        ).pack(fill='x', pady=5)
        
        tk.Button(
            btn_frame,
            text="RESET",
            font=('Segoe UI', 10, 'bold'),
            bg='#95a5a6',
            fg='white',
            command=self.reset,
            height=2,
            cursor='hand2',
            relief='flat'
        ).pack(fill='x', pady=5)
        
        # Output frame
        output_frame = tk.LabelFrame(
            scrollable_frame,
            text="Actuator Outputs (PWM %)",
            font=('Segoe UI', 11, 'bold'),
            bg='white',
            fg='#34495e',
            padx=20,
            pady=15
        )
        output_frame.pack(fill='x', padx=15, pady=10)
        
        actuators = [
            ("Fan Cooling", "fan_cooling", "#e74c3c"),
            ("Fan Circulation", "fan_circulation", "#3498db"),
            ("Water Pump", "water_pump", "#1abc9c"),
            ("Grow Light", "grow_light", "#f39c12")
        ]
        
        for i, (label, key, color) in enumerate(actuators):
            tk.Label(
                output_frame,
                text=label,
                font=('Segoe UI', 10),
                bg='white',
                anchor='w'
            ).grid(row=i, column=0, sticky='w', pady=8)
            
            var = tk.StringVar(value="--")
            tk.Label(
                output_frame,
                textvariable=var,
                font=('Segoe UI', 14, 'bold'),
                bg='#ecf0f1',
                fg=color,
                width=10,
                anchor='center',
                relief='sunken',
                bd=2
            ("Humidity (%)", "humidity", 30, 90, 65),
            ("Light (%)", "light_intensity", 0, 100, 70),
            ("Motion", "motion", 0, 1, 0)
        ]
        
        for i, (label, key, min_val, max_val, default) in enumerate(sensors):
            # Label
            tk.Label(
                parent,
                text=label,
                font=('Segoe UI', 10),
                bg='white',
                anchor='w'
            ).grid(row=i, column=0, sticky='w', pady=10)
            
            if key == "motion":
                var = tk.IntVar(value=default)
                tk.Checkbutton(
                    parent,
                    variable=var,
                    bg='white',
                    font=('Segoe UI', 10)
                ).grid(row=i, column=1, sticky='w', padx=10)
            else:
                var = tk.DoubleVar(value=default)
                
                # Container frame
                container = tk.Frame(parent, bg='white')
                container.grid(row=i, column=1, sticky='ew', padx=10)
                
                # Slider
                slider = tk.Scale(
                    container,
                    from_=min_val,
                    to=max_val,
                    orient='horizontal',
                    variable=var,
                    bg='white',
                    length=200,
                    resolution=0.1,
                    showvalue=0
                )
                slider.pack(side='left', fill='x', expand=True)
                
                # Value display (bigger and clearer)
                value_label = tk.Label(
                    container,
                    textvariable=var,
                    font=('Segoe UI', 11, 'bold'),
                    bg='#3498db' if is_current else '#2ecc71',
                    fg='white',
                    width=8,
                    anchor='center',
                    relief='raised',
                    bd=2
                )
                value_label.pack(side='right', padx=(10, 0))
            
            storage[key] = var
        
        parent.columnconfigure(1, weight=1)
    
    def setup_visualization(self, parent):
        """Setup visualization panel"""
        tk.Label(
            parent,
            text="Visualization",
            font=('Segoe UI', 14, 'bold'),
            bg='white',
            fg='#2c3e50'
        ).pack(pady=15)
        
        self.fig = Figure(figsize=(13, 8), dpi=100, facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        self.plot_empty()
    
    def plot_empty(self):
        """Plot empty state"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(
            0.5, 0.5,
            "Select an action:\n\nPREDICT - Single model prediction\nSIMULATE - Transition simulation\nCOMPARE - Compare all models",
            ha='center', va='center',
            fontsize=12, color='#7f8c8d',
            transform=ax.transAxes
        )
        ax.axis('off')
        self.canvas.draw()
    
    # Prediction methods
    def predict_neural_network(self, temp, hum, light, motion):
        X = np.array([[temp, hum, light, motion]])
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.nn_model.predict(X_scaled, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return np.clip(y_pred[0], 0, 100)
    
    def predict_mpc(self, temp, hum, light, motion):
        fan_cooling = np.clip((temp - 25) * 10, 0, 100)
        water_pump = np.clip((65 - hum) * 5, 0, 100)
        grow_light = np.clip((70 - light) * 1.4, 0, 100)
        fan_circ = 20 + (30 if motion else 0) + (20 if hum > 75 else 0)
        fan_circ = np.clip(fan_circ, 0, 100)
        return np.array([fan_cooling, fan_circ, water_pump, grow_light])
    
    def predict_qlearning(self, temp, hum, light, motion):
        fan_cooling = np.clip((temp - 25) * 12, 0, 100)
        water_pump = np.clip((65 - hum) * 6, 0, 100)
        grow_light = np.clip((70 - light) * 1.5, 0, 100)
        fan_circ = 25 + (35 if motion else 0) + (25 if hum > 75 else 0)
        fan_circ = np.clip(fan_circ, 0, 100)
        return np.array([fan_cooling, fan_circ, water_pump, grow_light])
    
    def predict(self):
        """Predict using selected model"""
        try:
            temp = self.current_conditions['temperature'].get()
            hum = self.current_conditions['humidity'].get()
            light = self.current_conditions['light_intensity'].get()
            motion = self.current_conditions['motion'].get()
            
            if self.selected_model == "NEURAL_NETWORK":
                y_pred = self.predict_neural_network(temp, hum, light, motion)
            curr_hum = self.current_conditions['humidity'].get()
            curr_light = self.current_conditions['light_intensity'].get()
            curr_motion = self.current_conditions['motion'].get()
            
            tgt_temp = self.target_conditions['temperature'].get()
            tgt_hum = self.target_conditions['humidity'].get()
            tgt_light = self.target_conditions['light_intensity'].get()
            
            # Generate transition path
            n_steps = 50
            temps = np.linspace(curr_temp, tgt_temp, n_steps)
            hums = np.linspace(curr_hum, tgt_hum, n_steps)
            lights = np.linspace(curr_light, tgt_light, n_steps)
            
            # Predict for each step
            predictions = []
            for i in range(n_steps):
                if self.selected_model == "NEURAL_NETWORK":
                    pred = self.predict_neural_network(temps[i], hums[i], lights[i], curr_motion)
                elif self.selected_model == "MPC":
                    pred = self.predict_mpc(temps[i], hums[i], lights[i], curr_motion)
                elif self.selected_model == "QLEARNING":
                    pred = self.predict_qlearning(temps[i], hums[i], lights[i], curr_motion)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # Plot simulation
            self.plot_simulation(predictions, curr_temp, curr_hum, curr_light, tgt_temp, tgt_hum, tgt_light)
            
        except Exception as e:
            messagebox.showerror("Error", f"Simulation failed: {e}")
    
    def compare_models(self):
        """Compare all 3 models"""
        try:
            temp = self.current_conditions['temperature'].get()
            hum = self.current_conditions['humidity'].get()
            light = self.current_conditions['light_intensity'].get()
            motion = self.current_conditions['motion'].get()
            
            # Get predictions from all models
            try:
                nn_model = tf.keras.models.load_model(
                    'models/hfac_model.h5',
                    custom_objects={'mse': tf.keras.losses.MeanSquaredError}
                )
                scaler_X = joblib.load('models/scaler_X.pkl')
                scaler_y = joblib.load('models/scaler_y.pkl')
                
                X = np.array([[temp, hum, light, motion]])
                X_scaled = scaler_X.transform(X)
                y_pred_scaled = nn_model.predict(X_scaled, verbose=0)
                nn_pred = scaler_y.inverse_transform(y_pred_scaled)[0]
                nn_pred = np.clip(nn_pred, 0, 100)
                nn_available = True
            except:
                nn_pred = None
                nn_available = False
            
            mpc_pred = self.predict_mpc(temp, hum, light, motion)
            ql_pred = self.predict_qlearning(temp, hum, light, motion)
            
            # Plot comparison
            self.plot_comparison(nn_pred, mpc_pred, ql_pred, temp, hum, light, nn_available)
            
        except Exception as e:
            messagebox.showerror("Error", f"Comparison failed: {e}")
    
    def plot_single(self, pwm_values, temp, hum, light):
        """Plot single model prediction with enhanced visualization"""
        self.fig.clear()
        
        gs = self.fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)
        
        # Top left: Actuator outputs (horizontal bar)
        ax1 = self.fig.add_subplot(gs[0, :2])
        actuators = ['Fan Cooling', 'Fan Circulation', 'Water Pump', 'Grow Light']
        colors = ['#e74c3c', '#3498db', '#1abc9c', '#f39c12']
        
        bars = ax1.barh(actuators, pwm_values, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
        
        for i, (bar, val) in enumerate(zip(bars, pwm_values)):
            ax1.text(val + 3, i, f'{val:.1f}%', va='center', fontsize=11, fontweight='bold')
        
        ax1.set_xlabel('PWM (%)', fontsize=11, fontweight='bold')
        ax1.set_title(f'Actuator Outputs - {self.selected_model}', fontsize=12, fontweight='bold')
        ax1.set_xlim(0, 115)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Top right: Energy distribution pie chart
        ax2 = self.fig.add_subplot(gs[0, 2])
        ax2.pie(pwm_values, labels=actuators, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Energy Distribution', fontsize=11, fontweight='bold')
        
        # Bottom left: Current vs Target
        ax3 = self.fig.add_subplot(gs[1, 0])
        conditions = ['Temp', 'Humidity', 'Light']
        current_vals = [temp, hum, light]
        target_vals = [25, 65, 70]
        
        x = np.arange(len(conditions))
        width = 0.35
        
        ax3.bar(x - width/2, current_vals, width, label='Current', alpha=0.8, color='#3498db')
        ax3.bar(x + width/2, target_vals, width, label='Target', alpha=0.8, color='#2ecc71')
        
        ax3.set_ylabel('Value', fontsize=10, fontweight='bold')
        ax3.set_title('Conditions vs Target', fontsize=11, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(conditions, fontsize=9)
        ax3.legend(fontsize=9)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Bottom middle: Error analysis
        ax4 = self.fig.add_subplot(gs[1, 1])
        errors = [abs(c - t) for c, t in zip(current_vals, target_vals)]
        
        bars = ax4.bar(conditions, errors, color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('Absolute Error', fontsize=10, fontweight='bold')
        ax4.set_title('Tracking Error', fontsize=11, fontweight='bold')
        ax4.set_xticklabels(conditions, fontsize=9)
        ax4.grid(axis='y', alpha=0.3, linestyle='--')
        
        for i, (bar, err) in enumerate(zip(bars, errors)):
            ax4.text(i, err + 0.5, f'{err:.1f}', ha='center', fontsize=9, fontweight='bold')
        
        # Bottom right: Performance summary
        ax5 = self.fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        avg_error = np.mean(errors)
        avg_energy = np.mean(pwm_values)
        
        summary_text = f"""
PERFORMANCE SUMMARY

Average Error: {avg_error:.2f}
Average Energy: {avg_energy:.1f}%

Temperature: {temp:.1f}C
Humidity: {hum:.1f}%
Light: {light:.1f}%

Model: {self.selected_model}
"""
        ax5.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
        
        self.canvas.draw()
    
    def plot_simulation(self, predictions, c_temp, c_hum, c_light, t_temp, t_hum, t_light):
        """Plot transition simulation"""
        self.fig.clear()
        
        actuators = ['Fan Cooling', 'Fan Circulation', 'Water Pump', 'Grow Light']
        colors = ['#e74c3c', '#3498db', '#1abc9c', '#f39c12']
        
        for i in range(4):
            ax = self.fig.add_subplot(2, 2, i+1)
            
            steps = np.arange(len(predictions))
            pwm_values = predictions[:, i]
            
            ax.plot(steps, pwm_values, color=colors[i], linewidth=2.5, label=actuators[i])
            ax.fill_between(steps, 0, pwm_values, color=colors[i], alpha=0.2)
            
            # Mark start and end
            ax.scatter([0], [pwm_values[0]], color=colors[i], s=120, zorder=5, 
                      edgecolors='black', linewidths=2, marker='o', label='Start')
            ax.scatter([len(steps)-1], [pwm_values[-1]], color=colors[i], s=120, 
                      zorder=5, marker='s', edgecolors='black', linewidths=2, label='End')
            
            ax.set_xlabel('Steps', fontsize=10, fontweight='bold')
            ax.set_ylabel('PWM (%)', fontsize=10, fontweight='bold')
            ax.set_title(f'{actuators[i]}', fontsize=11, fontweight='bold')
            ax.set_ylim(-5, 105)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=8)
        
        self.fig.suptitle(
            f'Transition Simulation - {self.selected_model}\n'
            f'Temp: {c_temp:.1f}C -> {t_temp:.1f}C | '
            f'Humidity: {c_hum:.1f}% -> {t_hum:.1f}% | '
            f'Light: {c_light:.1f}% -> {t_light:.1f}%',
            fontsize=11,
            fontweight='bold'
        )
        
        self.fig.tight_layout(rect=[0, 0, 1, 0.94])
        self.canvas.draw()
    
    def plot_comparison(self, nn_pred, mpc_pred, ql_pred, temp, hum, light, nn_available):
        """Plot enhanced model comparison"""
        self.fig.clear()
        
        actuators = ['Fan Cooling', 'Fan Circulation', 'Water Pump', 'Grow Light']
        
        for i in range(4):
            ax = self.fig.add_subplot(2, 2, i+1)
            
            models = []
            values = []
            colors_list = []
            
            if nn_available and nn_pred is not None:
                models.append('Neural\nNetwork')
                values.append(nn_pred[i])
                colors_list.append('#3498db')
            
            models.append('MPC')
            values.append(mpc_pred[i])
            colors_list.append('#e74c3c')
            
            models.append('Q-Learning')
            values.append(ql_pred[i])
            colors_list.append('#2ecc71')
            
            bars = ax.bar(models, values, color=colors_list, alpha=0.85, edgecolor='black', linewidth=1.5)
            
            # Add value labels and percentage difference
            for j, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{val:.1f}%',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Show difference from mean
            mean_val = np.mean(values)
            for j, (bar, val) in enumerate(zip(bars, values)):
                diff = val - mean_val
                if abs(diff) > 1:
                    color = 'green' if diff > 0 else 'red'
                    ax.text(bar.get_x() + bar.get_width()/2., 5,
                           f'{diff:+.1f}',
                           ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')
            
            ax.set_ylabel('PWM (%)', fontsize=10, fontweight='bold')
            ax.set_title(actuators[i], fontsize=11, fontweight='bold')
            ax.set_ylim(0, 115)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        self.fig.suptitle(
            f'Model Comparison Analysis\n'
            f'Temperature: {temp:.1f}C | Humidity: {hum:.1f}% | Light: {light:.1f}%',
            fontsize=12,
            fontweight='bold'
        )
        
        self.fig.tight_layout(rect=[0, 0, 1, 0.94])
        self.canvas.draw()
    
    def save_results(self):
        """Save current visualization"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                initialfile=f"hfac_result_{timestamp}.png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            
            if filename:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Results saved to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")
    
    def reset(self):
        """Reset all fields"""
        defaults = {'temperature': 25.0, 'humidity': 65.0, 'light_intensity': 70.0, 'motion': 0}
        
    
    Features:
    - Single model prediction with detailed analysis
    - Transition simulation from current to target
    - Multi-model comparison
    - Performance metrics
    - Save results
    
    To change model, edit SELECTED_MODEL at line 14
    """
    root = tk.Tk()
    app = HFACControlSystemV2(root)
    root.mainloop()

if __name__ == "__main__":
    main()
