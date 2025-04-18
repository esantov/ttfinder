import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from io import BytesIO

# ----- Simple 5PL logistic function for fitting -----
def logistic_5pl(t, a, d, c, b, g):
    return d + (a - d) / (1 + (t / c)**b)**g

# ----- Function to calculate the threshold time (Tt) -----
def calculate_threshold_time(threshold, popt):
    a, d, c, b, g = popt
    if not (d < threshold < a):
        raise ValueError(f"Threshold must be within the range of the 5PL function (d={d}, a={a}).")
    return c * (((a - d) / (threshold - d)) ** (1 / g) - 1) ** (1 / b)

# ----- APP LOGIC -----
st.title("📈 5PL Curve Fitting Web App")
st.markdown("Paste or enter your fluorescence/time data below. First column should be time (in hours), others are samples.")

# Input labels
x_label = st.text_input("X-axis label", value="Time (h)")
y_label = st.text_input("Y-axis label", value="Signal")

# Dynamic sample data
num_samples = st.number_input("How many samples do you want to enter?", min_value=1, max_value=20, value=2, step=1)
sample_data = {"Time": np.arange(0, 4.25, 0.25)}
labels = []

for i in range(1, num_samples + 1):
    label = st.text_input(f"Label for Sample {i}", value=f"Sample{i}")
    labels.append(label)
    sample_data[label] = np.linspace(1 + i, 25 - i, len(sample_data["Time"]))

example_data = pd.DataFrame(sample_data)

# File upload or manual data entry
uploaded_file = st.file_uploader("Upload a CSV file (wide format, first column = Time)", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Data loaded from file.")
else:
    data = st.data_editor(example_data, use_container_width=True, num_rows="dynamic")

# Threshold and plot settings
manual_thresh = st.number_input("Enter manual threshold:", min_value=0.0, value=3.0, step=0.1)
fmt = st.selectbox("Select image format for download", options=["png", "jpeg", "svg", "pdf"], index=0)
dpi = st.slider("Image resolution (DPI)", min_value=100, max_value=600, value=300, step=50)

# 5PL Fitting Process
if st.button("Run Analysis"):
    st.subheader("📊 Results")
    time = data.iloc[:, 0].dropna().values
    
    if len(time) == 0:
        st.error("❌ No valid time values found in the first column.")
    else:
        for col in data.columns[1:]:
            y = data[col].dropna().values
            t_fit = time[:len(y)]

            if len(t_fit) != len(y):
                st.error(f"❌ Sample '{col}' has mismatched time and data lengths.")
                continue

            try:
                initial_guess = [np.max(y), np.min(y), np.median(t_fit), 1, 1]
                popt, pcov = curve_fit(logistic_5pl, t_fit, y, p0=initial_guess, maxfev=10000)
                y_fit = logistic_5pl(t_fit, *popt)
                r2 = np.corrcoef(y, y_fit)[0, 1]**2

                try:
                    Tt = calculate_threshold_time(manual_thresh, popt)
                except ValueError as ve:
                    st.error(f"❌ {ve}")
                    Tt = np.nan

                fig, ax = plt.subplots(figsize=(10, 10))
                ax.plot(t_fit, y, 'ko', label="Raw Data")
                ax.plot(t_fit, y_fit, 'b-', label="5PL Fit")
                ax.axhline(manual_thresh, color='green', linestyle='-', linewidth=2, label="Threshold")
                ax.set_title(f"{col} Fit")
                ax.set_xlabel(x_label, fontweight='bold')
                ax.set_ylabel(y_label, fontweight='bold')
                ax.legend(title=f"{col} (Tt = {Tt:.2f} h)" if not np.isnan(Tt) else col)
                st.pyplot(fig)

                buffer = BytesIO()
                fig.savefig(buffer, format=fmt, dpi=dpi)
                buffer.seek(0)

                st.download_button(
                    label=f"📥 Download Fit Plot for {col}",
                    data=buffer.getvalue(),
                    file_name=f"{col}_fit_plot.{fmt}",
                    mime=f"image/{fmt}"
                )

                st.write(f"Fitting Results for {col}:")
                st.write(f"- R²: {r2:.4f}")
                st.write(f"- Parameters: {popt}")
                if not np.isnan(Tt):
                    st.write(f"- Threshold Time (Tt): {Tt:.2f} hours")

            except Exception as e:
                st.error(f"❌ Could not fit {col}: {e}")
