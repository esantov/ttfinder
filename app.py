import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from io import BytesIO

# ----- Simple 5PL logistic function for fitting -----
def logistic_5pl(t, a, d, c, b, g):
    return d + (a - d) / (1 + (t / c)**b)**g

# ----- Debugging function to check data -----
def debug_data_state(time, y, sample_label):
    st.write(f"Debugging {sample_label}:")
    st.write(f"Time: {time}")
    st.write(f"Y values: {y}")
    st.write(f"Data length check: Time length = {len(time)}, Y length = {len(y)}")
    st.write(f"NaN check for Time: {np.isnan(time).sum()} NaNs in Time")
    st.write(f"NaN check for Y: {np.isnan(y).sum()} NaNs in Y")

# ----- APP LOGIC -----
x_label = st.text_input("X-axis label", value="Time (h)")
y_label = st.text_input("Y-axis label", value="Signal")

st.title("📈 5PL Curve Fitting Web App")
st.markdown("Paste or enter your fluorescence/time data below. First column should be time (in hours), others are samples.")

# Handle the dynamic number of samples properly
num_samples = st.number_input("How many samples do you want to enter?", min_value=1, max_value=20, value=2, step=1)

# Sample data: We'll keep it simple for now
sample_data = {"Time": np.arange(0, 4.25, 0.25)}  # Time is 0 to 4.25 in 0.25 intervals
labels = []
for i in range(1, num_samples + 1):
    default_label = f"Sample{i}"
    label = st.text_input(f"Label for Sample {i}", value=default_label)
    labels.append(label)
    sample_data[label] = np.linspace(1 + i, 25 - i, 17)  # Generate some simple example data

example_data = pd.DataFrame(sample_data)

uploaded_file = st.file_uploader("Upload a CSV file (wide format, first column = Time)", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Data loaded from file")
else:
    data = st.data_editor(example_data, use_container_width=True, num_rows="dynamic")

auto_thresh = st.checkbox("Auto threshold (50% of max)", value=True)
manual_thresh = st.number_input("Or enter manual threshold:", min_value=0.0, value=3.0, step=0.1)

fmt = st.selectbox("Select image format for download", options=["png", "jpeg", "svg", "pdf"], index=0)
dpi = st.slider("Image resolution (DPI)", min_value=100, max_value=600, value=300, step=50)

# 5PL Fitting Process
if st.button("Run Analysis"):
    st.subheader("📊 Results")
    
    # Extract time data and ensure proper alignment
    time = data.iloc[:, 0].dropna().values

    for col in data.columns[1:]:
        y = data[col].dropna().values
        t_fit = time[:len(y)]  # Ensure time and y values match in length

        # Debug data state
        debug_data_state(t_fit, y, col)

        # Check if lengths of time and y match
        if len(t_fit) != len(y):
            st.error(f"❌ Sample '{col}' has mismatched time and data lengths.")
            continue
        
        try:
            # Logistic Fit
            if len(t_fit) > 1:  # Ensure we have enough data points to fit
                popt_logistic, _ = curve_fit(logistic_5pl, t_fit, y, p0=[min(y), max(y), np.median(t_fit), 1, 1], maxfev=10000)
                y_fit_logistic = logistic_5pl(t_fit, *popt_logistic)
                r2_logistic = np.corrcoef(y, y_fit_logistic)[0, 1]**2  # Compute R^2 manually

                # Plot Results
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.plot(t_fit, y, 'ko', label="Raw Data")
                ax.plot(t_fit, y_fit_logistic, 'b-', label="5PL Fit")
                ax.axhline(auto_thresh and (max(y_fit_logistic) * 0.5) or manual_thresh, color='green', linestyle='-', linewidth=2, label="Threshold")
                ax.set_title(f"{col} Fit")
                ax.set_xlabel(x_label, fontweight='bold')
                ax.set_ylabel(y_label, fontweight='bold')
                ax.legend()
                ax.grid(False)
                st.pyplot(fig)

                # Prepare for download (Optional: Just showing basic CSV for now)
                st.download_button(
                    label=f"📥 Download Fit Plot for {col}",
                    data=fig.savefig(BytesIO(), format='png', dpi=dpi),
                    file_name=f"{col}_fit_plot.png",
                    mime="image/png"
                )
                
                # Display fitting parameters
                st.write(f"Fitting Results for {col}:")
                st.write(f"- R²: {r2_logistic:.4f}")
                st.write(f"- Parameters: {popt_logistic}")
            else:
                st.warning(f"❌ Sample '{col}' has insufficient data for fitting (at least two data points required).")

        except Exception as e:
            st.error(f"❌ Could not fit {col}: {e}")
