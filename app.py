import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.stats import t

# ----- Simple 5PL logistic function for fitting -----
def logistic_5pl(t, a, d, c, b, g):
    return d + (a - d) / (1 + (t / c)**b)**g

# ----- Function to calculate the threshold time (Tt) -----
def calculate_threshold_time(threshold, popt, param_cov, t_values):
    # Compute the inverse of the 5PL function to find Tt
    a, d, c, b, g = popt
    # Calculate the standard errors of the parameters
    param_errors = np.sqrt(np.diag(param_cov))
    
    # Compute the threshold time (Tt) where y crosses the threshold
    y_at_thresh = threshold
    try:
        Tt = c * (((a - d) / (y_at_thresh - d)) ** (1 / g) - 1) ** (1 / b)
    except Exception as e:
        Tt = np.nan  # In case of any issues with the calculation
        st.error(f"❌ Error calculating threshold time for parameters: {e}")
    return Tt

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

manual_thresh = st.number_input("Enter manual threshold:", min_value=0.0, value=3.0, step=0.1)

fmt = st.selectbox("Select image format for download", options=["png", "jpeg", "svg", "pdf"], index=0)
dpi = st.slider("Image resolution (DPI)", min_value=100, max_value=600, value=300, step=50)

# 5PL Fitting Process
if st.button("Run Analysis"):
    st.subheader("📊 Results")
    
    # Extract time data and ensure proper alignment
    time = data.iloc[:, 0].dropna().values
    if len(time) == 0:
        st.error("❌ No valid time values found in the first column.")
    
    for col in data.columns[1:]:
        y = data[col].dropna().values
        t_fit = time[:len(y)]  # Ensure time and y values match in length

        # Validate if time and y match in length
        if len(t_fit) != len(y):
            st.error(f"❌ Sample '{col}' has mismatched time and data lengths.")
            continue
        
        # Check if there are any NaN or None values in time or y
        if np.any(np.isnan(t_fit)) or np.any(np.isnan(y)):
            st.error(f"❌ Sample '{col}' contains NaN values in time or data.")
            continue

        try:
            # Ensure data is valid before fitting
            if len(t_fit) > 1:  # Ensure we have enough data points to fit
                popt_logistic, pcov = curve_fit(logistic_5pl, t_fit, y, p0=[min(y), max(y), np.median(t_fit), 1, 1], maxfev=10000)
                y_fit_logistic = logistic_5pl(t_fit, *popt_logistic)
                r2_logistic = np.corrcoef(y, y_fit_logistic)[0, 1]**2  # Compute R^2 manually

                # Calculate threshold time (Tt) based on manual threshold
                Tt = calculate_threshold_time(manual_thresh, popt_logistic, pcov, t_fit)

                # Calculate confidence intervals (CI)
                # Calculate standard errors for each parameter
                param_errors = np.sqrt(np.diag(pcov))
                ci_low = logistic_5pl(t_fit, *(popt_logistic - param_errors))
                ci_high = logistic_5pl(t_fit, *(popt_logistic + param_errors))

                # Plot Results
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.plot(t_fit, y, 'ko', label="Raw Data")
                ax.plot(t_fit, y_fit_logistic, 'b-', label="5PL Fit")

                # Plot confidence intervals (95% CI)
                ax.plot(t_fit, ci_low, 'r-', linewidth=1, label="95% CI (Low)")
                ax.plot(t_fit, ci_high, 'r-', linewidth=1, label="95% CI (High)")

                ax.axhline(manual_thresh, color='green', linestyle='-', linewidth=2, label="Threshold")
                ax.set_title(f"{col} Fit")
                ax.set_xlabel(x_label, fontweight='bold')
                ax.set_ylabel(y_label, fontweight='bold')

                # Show Tt (threshold time) in the legend
                ax.legend(title=f"{col} (Tt = {Tt:.2f} h)")
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
                st.write(f"- Threshold Time (Tt): {Tt:.2f} hours")

            else:
                st.warning(f"❌ Sample '{col}' has insufficient data for fitting (at least two data points required).")

        except Exception as e:
            st.error(f"❌ Could not fit {col}: {e}")
