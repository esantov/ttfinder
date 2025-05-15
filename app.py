import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root_scalar
from sklearn.metrics import r2_score
from scipy.stats import t
from io import BytesIO
import datetime
import zipfile

# --- Model Functions ---
def logistic_5pl(x, a, d, c, b, g):
    return d + (a - d) / (1 + (x / c) ** b) ** g


def logistic_4pl(x, a, d, c, b):
    return d + (a - d) / (1 + (x / c) ** b)


def sigmoid(x, L, x0, k):
    return L / (1 + np.exp(-k * (x - x0)))


def gompertz(x, A, B, C):
    """Gompertz growth model."""
    return A * np.exp(-B * np.exp(-C * x))


def inverse_threshold_curve(y, model_func, popt):
    try:
        result = root_scalar(lambda t: model_func(t, *popt) - y, bracket=[0, 1e3], method='brentq')
        return result.root if result.converged else None
    except Exception as e:
        st.warning(f"⚠️ Could not find inverse threshold curve: {e}")
        return None


# --- Plotting ---
def generate_sample_plot(sample, df, x_label, y_label, threshold, tt_val=None, logcfu=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df['Time'], df['Raw'], 'o', label='Data', color='black')
    ax.plot(df['Time'], df['Fit'], label='Fit', color='blue')
    ax.fill_between(df['Time'], df['CI Lower'], df['CI Upper'], color='red', alpha=0.1, label='95% CI')
    ax.axhline(y=threshold, color='green', linestyle='--', label='Threshold')
    if tt_val is not None:
        ax.axvline(x=tt_val, color='orange', linestyle=':', label='TT')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{sample} Fit (TT: {tt_val:.2f} h, LogCFU/mL: {logcfu:.2f})" if tt_val and logcfu else f"{sample} Fit")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    plt.close(fig)
    return buf


# --- Streamlit App ---
st.title("📈 TT Finder - Curve Fitting Tool")

# Inputs
x_label = st.text_input("X-axis label", "Time (h)")
y_label = st.text_input("Y-axis label", "Signal")
threshold = st.number_input("Threshold", 0.0, 100.0, 3.0, 0.1)

# Calibration
manual_calib = st.checkbox("Use manual calibration")
if manual_calib:
    slope = st.number_input("Slope (a)", value=1.0)
    intercept = st.number_input("Intercept (b)", value=0.0)
    calibration = ([slope, intercept], None)
else:
    calibration = None

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.dataframe(data.head())
else:
    st.warning("Please upload a CSV file.")
    st.stop()

# Fit models
fit_results = {}
summary_rows = []

for col in data.columns[1:]:
    x = data.iloc[:, 0].values
    y = data[col].values

    with st.expander(f"Fit: {col}"):
        model = st.selectbox("Model", ["5PL", "4PL", "Sigmoid", "Gompertz", "Linear"], key=col)
        try:
            # Define initial parameters and fitting function
            if model == "Linear":
                popt = np.polyfit(x, y, 1)
                y_fit = np.polyval(popt, x)
            elif model == "5PL":
                popt, _ = curve_fit(logistic_5pl, x, y, maxfev=10000)
                y_fit = logistic_5pl(x, *popt)
            elif model == "4PL":
                popt, _ = curve_fit(logistic_4pl, x, y, maxfev=10000)
                y_fit = logistic_4pl(x, *popt)
            elif model == "Sigmoid":
                popt, _ = curve_fit(sigmoid, x, y, maxfev=10000)
                y_fit = sigmoid(x, *popt)
            elif model == "Gompertz":
                popt, _ = curve_fit(gompertz, x, y, maxfev=10000)
                y_fit = gompertz(x, *popt)
            else:
                st.error("Unsupported model selected.")
                continue

            # Calculate confidence intervals
            dof = len(x) - len(popt)
            tval = t.ppf(0.975, dof)
            ci_lower, ci_upper = [], []
            for i, xi in enumerate(x):
                grad = np.array([(gompertz(xi, *(popt + np.eye(len(popt))[j]*1e-5)) - y_fit[i]) / 1e-5 for j in range(len(popt))])
                se = np.sqrt(grad @ np.cov(x) )
