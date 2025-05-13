
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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

def inverse_threshold_curve(y, model_func, popt):
    from scipy.optimize import root_scalar
    try:
        result = root_scalar(lambda t: model_func(t, *popt) - y, bracket=[0, 1e3], method='brentq')
        return result.root if result.converged else None
    except:
        return None

# --- Plotting & Export Functions ---
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
    title = f"{sample} Fit"
    if tt_val is not None and logcfu is not None:
        title += f" (TT: {tt_val:.2f} h, LogCFU/mL: {logcfu:.2f})"
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    plt.close(fig)
    return buf


def generate_combined_plot(fit_results_dict, threshold, x_label, y_label, summary_rows):
    fig, ax = plt.subplots(figsize=(10, 6))
    for row in summary_rows:
        sample = row['Sample']
        df = fit_results_dict[sample]
        label = f"{sample} (TT={row['Threshold Time']:.2f}, CFU={row['Log CFU/mL']:.2f})"
        ax.plot(df['Time'], df['Fit'], label=label)
        ax.fill_between(df['Time'], df['CI Lower'], df['CI Upper'], alpha=0.3)
    ax.axhline(y=threshold, color='green', linestyle='--', label='Threshold')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Combined Fit Plot")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    plt.close(fig)
    return buf


def create_excel_report(data, fit_results, summary_rows, calibration, x_label, y_label):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        if not data.empty:
            data.to_excel(writer, sheet_name="Original Data", index=False)
        if calibration:
            (a, b), _ = calibration
            pd.DataFrame({"Calibration Name": ["Manual"], "Slope": [a], "Intercept": [b]}).to_excel(writer, sheet_name="Calibration", index=False)
        if summary_rows:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

        # Add parameters with formulas
        param_rows = []
        formula_map = {
            '5PL': "y = d + (a - d) / (1 + (x / c)^b)^g",
            '4PL': "y = d + (a - d) / (1 + (x / c)^b)",
            'Sigmoid': "y = L / (1 + exp(-k*(x - x0)))",
            'Linear': "y = a*x + b"
        }
        inverse_map = {
            '5PL': "x = c * (((a - d)/(y - d))^(1/g) - 1)^(1/b)",
            '4PL': "x = c * ((a - d)/(y - d) - 1)^(1/b)",
            'Sigmoid': "x = x0 - log((L/y) - 1)/k",
            'Linear': "x = (y - b) / a"
        }

        for row in summary_rows:
            model = row['Model']
            params = fit_results.get(row['Sample'])
            param_vals = params.get('Fit') if params is not None else []
            if isinstance(param_vals, (list, tuple, np.ndarray)):
                values = list(param_vals)
            else:
                values = []
            entry = {
                "Sample": row['Sample'],
                "Model": model,
                "Formula": formula_map.get(model, ""),
                "Inverse": inverse_map.get(model, "")
            }
            for i, v in enumerate(values):
                entry[f"p{i+1}"] = round(v, 4)
            param_rows.append(entry)

        pd.DataFrame(param_rows).to_excel(writer, sheet_name="Fit Parameters", index=False)

        for sample, df in fit_results.items():
            df.to_excel(writer, sheet_name=sample[:31], index=False)

    output.seek(0)
    return output


# --- UI ---
st.title("📈 TT Finder - Curve Fitting Tool")
x_label = st.text_input("X-axis label", "Time (h)")
y_label = st.text_input("Y-axis label", "Signal")
manual_thresh = st.number_input("Threshold", 0.0, 100.0, 3.0, 0.1)
dpi = st.slider("Plot DPI", 100, 600, 300, 50)

# Calibration
st.markdown("### Calibration")
manual_calib = st.checkbox("Use manual calibration")
if manual_calib:
    a = st.number_input("Slope (a)", value=0.0)
    b = st.number_input("Intercept (b)", value=0.0)
    st.session_state['calibration_coef'] = ([a, b], None)

# Data Upload
uploaded = st.file_uploader("Upload CSV", type="csv")
if uploaded:
    data = pd.read_csv(uploaded)
    st.dataframe(data.head())
else:
    data = pd.DataFrame({"Time": []})

# Init
if 'summary_rows' not in st.session_state:
    st.session_state['summary_rows'] = []

fit_results = {}
