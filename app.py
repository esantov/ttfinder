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


def inverse_threshold_curve(y, model_func, popt):
    try:
        result = root_scalar(lambda t: model_func(t, *popt) - y, bracket=[0, 1e3], method='brentq')
        return result.root if result.converged else None
    except Exception as e:
        st.warning(f"⚠️ Could not find inverse threshold curve: {e}")
        return None


# --- Excel Export Function ---
def create_excel_report(data, fit_results, summary_rows, calibration, x_label, y_label, threshold):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Original Data
        if not data.empty:
            data.to_excel(writer, sheet_name="Original Data", index=False)

        # Summary Sheet
        summary_export = [
            {
                "Sample": row.get("Sample"),
                "Threshold Value": threshold,
                "Threshold Time (Tt, h)": row.get("Threshold Time"),
                "TT CI Lower": row.get("TT CI Lower"),
                "TT CI Upper": row.get("TT CI Upper"),
                "TT StdErr": row.get("TT StdErr"),
                "Log CFU/mL": row.get("Log CFU/mL"),
            }
            for row in summary_rows
        ]
        pd.DataFrame(summary_export).to_excel(writer, sheet_name="Summary", index=False)

        # Fit Parameters
        formula_map = {
            '5PL': lambda p: f"y = {p[1]:.2f} + ({p[0]:.2f} - {p[1]:.2f}) / (1 + (x / {p[2]:.2f})^{p[3]:.2f})^{p[4]:.2f}",
            '4PL': lambda p: f"y = {p[1]:.2f} + ({p[0]:.2f} - {p[1]:.2f}) / (1 + (x / {p[2]:.2f})^{p[3]:.2f})",
            'Sigmoid': lambda p: f"y = {p[0]:.2f} / (1 + exp(-{p[2]:.2f}*(x - {p[1]:.2f})))",
            'Linear': lambda p: f"y = {p[0]:.2f} * x + {p[1]:.2f}",
        }

        param_rows = [
            {
                "Sample": row["Sample"],
                "Model": row["Model"],
                "R² of Fit": row.get("R²"),
                "Formula": formula_map.get(row["Model"], lambda _: "")(
                    fit_results[row["Sample"]].attrs.get('popt', [])
                ),
            }
            for row in summary_rows
        ]
        pd.DataFrame(param_rows).to_excel(writer, sheet_name="Fit Parameters", index=False)

    output.seek(0)
    return output


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
        model = st.selectbox("Model", ["5PL", "4PL", "Sigmoid", "Linear"], key=col)
        try:
            if model == "Linear":
                popt = np.polyfit(x, y, 1)
                y_fit = np.polyval(popt, x)
            elif model == "5PL":
                popt, _ = curve_fit(logistic_5pl, x, y, maxfev=10000)
                y_fit = logistic_5pl(x, *popt)
            else:
                st.error("Only Linear and 5PL models implemented.")
                continue

            # Save Results
            fit_df = pd.DataFrame({"Time": x, "Raw": y, "Fit": y_fit})
            fit_results[col] = fit_df
            st.line_chart(y_fit)

        except Exception as e:
            st.error(f"Fit failed for {col}: {e}")
