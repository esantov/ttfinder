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
    except:
        return None

# --- Excel Export Function ---
def create_excel_report(data, fit_results, summary_rows, calibration, x_label, y_label, threshold):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        if not data.empty:
            data.to_excel(writer, sheet_name="Original Data", index=False)

        summary_export = []
        for row in summary_rows:
            summary_export.append({
                "Sample": row.get("Sample"),
                "Threshold Value": threshold,
                "Threshold Time (Tt, h)": row.get("Threshold Time"),
                "TT CI Lower": row.get("TT CI Lower"),
                "TT CI Upper": row.get("TT CI Upper"),
                "TT StdErr": row.get("TT StdErr"),
                "Log CFU/mL": row.get("Log CFU/mL")
            })
        pd.DataFrame(summary_export).to_excel(writer, sheet_name="Summary", index=False)

        formula_map = {
            '5PL': lambda p: f"y = {p[1]:.2f} + ({p[0]:.2f} - {p[1]:.2f}) / (1 + (x / {p[2]:.2f})^{p[3]:.2f})^{p[4]:.2f}",
            '4PL': lambda p: f"y = {p[1]:.2f} + ({p[0]:.2f} - {p[1]:.2f}) / (1 + (x / {p[2]:.2f})^{p[3]:.2f})",
            'Sigmoid': lambda p: f"y = {p[0]:.2f} / (1 + exp(-{p[2]:.2f}*(x - {p[1]:.2f})))",
            'Linear': lambda p: f"y = {p[0]:.2f} * x + {p[1]:.2f}"
        }
        inverse_map = {
            '5PL': lambda p: f"x = {p[2]:.2f} * ((({p[0]:.2f} - {p[1]:.2f}) / (y - {p[1]:.2f}))**(1/{p[4]:.2f}) - 1)**(1/{p[3]:.2f})",
            '4PL': lambda p: f"x = {p[2]:.2f} * (({p[0]:.2f} - {p[1]:.2f}) / (y - {p[1]:.2f}) - 1)**(1/{p[3]:.2f})",
            'Sigmoid': lambda p: f"x = {p[1]:.2f} - log(({p[0]:.2f}/y) - 1) / {p[2]:.2f}",
            'Linear': lambda p: f"x = (y - {p[1]:.2f}) / {p[0]:.2f}"
        }

        param_rows = []
        for row in summary_rows:
            sample = row["Sample"]
            model = row["Model"]
            r2 = row.get("R²")
            df = fit_results.get(sample)
            popt = df.attrs.get('popt') if df is not None and hasattr(df, 'attrs') else None
            formula = formula_map.get(model, lambda _: "")(popt) if popt is not None else ""
            inverse = inverse_map.get(model, lambda _: "")(popt) if popt is not None else ""
            param_rows.append({
                "Sample": sample,
                "Model": model,
                "R² of Fit": r2,
                "Formula": formula,
                "Inverse": inverse
            })
        pd.DataFrame(param_rows).to_excel(writer, sheet_name="Fit Parameters", index=False)

        merged_rows = []
        for sample, df in fit_results.items():
            for _, row in df.iterrows():
                merged_rows.append({
                    "Sample": sample,
                    "Time": row["Time"],
                    "Raw": row.get("Raw"),
                    "Fit": row["Fit"],
                    "CI Lower": row["CI Lower"],
                    "CI Upper": row["CI Upper"]
                })
        pd.DataFrame(merged_rows).to_excel(writer, sheet_name="Fit Data", index=False)

        if calibration:
            (a, b), _ = calibration
            pd.DataFrame({
                "Calibration Name": ["Manual"],
                "Slope": [a],
                "Intercept": [b]
            }).to_excel(writer, sheet_name="Calibration", index=False)

    output.seek(0)
    return output

# --- Hook into Streamlit ---
if 'fit_results' in st.session_state and 'summary_rows' in st.session_state:
    data = st.session_state.get("uploaded_data", pd.DataFrame())
    fit_results = st.session_state["fit_results"]
    summary_rows = st.session_state["summary_rows"]
    calibration = st.session_state.get("calibration_coef")
    x_label = st.session_state.get("x_label", "Time (h)")
    y_label = st.session_state.get("y_label", "Signal")
    threshold = st.session_state.get("threshold", 3.0)

    if fit_results and summary_rows:
        excel_buf = create_excel_report(
            data, fit_results, summary_rows,
            calibration, x_label, y_label, threshold
        )

        st.download_button(
            "📥 Download Excel Report",
            data=excel_buf,
            file_name=f"tt_finder_report_{datetime.datetime.now():%Y%m%d_%H%M%S}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
