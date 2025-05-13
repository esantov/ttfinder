import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import t
from io import BytesIO
import datetime
import plotly.graph_objects as go
import xlsxwriter
import tempfile
import os
import zipfile

# SESSION STATE INIT
if 'summary_rows' not in st.session_state:
    st.session_state.summary_rows = []
for key in ['fits', 'ci', 'model_choices', 'calibration_coef']:
    if key not in st.session_state:
        st.session_state[key] = {}

# Model functions
def logistic_5pl(x, a, d, c, b, g):
    return d + (a - d) / (1 + (x / c) ** b) ** g

def logistic_4pl(x, a, d, c, b):
    return d + (a - d) / (1 + (x / c) ** b)

def sigmoid(x, L, x0, k):
    return L / (1 + np.exp(-k * (x - x0)))

def inverse_threshold_curve(y, model_func, popt):
    try:
        from scipy.optimize import root_scalar
        result = root_scalar(lambda t: model_func(t, *popt) - y, bracket=[0, 1e3], method='brentq')
        return result.root if result.converged else None
    except:
        return None

# UI
st.title("📈 TT Finder - Curve Fitting Tool")
if st.button("Clear All"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.experimental_rerun()

x_label = st.text_input("X-axis label", value="Time (h)")
y_label = st.text_input("Y-axis label", value="Signal")
manual_thresh = st.number_input("Manual threshold", 0.0, 100.0, 3.0, 0.1)
fmt = st.selectbox("Image format", ["png", "jpeg", "svg", "pdf"])
dpi = st.slider("DPI", 100, 600, 300, 50)

# Calibration
st.markdown("### Calibration")
calib_name = st.text_input("Calibration name", value="Default Calibration")
cal_slope = st.number_input("Calibration slope (a)", value=0.0)
cal_intercept = st.number_input("Calibration intercept (b)", value=0.0)
manual_calib = st.checkbox("Use manual calibration", value=False)

if manual_calib:
    coef = [cal_slope, cal_intercept]
    cov = np.array([[0.01, 0.0], [0.0, 0.01]])
    st.session_state.calibration_coef = (coef, cov)
else:
    cal_df = st.file_uploader("Upload calibration CSV (TT, logCFU/mL)", type="csv", key="calib")
    if cal_df:
        calib_data = pd.read_csv(cal_df)
        if calib_data.shape[1] >= 2:
            tt = calib_data.iloc[:, 0]
            logcfu = calib_data.iloc[:, 1]
            coef, cov = np.polyfit(tt, logcfu, 1, cov=True)
            st.session_state.calibration_coef = (coef, cov)

# Data input
st.markdown("### Data Input")
uploaded_data = st.file_uploader("Upload CSV (first column = Time)", type=["csv"])
if uploaded_data:
    data = pd.read_csv(uploaded_data)
    st.success("✅ Data loaded from uploaded file.")
    st.dataframe(data.head())
    if len(data.columns) <= 1:
        st.warning("⚠️ Please upload a CSV with at least two columns (Time + Samples).")
else:
    data = st.data_editor(pd.DataFrame({"Time": []}), num_rows="dynamic", use_container_width=True)

# Run analysis if data is available
if not data.empty and len(data.columns) > 1:
    st.session_state.summary_rows.clear()
    fit_results = {}
    combined_fig = go.Figure()

    time_vals = data.iloc[:, 0].dropna().values

    for col in data.columns[1:]:
        y_vals = data[col].dropna().values
        x_vals = time_vals[:len(y_vals)]

        with st.expander(f"{col}"):


        model_choice = st.selectbox("Choose model", ["5PL", "4PL", "Sigmoid", "Linear"], key=f"model_{col}")
        st.session_state.model_choices[col] = model_choice
        x_vals = time_vals[:len(y_vals)]
        

        model_func, p0 = None, None
        if model_choice == "5PL":
            model_func = logistic_5pl
            p0 = [min(y_vals), max(y_vals), np.median(x_vals), 1, 1]
        elif model_choice == "4PL":
            model_func = logistic_4pl
            p0 = [min(y_vals), max(y_vals), np.median(x_vals), 1]
        elif model_choice == "Sigmoid":
            model_func = sigmoid
            p0 = [max(y_vals), np.median(x_vals), 1]

        try:
            if model_choice == "Linear":
                coef = np.polyfit(x_vals, y_vals, 1)
                y_fit = np.polyval(coef, x_vals)
                r2 = r2_score(y_vals, y_fit)
                popt = coef
                y_ci = (y_fit, y_fit)
            else:
                popt, pcov = curve_fit(model_func, x_vals, y_vals, p0=p0, maxfev=10000)
                y_fit = model_func(x_vals, *popt)
                r2 = r2_score(y_vals, y_fit)
                dof = max(len(x_vals) - len(popt), 1)
                tval = t.ppf(0.975, dof)
                ci_low, ci_high = [], []
                for i, xi in enumerate(x_vals):
                    grad = np.array([
                        (model_func(xi, *(popt + np.eye(len(popt))[j]*1e-5)) - y_fit[i]) / 1e-5
                        for j in range(len(popt))
                    ])
                    se = np.sqrt(grad @ pcov @ grad.T)
                    delta = tval * se
                    ci_low.append(y_fit[i] - delta)
                    ci_high.append(y_fit[i] + delta)
                y_ci = (np.array(ci_low), np.array(ci_high))

            tt_val = inverse_threshold_curve(manual_thresh, model_func if model_choice != "Linear" else lambda t, a, b: a*t + b, popt)
            logcfu = None
            if tt_val and 'calibration_coef' in st.session_state:
                (a, b), cov = st.session_state.calibration_coef
                logcfu = a * tt_val + b

            combined_fig.add_trace(go.Scatter(x=x_vals, y=y_fit, mode='lines', name=f'{col} (TT={tt_val:.2f}, CFU={logcfu:.2f})'))
            combined_fig.add_trace(go.Scatter(x=x_vals, y=y_ci[0], fill=None, mode='lines', line=dict(color='rgba(0,0,0,0.2)', width=0), showlegend=False))
            combined_fig.add_trace(go.Scatter(x=x_vals, y=y_ci[1], fill='tonexty', mode='lines', name=f'{col} 95% CI', line=dict(color='rgba(0,0,0,0.2)', width=0)))

            st.session_state.model_params = st.session_state.get('model_params', {})
            st.session_state.model_params[col] = popt if 'popt' in locals() else []
            st.session_state.summary_rows.append({
                'Sample': col,
                'Model': model_choice,
                'R²': round(r2, 4),
                'Threshold Time': tt_val,
                'Log CFU/mL': logcfu
            })

            fit_results[col] = pd.DataFrame({
                'Time': x_vals,
                'Raw': y_vals,
                'Fit': y_fit,
                'CI Lower': y_ci[0],
                'CI Upper': y_ci[1]
            })

        except Exception as e:
            st.sidebar.error(f"Error fitting {col}: {e}")

        if 'y_fit' in locals():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers', name='Data', marker=dict(color='black')))
            fig.add_trace(go.Scatter(x=x_vals, y=y_fit, mode='lines', name='Fit', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=x_vals, y=y_ci[0], fill=None, mode='lines', line=dict(color='rgba(255,0,0,0.2)', width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=x_vals, y=y_ci[1], fill='tonexty', mode='lines', name='95% CI', line=dict(color='rgba(255,0,0,0.2)', width=0)))
            fig.add_hline(y=manual_thresh, line_dash="dash", line_color="green", annotation_text="Threshold", annotation_position="top right")
            if tt_val is not None:
                fig.add_vline(x=tt_val, line_dash="dot", line_color="green", annotation_text="TT", annotation_position="bottom right")
            fig.update_layout(title=f"{col} Fit", margin=dict(l=40, r=40, t=60, b=40),
                              xaxis=dict(dtick=1, tickformat=".2f", color='black', linecolor='black', linewidth=2, showgrid=False, mirror=True),
                              yaxis=dict(tickformat=".2f", color='black', linecolor='black', linewidth=2, showgrid=False, mirror=True),
                              plot_bgcolor='white', legend=dict(x=1.02, y=1, xanchor='left', yanchor='top', bordercolor='black', borderwidth=1))
            st.plotly_chart(fig, use_container_width=True)
        if tt_val is not None:
            st.markdown(f"**Threshold Time (TT):** {tt_val:.2f} h")
        if logcfu is not None:
            st.markdown(f"**Log CFU/mL:** {logcfu:.2f}")

    st.subheader("Combined Fit Plot")
    st.plotly_chart(combined_fig, use_container_width=True)

    zip_buffer = BytesIO()
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_excel:
        with pd.ExcelWriter(temp_excel.name, engine='xlsxwriter') as writer:
            pd.DataFrame(st.session_state.summary_rows).to_excel(writer, sheet_name='Summary', index=False)
            for sample, df in fit_results.items():
                df.to_excel(writer, sheet_name=sample[:31], index=False)
            if 'calibration_coef' in st.session_state:
                (a, b), _ = st.session_state['calibration_coef']
                calib_df = pd.DataFrame({"Calibration Name": [calib_name], "Slope": [a], "Intercept": [b]})
                calib_df.to_excel(writer, sheet_name='Calibration', index=False)
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
            for row in st.session_state.summary_rows:
                model = row['Model']
                params = st.session_state.get('model_params', {}).get(row['Sample'], [])
                param_rows.append({
                    'Sample': row['Sample'],
                    'Model': model,
                    'Curve Formula': formula_map.get(model, ""),
                    'Inverse Formula': inverse_map.get(model, ""),
                    'Parameters': ', '.join([f'{v:.4f}' for v in params]) if isinstance(params, (list, tuple, np.ndarray)) else f'{params:.4f}' if params is not None else 'N/A'
                })
            pd.DataFrame(param_rows).to_excel(writer, sheet_name='Fit Parameters', index=False)
            data.to_excel(writer, sheet_name='Original Data', index=False)
        temp_excel.seek(0)
        st.download_button("📥 Download Excel Report", data=temp_excel.read(), file_name="tt_finder_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("📦 Download All Results as ZIP", data=zip_buffer.read(), file_name="tt_finder_results.zip", mime="application/zip")
