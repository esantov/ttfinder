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
    from scipy.optimize import root_scalar
    st.session_state.summary_rows.clear()
    fit_results = {}
    combined_fig = go.Figure()
    combined_fig.update_layout(
    title="Combined Model Fits",
    xaxis_title=x_label,
    yaxis_title=y_label,
    plot_bgcolor='white',
    margin=dict(l=40, r=40, t=60, b=40),
    xaxis=dict(
        range=[0, 24],
        type='linear',
        tickmode='linear',
        dtick=1,
        tickformat=".2f",
        color='black',
        linecolor='black',
        linewidth=2,
        showgrid=False,
        mirror=True
    ),
    yaxis=dict(
        range=[0, 100],
        tickformat=".2f",
        color='black',
        linecolor='black',
        linewidth=2,
        showgrid=False,
        mirror=True
    )
),
    xaxis=dict(
        range=[0, 24],
        type='linear',
        tickmode='linear',
        dtick=1,
        tickformat=".2f",
        color='black',
        linecolor='black',
        linewidth=2,
        showgrid=False,
        mirror=True
    ),
    yaxis=dict(
        range=[0, 100],
        tickformat=".2f",
        color='black',
        linecolor='black',
        linewidth=2,
        showgrid=False,
        mirror=True
    ),
    xaxis=dict(
        type='linear',
        tickmode='linear',
        dtick=1,
        tickformat=".2f",
        color='black',
        linecolor='black',
        linewidth=2,
        showgrid=False,
        mirror=True
    ),
    yaxis=dict(
        tickformat=".2f",
        color='black',
        linecolor='black',
        linewidth=2,
        showgrid=False,
        mirror=True
    ),
    yaxis=dict(color='black', linecolor='black', showgrid=False)
    )

    time_vals = data.iloc[:, 0].dropna().values
    for col in data.columns[1:]:
        y_vals = data[col].dropna().values
        x_vals = time_vals[:len(y_vals)]
        model_choice = st.selectbox(f"Model for {col}", ["5PL", "4PL", "Sigmoid", "Linear"], key=f"model_{col}")
        st.session_state.model_choices[col] = model_choice

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

            title = f"{col} – {model_choice} Fit"
            if tt_val is not None:
                title += f" | TT: {tt_val:.2f}"
            if logcfu is not None:
                title += f" | LogCFU/mL: {logcfu:.2f}"

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers', name='Data'))
            fig.add_trace(go.Scatter(x=x_vals, y=y_fit, mode='lines', name='Fit'))
            fig.add_trace(go.Scatter(x=x_vals, y=y_ci[0], fill=None, mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=x_vals, y=y_ci[1], fill='tonexty', mode='lines', name='95% CI', line=dict(width=0)))
            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                plot_bgcolor='white',
                xaxis=dict(color='black', linecolor='black', showgrid=False),
                yaxis=dict(color='black', linecolor='black', showgrid=False)
            )
            st.plotly_chart(fig, use_container_width=True)

            combined_fig.add_trace(go.Scatter(x=x_vals, y=y_fit, mode='lines', name=f'{col} (TT={tt_val:.2f}, CFU={logcfu:.2f})'))
            combined_fig.add_trace(go.Scatter(x=x_vals, y=y_ci[0], fill=None, mode='lines', line=dict(width=0), showlegend=False))
            combined_fig.add_trace(go.Scatter(x=x_vals, y=y_ci[1], fill='tonexty', mode='lines', name=f'{col} 95% CI', line=dict(width=0)))

            st.session_state.summary_rows.append({
                'Sample': col,
                'Model': model_choice,
                'R²': round(r2, 4),
                'Threshold Time': tt_val,
                'Log CFU/mL': logcfu
            })

            fit_results[col] = pd.DataFrame({
                'Time': x_vals,
                'Fit': y_fit,
                'CI Lower': y_ci[0],
                'CI Upper': y_ci[1]
            })

        except Exception as e:
            st.sidebar.error(f"Error fitting {col}: {e}")

    # Show combined plot
    st.subheader("Combined Fit Plot")
    st.plotly_chart(combined_fig, use_container_width=True)
