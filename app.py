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

# SESSION STATE INIT
for key in ['fits', 'ci', 'summary_rows', 'model_choices', 'calibration_coef']:
    if key not in st.session_state:
        st.session_state[key] = {}

# ----- MODEL DEFINITIONS -----
def logistic_5pl(x, a, d, c, b, g):
    return d + (a - d) / (1 + (x / c) ** b) ** g

def logistic_4pl(x, a, d, c, b):
    return d + (a - d) / (1 + (x / c) ** b)

def sigmoid(x, L ,x0, k):
    return L / (1 + np.exp(-k * (x - x0)))

def inverse_threshold_curve(y, model_func, popt):
    try:
        from scipy.optimize import root_scalar
        result = root_scalar(lambda t: model_func(t, *popt) - y, bracket=[0, 1e3], method='brentq')
        return result.root if result.converged else None
    except:
        return None

# ----- USER INPUT -----
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

# Calibration input (optional)
st.markdown("### Calibration")
calib_name = st.text_input("Calibration name", value="Default Calibration")
cal_slope = st.number_input("Calibration slope (a)", value=0.0)
cal_intercept = st.number_input("Calibration intercept (b)", value=0.0)

manual_calib = st.checkbox("Use manual calibration", value=False)

if manual_calib:
    coef = [cal_slope, cal_intercept]
    cov = np.array([[0.01, 0.0], [0.0, 0.01]])
    st.session_state.calibration_coef = (coef, cov)
    st.success(f"Manual calibration model: logCFU/mL = {coef[0]:.4f}*TT + {coef[1]:.4f}")
else:
    cal_df = st.file_uploader("Upload calibration CSV (TT, logCFU/mL)", type="csv", key="calib")
    if cal_df:
        calib_data = pd.read_csv(cal_df)
        if calib_data.shape[1] >= 2:
            tt = calib_data.iloc[:, 0]
            logcfu = calib_data.iloc[:, 1]
            coef, cov = np.polyfit(tt, logcfu, 1, cov=True)
            st.session_state.calibration_coef = (coef, cov)
            st.success(f"Calibration model: logCFU/mL = {coef[0]:.4f}*TT + {coef[1]:.4f}")

# Data input
st.markdown("### Data Input")
uploaded = st.file_uploader("Upload data CSV (first column = Time)", type="csv")

if uploaded:
    data = pd.read_csv(uploaded)
    st.success("Data loaded from file.")
else:
    example_time = np.arange(0, 4.25, 0.25)
    example_data = pd.DataFrame({
        "Time": example_time,
        "Sample1": np.linspace(2, 20, len(example_time)),
        "Sample2": np.linspace(3, 25, len(example_time))
    })
    st.info("Paste or edit your data below:")
    data = st.data_editor(example_data, num_rows="dynamic", use_container_width=True)

st.session_state.summary_rows.clear()
fit_results = {}
combined_fig = go.Figure()

if not data.empty:
    time_vals = data.iloc[:, 0].dropna().values
    st.sidebar.markdown("### Fit Errors")

    for col in data.columns[1:]:
        y_vals = data[col].dropna().values
        x_vals = time_vals[:len(y_vals)]
        thresh_time, ci_thresh, logcfu, logcfu_ci = None, (None, None), None, (None, None)

        with st.expander(f"{col} – Analysis"):
            model_choice = st.selectbox(f"Select model for {col}", ["5PL", "4PL", "Sigmoid", "Linear"], key=f"model_{col}")
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

                thresh_time = inverse_threshold_curve(manual_thresh, model_func if model_choice != "Linear" else lambda t, a, b: a*t + b, popt)

                if thresh_time is not None and 'calibration_coef' in st.session_state:
                    (a, b), cov = st.session_state.calibration_coef
                    logcfu = a * thresh_time + b
                    se_pred = np.sqrt(cov[0,0]*thresh_time**2 + cov[1,1] + 2*thresh_time*cov[0,1])
                    tval_calib = t.ppf(0.975, 10)
                    delta_logcfu = tval_calib * se_pred
                    logcfu_ci = (logcfu - delta_logcfu, logcfu + delta_logcfu)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers', name='Raw Data'))
                fig.add_trace(go.Scatter(x=x_vals, y=y_fit, mode='lines', name=f'{model_choice} Fit'))
                fig.add_trace(go.Scatter(x=x_vals, y=y_ci[0], fill=None, mode='lines', line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=x_vals, y=y_ci[1], fill='tonexty', mode='lines', name='95% CI', line=dict(width=0)))
                fig.add_hline(y=manual_thresh, line_dash="dash", line_color="green")
                fig.update_layout(title=f"{col} – {model_choice} Fit\nTT: {thresh_time:.2f} | LogCFU/mL: {logcfu:.2f}" if thresh_time and logcfu else f"{col} – {model_choice} Fit", xaxis_title=x_label, yaxis_title=y_label)
                st.plotly_chart(fig, use_container_width=True)

                combined_fig.add_trace(go.Scatter(x=x_vals, y=y_fit, mode='lines', name=f'{col} Fit'))
                combined_fig.add_trace(go.Scatter(x=x_vals, y=y_ci[0], fill=None, mode='lines', line=dict(width=0), showlegend=False))
                combined_fig.add_trace(go.Scatter(x=x_vals, y=y_ci[1], fill='tonexty', mode='lines', name=f'{col} 95% CI', line=dict(width=0)))

                st.session_state.summary_rows.append({
                    'Sample': col,
                    'Model': model_choice,
                    'R²': round(r2, 4),
                    'Threshold Time': thresh_time,
                    'Log CFU/mL': logcfu,
                    'Log CFU/mL CI': f"{logcfu_ci[0]:.3f}–{logcfu_ci[1]:.3f}" if logcfu_ci[0] is not None else None
                })

                fit_results[col] = pd.DataFrame({
                    'Time': x_vals,
                    'Fit': y_fit,
                    'CI Lower': y_ci[0],
                    'CI Upper': y_ci[1]
                })

            except Exception as e:
                st.sidebar.error(f"❌ {col}: {str(e)}")
                continue

    # Display summary
    summary_df = pd.DataFrame(st.session_state.summary_rows)
    st.subheader("Summary Table")
    if not summary_df.empty:
        st.dataframe(summary_df)

        # Cumulative Plot
        st.subheader("Combined Fit Plot")
        combined_fig.update_layout(title="Combined Model Fits", xaxis_title=x_label, yaxis_title=y_label)
        st.plotly_chart(combined_fig, use_container_width=True)

        # Excel export
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            data.to_excel(writer, index=False, sheet_name="Raw Data")
            summary_df.to_excel(writer, index=False, sheet_name="Summary")
            if manual_calib:
                calib_df = pd.DataFrame({
                    'Calibration Name': [calib_name],
                    'Slope': [cal_slope],
                    'Intercept': [cal_intercept]
                })
            elif 'calib_data' in locals():
                calib_df = calib_data
            else:
                calib_df = pd.DataFrame()
            if not calib_df.empty:
                calib_df.to_excel(writer, index=False, sheet_name="Calibration")
            for name, df in fit_results.items():
                df.to_excel(writer, index=False, sheet_name=name[:31])
            combined_data = pd.concat(fit_results.values(), keys=fit_results.keys()).reset_index()
            combined_data.to_excel(writer, index=False, sheet_name="Combined Fits")
        excel_buffer.seek(0)
        st.download_button("Download All Results (Excel)", data=excel_buffer.read(), file_name="tt_finder_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.subheader("Selected Models")
    model_df = pd.DataFrame.from_dict(st.session_state.model_choices, orient='index', columns=['Model'])
    st.dataframe(model_df.reset_index().rename(columns={'index': 'Sample'}))
