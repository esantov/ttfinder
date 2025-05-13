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

# --- UI & Inputs ---
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

# --- Plotting Functions ---
def generate_sample_plot(sample, df, x_label, y_label, threshold, tt_val=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df['Time'], df['Raw'], 'o', label='Data', color='black')
    ax.plot(df['Time'], df['Fit'], label='Fit', color='blue')
    ax.fill_between(df['Time'], df['CI Lower'], df['CI Upper'], color='red', alpha=0.1, label='95% CI')
    ax.axhline(y=threshold, color='green', linestyle='--', label='Threshold')
    if tt_val is not None:
        ax.axvline(x=tt_val, color='orange', linestyle=':', label='TT')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{sample} Fit")
    ax.legend()
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    return buf

def generate_combined_plot(fit_results_dict, threshold, x_label, y_label):
    fig, ax = plt.subplots(figsize=(10, 6))
    for sample, df in fit_results_dict.items():
        ax.plot(df['Time'], df['Fit'], label=f"{sample} Fit")
        ax.fill_between(df['Time'], df['CI Lower'], df['CI Upper'], alpha=0.3, label=f"{sample} CI")
    ax.axhline(y=threshold, color='green', linestyle='--', label='Threshold')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Combined Fit Plot")
    ax.legend()
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    return buf

def export_all_plots_zip(fit_results, summary_rows, x_label, y_label, threshold, combined_img_buf):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for row in summary_rows:
            sample = row['Sample']
            model = row['Model']
            tt_val = row['Threshold Time']
            df = fit_results[sample]
            img = generate_sample_plot(sample, df, x_label, y_label, threshold, tt_val)
            zip_file.writestr(f"{sample}_{model}_fit.png", img.read())
        zip_file.writestr("combined_fit.png", combined_img_buf.read())
    zip_buffer.seek(0)
    return zip_buffer

# --- Analysis Loop ---
if not data.empty and len(data.columns) > 1:
    time = data.iloc[:, 0].dropna().values
    for col in data.columns[1:]:
        y = data[col].dropna().values
        x = time[:len(y)]

        with st.expander(f"{col}"):
            model = st.selectbox("Model", ["5PL", "4PL", "Sigmoid", "Linear"], key=col)

            if model == "5PL":
                func = logistic_5pl
                p0 = [min(y), max(y), np.median(x), 1, 1]
            elif model == "4PL":
                func = logistic_4pl
                p0 = [min(y), max(y), np.median(x), 1]
            elif model == "Sigmoid":
                func = sigmoid
                p0 = [max(y), np.median(x), 1]
            else:
                func = lambda x, a, b: a * x + b
                p0 = None

            try:
                if model == "Linear":
                    popt = np.polyfit(x, y, 1)
                    y_fit = np.polyval(popt, x)
                    y_ci = (y_fit, y_fit)
                else:
                    popt, pcov = curve_fit(func, x, y, p0=p0, maxfev=10000)
                    y_fit = func(x, *popt)
                    dof = len(x) - len(popt)
                    tval = t.ppf(0.975, dof)
                    ci_low, ci_high = [], []
                    for i, xi in enumerate(x):
                        grad = np.array([(func(xi, *(popt + np.eye(len(popt))[j]*1e-5)) - y_fit[i]) / 1e-5 for j in range(len(popt))])
                        se = np.sqrt(grad @ pcov @ grad.T)
                        delta = tval * se
                        ci_low.append(y_fit[i] - delta)
                        ci_high.append(y_fit[i] + delta)
                    y_ci = (np.array(ci_low), np.array(ci_high))

                r2 = r2_score(y, y_fit)
                tt_val = inverse_threshold_curve(manual_thresh, func, popt)
                logcfu = None
                if tt_val and 'calibration_coef' in st.session_state:
                    a, b = st.session_state['calibration_coef'][0]
                    logcfu = a * tt_val + b

                fit_df = pd.DataFrame({'Time': x, 'Raw': y, 'Fit': y_fit, 'CI Lower': y_ci[0], 'CI Upper': y_ci[1]})
                fit_results[col] = fit_df
                st.session_state.summary_rows.append({
                    'Sample': col,
                    'Model': model,
                    'R²': round(r2, 3),
                    'Threshold Time': tt_val,
                    'Log CFU/mL': logcfu
                })

                img_buf = generate_sample_plot(col, fit_df, x_label, y_label, manual_thresh, tt_val)
                st.image(img_buf, caption=f"{col} Fit", use_container_width=True)

            except Exception as e:
                st.error(f"❌ Fitting failed for {col}: {e}")

    # --- Final Export ---
    st.subheader("📦 Export All Plots")
    combined_buf = generate_combined_plot(fit_results, manual_thresh, x_label, y_label)
    zip_buf = export_all_plots_zip(fit_results, st.session_state.summary_rows, x_label, y_label, manual_thresh, combined_buf)

    st.download_button(
        "Download All Plots (ZIP)",
        data=zip_buf,
        file_name=f"tt_finder_plots_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip"
    )
if data is not None and summary_rows:
    excel_buf = create_excel_report(
        data=data,
        fit_results=fit_results,
        summary_rows=st.session_state['summary_rows'],
        calibration=st.session_state.get('calibration_coef'),
        x_label=x_label,
        y_label=y_label
    )

    st.download_button(
        label="📥 Download Excel Report",
        data=excel_buf,
        file_name=f"tt_finder_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
