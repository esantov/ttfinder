import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import t
from scipy.optimize import root_scalar
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
        result = root_scalar(lambda t: model_func(t, *popt) - y,
                             bracket=[0, 1e3], method='brentq')
        return result.root if result.converged else None
    except:
        return None

# --- Delta‐method for TT standard error ---
def compute_tt_se(func, popt, pcov, threshold, eps=1e-6):
    """
    Approximate SE of the root t such that func(t, *popt)==threshold,
    using the delta‐method and the covariance matrix pcov.
    """
    tt = inverse_threshold_curve(threshold, func, popt)
    if tt is None or pcov is None:
        return None

    p = np.array(popt)
    # gradient wrt parameters
    jac_p = np.array([
        (func(tt, *(p + ei * eps)) - func(tt, *p)) / eps
        for ei in np.eye(len(p))
    ])
    # derivative wrt x
    f_plus  = func(tt + eps, *p)
    f_minus = func(tt - eps, *p)
    dfdx    = (f_plus - f_minus) / (2 * eps)

    # dt/dp = - jac_p / dfdx
    dt_dp = - jac_p / dfdx

    # variance propagation
    var_tt = dt_dp @ pcov @ dt_dp.T
    return float(np.sqrt(var_tt)) if var_tt > 0 else None

# --- Plotting & Export Functions ---
def generate_sample_plot(sample, df, x_label, y_label,
                         threshold, tt_val=None, tt_se=None, logcfu=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df['Time'], df['Raw'], 'o', label='Data', color='black')
    ax.plot(df['Time'], df['Fit'], label='Fit', color='blue')
    ax.fill_between(df['Time'], df['CI Lower'], df['CI Upper'],
                    color='red', alpha=0.1, label='95% CI')
    ax.axhline(y=threshold, color='green', linestyle='--', label='Threshold')
                             
    if tt_val is not None:
        # vertical line at TT
        ax.axvline(x=tt_val, color='orange', linestyle=':', label='TT')
        # draw green dot at (TT, threshold)
        ax.plot(tt_val, threshold,
                marker='o', color='green', label='TT')
        # horizontal error bar to show ±SE
        if tt_se is not None:
            ax.errorbar(tt_val, threshold,
                        xerr=tt_se,
                        fmt='none',
                        ecolor='green',
                        capsize=3,
                        label='TT ± SE')
            
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
                             
    title = f"{sample} Fit"
    if tt_val is not None and logcfu is not None:
        title += f" (TT: {tt_val:.2f}±{tt_se:.2f} h, CFU: {logcfu:.2f})"
    ax.set_title(title)
                             
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
                             
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    plt.close(fig)
    return buf

def generate_combined_plot(fit_results_dict, threshold,
                           x_label, y_label, summary_rows):
    fig, ax = plt.subplots(figsize=(10, 6))
    for row in summary_rows:
        sample = row['Sample']
        df     = fit_results_dict[sample]
        label  = (f"{sample} "
                  f"(TT={row['Threshold Time']:.2f}, "
                  f"±{row['TT SE']:.2f}, "
                  f"CFU={row['Log CFU/mL']:.2f})")
        ax.plot(df['Time'], df['Fit'], label=label)
        ax.fill_between(df['Time'], df['CI Lower'], df['CI Upper'],
                        alpha=0.3)
    ax.axhline(y=threshold, color='green', linestyle='--',
               label='Threshold')
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

def export_all_plots_zip(fit_results_dict, summary_rows,
                         x_label, y_label, threshold, combined_buf):
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        for row in summary_rows:
            sample = row['Sample']
            df     = fit_results_dict[sample]
            tt     = row['Threshold Time']
            tt_se  = row['TT SE']
            logcfu = row['Log CFU/mL']
            img = generate_sample_plot(sample, df,
                                       x_label, y_label,
                                       threshold,
                                       tt, tt_se, logcfu)
            z.writestr(f"{sample}.png", img.getvalue())
        z.writestr("combined.png", combined_buf.getvalue())
    buf.seek(0)
    return buf

def create_excel_report(data, fit_results, fit_params,
                        summary_rows, calibration,
                        x_label, y_label):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Original data
        if not data.empty:
            data.to_excel(writer, sheet_name="Original Data",
                          index=False)
        # Calibration
        if calibration:
            (a, b), _ = calibration
            pd.DataFrame({
                "Calibration Name": ["Manual"],
                "Slope": [a],
                "Intercept": [b]
            }).to_excel(writer, sheet_name="Calibration",
                       index=False)
        # Summary (now includes TT SE)
        if summary_rows:
            pd.DataFrame(summary_rows).to_excel(
                writer, sheet_name="Summary", index=False
            )

        # Fit parameters sheet
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

        param_rows = []
        for row in summary_rows:
            sample = row['Sample']
            model  = row['Model']
            popt, pcov = fit_params.get(sample, ([], None))
            entry = {
                "Sample": sample,
                "Model": model,
                "Formula": formula_map.get(model, ""),
                "Inverse": inverse_map.get(model, "")
            }
            for i, v in enumerate(popt):
                entry[f"p{i+1}"] = round(v, 4)
            param_rows.append(entry)

        if param_rows:
            pd.DataFrame(param_rows).to_excel(
                writer, sheet_name="Fit Parameters", index=False
            )

        # One sheet per sample
        for sample, df in fit_results.items():
            df.to_excel(writer, sheet_name=sample[:31], index=False)

    output.seek(0)
    return output

# --- UI ---
st.title("📈 TT Finder - Curve Fitting Tool")
x_label       = st.text_input("X-axis label", "Time (h)")
y_label       = st.text_input("Y-axis label", "Signal")
manual_thresh = st.number_input("Threshold", 0.0, 100.0, 3.0, 0.1)
dpi           = st.slider("Plot DPI", 100, 600, 300, 50)

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
    st.session_state['summary_rows'] = []  # reset on new file
    st.dataframe(data.head())
else:
    data = pd.DataFrame({"Time": []})

if 'summary_rows' not in st.session_state:
    st.session_state['summary_rows'] = []

fit_results = {}
fit_params  = {}

# --- Analysis Logic ---
if not data.empty and len(data.columns) > 1:
    time = data.iloc[:, 0].dropna().values

    for col in data.columns[1:]:
        y = data[col].dropna().values
        x = time[:len(y)]

        with st.expander(f"{col}"):
            model = st.selectbox("Model",
                ["5PL", "4PL", "Sigmoid", "Linear"], key=col)

            if model == "5PL":
                func, p0 = logistic_5pl, [min(y), max(y), np.median(x), 1, 1]
            elif model == "4PL":
                func, p0 = logistic_4pl, [min(y), max(y), np.median(x), 1]
            elif model == "Sigmoid":
                func, p0 = sigmoid, [max(y), np.median(x), 1]
            else:
                func = lambda x, a, b: a * x + b
                p0 = None

            try:
                if model == "Linear":
                    popt = np.polyfit(x, y, 1)
                    y_fit  = np.polyval(popt, x)
                    ci_low = y_fit
                    ci_high= y_fit
                    pcov   = None
                else:
                    popt, pcov = curve_fit(
                        func, x, y, p0=p0, maxfev=10000
                    )
                    y_fit = func(x, *popt)
                    dof   = len(x) - len(popt)
                    tval  = t.ppf(0.975, dof)
                    ci_low, ci_high = [], []
                    for i, xi in enumerate(x):
                        grad = np.array([
                            (func(xi, *(popt + np.eye(len(popt))[j]*1e-5))
                             - y_fit[i]) / 1e-5
                            for j in range(len(popt))
                        ])
                        se    = np.sqrt(grad @ pcov @ grad.T)
                        delta = tval * se
                        ci_low.append(y_fit[i] - delta)
                        ci_high.append(y_fit[i] + delta)

                # compute TT and its SE
                tt_val = inverse_threshold_curve(manual_thresh, func, popt)
                tt_se  = compute_tt_se(func, popt, pcov, manual_thresh)

                # CFU conversion
                logcfu = None
                if tt_val is not None and 'calibration_coef' in st.session_state:
                    a, b = st.session_state['calibration_coef'][0]
                    logcfu = a * tt_val + b

                # metrics & store
                r2 = r2_score(y, y_fit)
                fit_df = pd.DataFrame({
                    'Time':     x,
                    'Raw':      y,
                    'Fit':      y_fit,
                    'CI Lower': np.array(ci_low),
                    'CI Upper': np.array(ci_high)
                })

                fit_results[col] = fit_df
                fit_params[col]  = (popt, pcov)
                st.session_state['summary_rows'].append({
                    'Sample':          col,
                    'Model':           model,
                    'R²':              round(r2, 3),
                    'Threshold Time':  tt_val,
                    'TT SE':           tt_se,
                    'Log CFU/mL':      logcfu
                })

                # sample plot with TT ± SE
                img_buf = generate_sample_plot(
                    col, fit_df, x_label, y_label,
                    manual_thresh, tt_val, tt_se, logcfu
                )
                st.image(img_buf, caption=f"{col} Fit", use_container_width=True)

            except Exception as e:
                st.error(f"❌ Fitting failed for {col}: {e}")

    # combined plot inline
    combined_buf = generate_combined_plot(
        fit_results, manual_thresh, x_label, y_label,
        st.session_state['summary_rows']
    )
    st.subheader("Combined Fit Plot")
    st.image(combined_buf, caption="Combined Fit", use_container_width=True)

    # download buttons
    st.download_button(
        "📦 Download All Plots (ZIP)",
        data=export_all_plots_zip(
            fit_results, st.session_state['summary_rows'],
            x_label, y_label, manual_thresh, combined_buf
        ),
        file_name=f"tt_finder_plots_{datetime.datetime.now():%Y%m%d_%H%M%S}.zip",
        mime="application/zip"
    )

    excel_buf = create_excel_report(
        data,
        fit_results,
        fit_params,
        st.session_state['summary_rows'],
        st.session_state.get('calibration_coef'),
        x_label,
        y_label
    )
    st.download_button(
        "📥 Download Excel Report",
        data=excel_buf,
        file_name=f"tt_finder_report_{datetime.datetime.now():%Y%m%d_%H%M%S}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
