import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from io import BytesIO
import datetime
import matplotlib.pyplot as plt
from scipy.stats import t

# ----- SESSION STATE INITIALIZATION -----
if 'fits_5pl' not in st.session_state:
    st.session_state.fits_5pl = {}
if 'ci_5pl' not in st.session_state:
    st.session_state.ci_5pl = {}
if 'fits_lin' not in st.session_state:
    st.session_state.fits_lin = {}
if 'summary_rows' not in st.session_state:
    st.session_state.summary_rows = []

# ----- USER INPUT -----
x_label = st.text_input("X-axis label", value="Time (h)")
y_label = st.text_input("Y-axis label", value="Signal")

st.title("📈 5PL Curve Fitting Web App")
st.markdown(
    "Paste or enter your fluorescence/time data below. First column should be time (in hours), others are samples."
)

num_samples = st.number_input(
    "Number of samples?", min_value=1, max_value=20, value=2, step=1
)

# Example template
time_template = np.arange(0, 4.25, 0.25)
example_data = pd.DataFrame({"Time": time_template})
for i in range(1, num_samples + 1):
    example_data[f"Sample{i}"] = np.linspace(1 + i, 25 - i, len(time_template))

uploaded = st.file_uploader("Upload CSV (first column=Time)", type="csv")
if uploaded:
    data = pd.read_csv(
        uploaded,
        true_values=['TRUE','True','true'],
        false_values=['FALSE','False','false','fALSE']
    )
    data = data.apply(pd.to_numeric, errors='coerce')
    st.success("✅ Data loaded from file")
else:
    data = st.data_editor(example_data, num_rows="dynamic", use_container_width=True)

manual_thresh = st.number_input("Manual threshold", 0.0, 100.0, 3.0, 0.1)
fmt = st.selectbox("Image format", ["png","jpeg","svg","pdf"])
dpi = st.slider("DPI", 100, 600, 300, 50)

# 5PL logistic functions
def logistic_5pl(t, a, d, c, b, g):
    return d + (a - d) / (1 + (t / c)**b)**g

def inverse_5pl(y, a, d, c, b, g):
    try:
        base = ((a - d)/(y - d))**(1/g) - 1
        return c * base**(1/b)
    except:
        return np.nan

# ----- ANALYSIS -----
if st.button("Run Analysis"):
    # clear previous
    st.session_state.fits_5pl.clear()
    st.session_state.ci_5pl.clear()
    st.session_state.fits_lin.clear()
    st.session_state.summary_rows.clear()
    st.subheader("Individual Fits")
    time_vals = data.iloc[:,0].dropna().values

    for col in data.columns[1:]:
        y_vals = data[col].dropna().values
        t_fit = time_vals[:len(y_vals)]
        a = d = c = b = g = r2 = None

        # Try 5PL fit
        try:
            popt, pcov = curve_fit(
                logistic_5pl, t_fit, y_vals,
                p0=[min(y_vals), max(y_vals), np.median(t_fit), 1, 1],
                maxfev=10000
            )
            a, d, c, b, g = popt
            y_fit = logistic_5pl(t_fit, *popt)
            r2 = r2_score(y_vals, y_fit)

            # Compute 95% CI
            dof = max(len(t_fit) - len(popt), 1)
            alpha = 0.05
            tval = t.ppf(1.0 - alpha/2., dof)
            ci_low = []
            ci_high = []
            for i, xi in enumerate(t_fit):
                grad = np.array([
                    (logistic_5pl(xi, *(popt + np.eye(len(popt))[j]*1e-5)) - y_fit[i]) / 1e-5
                    for j in range(len(popt))
                ])
                se = np.sqrt(grad.dot(pcov).dot(grad))
                delta = tval * se
                ci_low.append(y_fit[i] - delta)
                ci_high.append(y_fit[i] + delta)
            ci_low = np.array(ci_low)
            ci_high = np.array(ci_high)

            st.session_state.fits_5pl[col] = (t_fit, y_fit)
            st.session_state.ci_5pl[col] = (ci_low, ci_high)

            # Plot with CI as red band
            st.markdown(f"**{col} – 5PL Fit** (R² = {r2:.4f})")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(t_fit, y_vals, 'ko', label='Raw Data')
            ax.plot(t_fit, y_fit, 'b-', label='5PL Fit')
            ax.fill_between(t_fit, ci_low, ci_high, color='red', alpha=0.2, label='95% CI')
            ax.axhline(manual_thresh, color='green', linestyle='--', linewidth=1, label='Threshold')
            ax.set_title(f"{col} – 5PL Fit")
            ax.set_xlabel(x_label, fontweight='bold')
            ax.set_ylabel(y_label, fontweight='bold')
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            # On 5PL fit error -> linear fallback
            st.error(f"❌ 5PL failed for {col}, using linear fallback: {e}")
            # Linear regression
            coef = np.polyfit(t_fit, y_vals, 1)
            y_lin = np.polyval(coef, t_fit)
            # Compute linear 95% CI
            n = len(t_fit)
            x_mean = np.mean(t_fit)
            Sxx = np.sum((t_fit - x_mean)**2)
            residuals = y_vals - y_lin
            sigma2 = np.sum(residuals**2) / (n - 2)
            tval_lin = t.ppf(0.975, n - 2)
            ci_lin_low = []
            ci_lin_high = []
            for xi, yhat in zip(t_fit, y_lin):
                se_pred = np.sqrt(sigma2 * (1/n + (xi - x_mean)**2 / Sxx))
                delta = tval_lin * se_pred
                ci_lin_low.append(yhat - delta)
                ci_lin_high.append(yhat + delta)
            ci_lin_low = np.array(ci_lin_low)
            ci_lin_high = np.array(ci_lin_high)
            st.session_state.fits_lin[col] = (t_fit, y_lin)
            st.session_state.ci_lin[col] = (ci_lin_low, ci_lin_high)

            # Plot linear fallback with CI bands
            st.markdown(f"**{col} – Linear Fallback Fit**")
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(t_fit, y_vals, 'ko', label='Data')
            ax.plot(t_fit, y_lin, 'b-', label='Linear Fallback')
            ax.fill_between(t_fit, ci_lin_low, ci_lin_high, color='red', alpha=0.2, label='95% CI')
            ax.axhline(manual_thresh, color='green', linestyle='--', label='Threshold')
            ax.set_xlabel(x_label); ax.set_ylabel(y_label); ax.legend()
            st.pyplot(fig)

        # summary row
        thresh_time = None
        if a is not None:
            thresh_time = inverse_5pl(manual_thresh, a, d, c, b, g)
        st.session_state.summary_rows.append({
            'Sample': col,
            'a': a, 'd': d, 'c': c, 'b': b, 'g': g,
            'R²': r2,
            'Threshold Time': thresh_time
        })

    # show summary
    st.subheader("Summary Table")
    st.dataframe(pd.DataFrame(st.session_state.summary_rows))

# Combined plot
if st.session_state.fits_5pl or st.session_state.fits_lin:
    st.markdown("---")
    st.subheader("Combined Fits")
    # Use original styling: 8x8 size, raw data in black circles,
    # 5PL fit in blue solid, CI in red dashed, threshold in green dashed
    fig_all, ax_all = plt.subplots(figsize=(8, 8))
    for col in data.columns[1:]:
        t_fit, y_vals = None, None
        # plot raw data
        y_raw = data[col].dropna().values
        t_raw = data.iloc[:,0].dropna().values[:len(y_raw)]
        ax_all.plot(t_raw, y_raw, 'ko', label=f'{col} Data' if col == data.columns[1] else "")
        # if 5PL fit exists
        if col in st.session_state.fits_5pl:
            tx, y5 = st.session_state.fits_5pl[col]
            ax_all.plot(tx, y5, 'b-', label=f'{col} 5PL')
            ci_low, ci_high = st.session_state.ci_5pl.get(col, (None, None))
            if ci_low is not None:
                ax_all.plot(tx, ci_low, 'r--', linewidth=1, label=f'{col} 95% CI')
                ax_all.plot(tx, ci_high, 'r--', linewidth=1)
            # threshold line only once
            if col == data.columns[1]:
                ax_all.axhline(manual_thresh, color='green', linestyle='--', linewidth=1, label='Threshold')
        # if linear fallback exists
        if col in st.session_state.fits_lin:
            tx, ylin = st.session_state.fits_lin[col]
            ax_all.plot(tx, ylin, 'b--', label=f'{col} Linear')
    ax_all.set_xlabel(x_label, fontweight='bold')
    ax_all.set_ylabel(y_label, fontweight='bold')
    ax_all.legend()
    st.pyplot(fig_all)
    # Download combined plot
    buf_all = BytesIO()
    fig_all.savefig(buf_all, format=fmt, dpi=dpi, bbox_inches='tight')
    buf_all.seek(0)
    st.download_button(
        "📥 Download Combined Plot", buf_all.read(),
        file_name=f"combined_fits.{fmt}", mime=f"image/{{'svg+xml' if fmt=='svg' else fmt}}"
    )

# Global download
st.download_button(
    "Download Summary CSV",
    data=pd.DataFrame(st.session_state.summary_rows).to_csv(index=False).encode(),
    file_name=f"summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)
