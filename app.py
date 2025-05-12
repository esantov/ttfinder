import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from io import BytesIO
import datetime
import matplotlib.pyplot as plt

# ----- SESSION STATE INITIALIZATION -----
if 'fits_5pl' not in st.session_state:
    st.session_state.fits_5pl = {}
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

# 5PL functions
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
    st.session_state.fits_lin.clear()
    st.session_state.summary_rows.clear()
    st.subheader("Individual Fits")
    time_vals = data.iloc[:,0].dropna().values

    for col in data.columns[1:]:
        y_vals = data[col].dropna().values
        t_fit = time_vals[:len(y_vals)]
        a = d = c = b = g = r2 = None

        # 5PL fit attempt
        try:
            popt, pcov = curve_fit(
                logistic_5pl, t_fit, y_vals,
                p0=[min(y_vals), max(y_vals), np.median(t_fit), 1, 1],
                maxfev=10000
            )
            a, d, c, b, g = popt
            y_fit = logistic_5pl(t_fit, *popt)
            r2 = r2_score(y_vals, y_fit)
            st.session_state.fits_5pl[col] = (t_fit, y_fit)

            st.markdown(f"**{col} – 5PL Fit** (R² = {r2:.4f})")
            fig, ax = plt.subplots()
            ax.plot(t_fit, y_vals, 'ko', label='Data')
            ax.plot(t_fit, y_fit, 'b-', label='5PL Fit')
            ax.axhline(manual_thresh, color='green', linestyle='--', label='Threshold')
            ax.set_xlabel(x_label); ax.set_ylabel(y_label); ax.legend()
            st.pyplot(fig)

            # Offer linear fallback if poor R²
            if r2 < 0.95:
                if st.button(f"Switch to Linear Fit for {col}", key=f"lin_r2_{col}"):
                    coef = np.polyfit(t_fit, y_vals, 1)
                    y_lin = np.polyval(coef, t_fit)
                    st.session_state.fits_lin[col] = (t_fit, y_lin)
                    st.markdown(f"**{col} – Linear Fit (R² too low)**: {r2:.3f} < 0.95")
                    fig2, ax2 = plt.subplots()
                    ax2.plot(t_fit, y_vals, 'ko', label='Data')
                    ax2.plot(t_fit, y_lin, 'm--', label='Linear Fit')
                    ax2.set_xlabel(x_label); ax2.set_ylabel(y_label); ax2.legend()
                    st.pyplot(fig2)

        except Exception as e:
            # 5PL failure -> always offer linear
            st.error(f"❌ Could not fit {col} with 5PL: {e}")
            if st.button(f"Perform Linear Fit for {col}", key=f"lin_err_{col}"):
                coef = np.polyfit(t_fit, y_vals, 1)
                y_lin = np.polyval(coef, t_fit)
                st.session_state.fits_lin[col] = (t_fit, y_lin)
                st.markdown(f"**{col} – Linear Fit (Fallback)**")
                fig3, ax3 = plt.subplots()
                ax3.plot(t_fit, y_vals, 'ko', label='Data')
                ax3.plot(t_fit, y_lin, 'm--', label='Linear Fit')
                ax3.set_xlabel(x_label); ax3.set_ylabel(y_label); ax3.legend()
                st.pyplot(fig3)

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

    # display summary
    st.subheader("Summary Table")
    st.dataframe(pd.DataFrame(st.session_state.summary_rows))

# ----- COMBINED PLOT -----
if st.session_state.fits_5pl or st.session_state.fits_lin:
    st.markdown("---")
    st.subheader("Combined Fits based on selections")
    fig_all, ax_all = plt.subplots(figsize=(8,5))
    for col, (tx, y5) in st.session_state.fits_5pl.items():
        ax_all.plot(tx, y5, '-', label=f"{col} 5PL")
    for col, (tx, ylin) in st.session_state.fits_lin.items():
        ax_all.plot(tx, ylin, '--', label=f"{col} Linear")
    ax_all.set_xlabel(x_label); ax_all.set_ylabel(y_label); ax_all.legend()
    st.pyplot(fig_all)
    buf_all = BytesIO(); fig_all.savefig(buf_all, format=fmt, dpi=dpi, bbox_inches='tight'); buf_all.seek(0)
    st.download_button("Download Combined Plot", buf_all.read(), file_name=f"combined_fits.{fmt}", mime=f"image/{{'svg+xml' if fmt=='svg' else fmt}}")
if st.session_state.fits_5pl or st.session_state.fits_lin:
    st.markdown("---")
    st.subheader("Combined Fits based on selections")
    fig_all, ax_all = plt.subplots(figsize=(8,5))
    for col, (tx, y5) in st.session_state.fits_5pl.items():
        ax_all.plot(tx, y5, '-', label=f"{col} 5PL")
    for col, (tx, ylin) in st.session_state.fits_lin.items():
        ax_all.plot(tx, ylin, '--', label=f"{col} Linear")
    ax_all.set_xlabel(x_label)
    ax_all.set_ylabel(y_label)
    ax_all.legend()
    st.pyplot(fig_all)
    buf_all = BytesIO(); fig_all.savefig(buf_all, format=fmt, dpi=dpi, bbox_inches='tight'); buf_all.seek(0)
    st.download_button("Download Combined Plot", buf_all.read(), file_name=f"combined_fits.{fmt}", mime=f"image/{{'svg+xml' if fmt=='svg' else fmt}}")

# ----- GLOBAL DOWNLOAD -----
st.download_button(
    "Download All Results (ZIP)",
    data=BytesIO().getvalue(),  # placeholder
    file_name=f"results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
    mime="application/zip"
)
