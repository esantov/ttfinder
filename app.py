import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
from scipy.stats import t
import smtplib
from email.mime.text import MIMEText
from io import BytesIO
import datetime
import matplotlib.pyplot as plt
import os
from zipfile import ZipFile
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# ----- PUBLIC ACCESS -----
if "rerun" in st.session_state and st.session_state.rerun:
    st.session_state.rerun = False
    st.stop()

    st.session_state.login_log = []


    st.stop()


# ----- APP LOGIC -----

x_label = st.text_input("X-axis label", value="Time (h)")
y_label = st.text_input("Y-axis label", value="Signal")

st.title("📈 5PL Curve Fitting Web App")
st.markdown("Paste or enter your fluorescence/time data below. First column should be time (in hours), others are samples.")

num_samples = st.number_input("How many samples do you want to enter?", min_value=1, max_value=20, value=2, step=1)

sample_data = {"Time": np.arange(0, 4.25, 0.25)}
labels = []
for i in range(1, num_samples + 1):
    default_label = f"Sample{i}"
    label = st.text_input(f"Label for Sample {i}", value=default_label)
    if label in labels:
        st.warning(f"⚠️ Duplicate label '{label}' found. Please choose a unique name.")
    labels.append(label)
    sample_data[label] = np.linspace(1 + i, 25 - i, 17)

example_data = pd.DataFrame(sample_data)

uploaded_file = st.file_uploader("Upload a CSV file (wide format, first column = Time)", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Data loaded from file")
else:
    data = st.data_editor(example_data, use_container_width=True, num_rows="dynamic")

auto_thresh = st.checkbox("Auto threshold (10% of max)", value=True)
manual_thresh = st.number_input("Or enter manual threshold:", min_value=0.0, value=3.0, step=0.1)

fmt = st.selectbox("Select image format for download", options=["png", "jpeg", "svg", "pdf"], index=0)
dpi = st.slider("Image resolution (DPI)", min_value=100, max_value=600, value=300, step=50)

# 5PL functions
def logistic_5pl(t, a, d, c, b, g):
    return d + (a - d) / (1 + (t / c)**b)**g

def inverse_5pl(y, a, d, c, b, g):
    try:
        base = ((a - d) / (y - d))**(1 / g) - 1
        return c * base**(1 / b)
    except:
        return np.nan

all_figs = []
all_csv_rows = []
all_formulas = []
zip_buffer = BytesIO()

if st.button("Run Analysis"):
    st.subheader("📊 Results")
    time = data.iloc[:, 0].dropna().values

    for col in data.columns[1:]:
        y = data[col].dropna().values
        t_fit = time[:len(y)]
        try:
            popt, pcov = curve_fit(logistic_5pl, t_fit, y, p0=[min(y), max(y), np.median(t_fit), 1, 1], maxfev=10000)
            y_fit = logistic_5pl(t_fit, *popt)
            r2 = r2_score(y, y_fit)
            a, d, c, b, g = popt

            dof = max(0, len(t_fit) - len(popt))
            alpha = 0.05
            tval = t.ppf(1.0 - alpha / 2., dof)
            mse = np.sum((y - y_fit)**2) / dof

            ci = []
            pi = []
            for i in range(len(t_fit)):
                dy_dx = np.array([
                    (logistic_5pl(t_fit[i], *(popt + np.eye(len(popt))[j]*1e-5)) - y_fit[i]) / 1e-5
                    for j in range(len(popt))
                ])
                se = np.sqrt(np.dot(dy_dx, np.dot(pcov, dy_dx)))
                delta = tval * se
                ci.append((y_fit[i] - delta, y_fit[i] + delta))
                pi.append((y_fit[i] - delta*np.sqrt(1 + 1/len(t_fit)), y_fit[i] + delta*np.sqrt(1 + 1/len(t_fit))))

            threshold = (max(y_fit) * 0.5) if auto_thresh else manual_thresh
            t_thresh = inverse_5pl(threshold, a, d, c, b, g)

            st.markdown(f"**{col}**")
            st.write(f"- R²: {r2:.4f}")
            st.write(f"- Threshold: {threshold:.2f} ➜ Time ≈ {t_thresh:.2f} h")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(t_fit, y, 'ko', label="Raw Data")
            ax.plot(t_fit, y_fit, 'b-', label="5PL Fit")
            ci_low, ci_high = zip(*ci)
            pi_low, pi_high = zip(*pi)
            ax.plot(t_fit, ci_low, 'b--', linewidth=1, label="95% CI")
            ax.plot(t_fit, ci_high, 'b--', linewidth=1)
            ax.plot(t_fit, pi_low, 'r:', linewidth=1, label="95% PI")
            ax.plot(t_fit, pi_high, 'r:', linewidth=1)
            ax.axhline(threshold, color='red', linestyle='--', linewidth=1, label="Threshold")
            ax.set_title(f"{col} Fit")
            ax.set_xlabel(x_label, fontweight='bold')
            ax.set_ylabel(y_label, fontweight='bold')
            ax.legend()
            ax.grid(False)
            st.pyplot(fig)

            # Save for ZIP
            buf = BytesIO()
            fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight')
            buf.seek(0)
            all_figs.append((f"{col}_fit_plot.{fmt}", buf.read()))

            all_csv_rows.append([col, a, d, c, b, g, r2, t_thresh])
            all_formulas.append([col,
                f"= {d:.6f} + ({a:.6f} - {d:.6f}) / (1 + (t / {c:.6f})^{b:.6f})^{g:.6f}",
                f"= {c:.6f} * ((({a:.6f} - {d:.6f}) / (y - {d:.6f}))^(1/{g:.6f}) - 1)^(1/{b:.6f})"])

        except Exception as e:
            st.error(f"❌ Could not fit {col}: {e}")

    if all_figs:
        for name, image_bytes in all_figs:
            st.download_button(
                label=f"📥 Download {name}",
                data=image_bytes,
                file_name=name,
                mime=f"image/{'svg+xml' if fmt=='svg' else fmt}"
            )

        df_csv = pd.DataFrame(all_csv_rows, columns=["Sample", "a", "d", "c", "b", "g", "R2", "Threshold Time"])
        st.download_button(
            label="📄 Download Fitting Parameters (CSV)",
            data=df_csv.to_csv(index=False),
            file_name="fitting_parameters.csv",
            mime="text/csv"
        )

        df_formulas = pd.DataFrame(all_formulas, columns=["Sample", "Excel 5PL", "Inverse 5PL"])
        st.download_button(
            label="📄 Download Excel Formulas (CSV)",
            data=df_formulas.to_csv(index=False),
            file_name="excel_formulas.csv",
            mime="text/csv"
        )

        # Optional: ZIP download button
        with ZipFile(zip_buffer, 'w') as zipf:
            for name, image_bytes in all_figs:
                zipf.writestr(name, image_bytes)
            zipf.writestr("fitting_parameters.csv", df_csv.to_csv(index=False))
            zipf.writestr("excel_formulas.csv", df_formulas.to_csv(index=False))

        zip_buffer.seek(0)
        st.download_button(
            label="📦 Download All Results (ZIP)",
            data=zip_buffer,
            file_name="5pl_fitting_outputs.zip",
            mime="application/zip"
        )
