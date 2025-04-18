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

# ----- EMAIL ALERT FUNCTION -----
def send_notification(username):
    try:
        msg = MIMEText(f"User '{username}' just accessed the 5PL web app on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
        msg['Subject'] = 'New 5PL App Session'
        msg['From'] = 'elisa.santovito@cnr.it'
        msg['To'] = 'elisa.santovito@cnr.it'

        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login('elisa.santovito@cnr.it', "your_email_app_password")  # Replace with actual app password or use secrets
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print("Email notification failed:", e)

# ----- PUBLIC ACCESS -----
if "rerun" in st.session_state and st.session_state.rerun:
    st.session_state.rerun = False
    st.stop()

if "login_log" not in st.session_state:
    st.session_state.login_log = []

user_email = st.text_input("Enter your email to begin:")
if st.button("Enter App"):
    if user_email:
        st.session_state.username = user_email
        st.session_state.login_log.append((user_email, str(datetime.datetime.now())))
        send_notification(user_email)
        st.session_state.rerun = True
    else:
        st.error("❌ Please enter your email")

if "username" not in st.session_state or not st.session_state.username:
    st.stop()

# ----- LOGOUT BUTTON -----
with st.sidebar:
    if st.button("🚪 Logout"):
        st.session_state.username = ""
        st.experimental_rerun()

# ----- APP LOGIC -----
st.title("📈 5PL Curve Fitting Web App")
st.markdown("Paste or enter your fluorescence/time data below. First column should be time (in hours), others are samples.")

# Sample structure
example_data = pd.DataFrame({
    "Time": np.arange(0, 4.25, 0.25),
    "Sample1": np.linspace(1, 25, 17),
    "Sample2": np.linspace(1.2, 24.5, 17)
})
data = st.data_editor(example_data, use_container_width=True, num_rows="dynamic")

# Threshold options
auto_thresh = st.checkbox("Auto threshold (50% of max)", value=True)
manual_thresh = st.number_input("Or enter manual threshold:", min_value=0.0, value=3.0, step=0.1)

# Define 5PL functions
def logistic_5pl(t, a, d, c, b, g):
    return d + (a - d) / (1 + (t / c)**b)**g

def inverse_5pl(y, a, d, c, b, g):
    try:
        base = ((a - d) / (y - d))**(1 / g) - 1
        return c * base**(1 / b)
    except:
        return np.nan

# Run fitting
if st.button("Run Analysis"):
    st.subheader("📊 Results")
    time = data.iloc[:, 0].dropna().values
    export_data = []

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
            mse = np.sum((y - y_fit) ** 2) / dof

            ci = []
            for i in range(len(t_fit)):
                dy_dx = np.array([
                    (logistic_5pl(t_fit[i], *(popt + np.eye(len(popt))[j]*1e-5)) - y_fit[i]) / 1e-5
                    for j in range(len(popt))
                ])
                se = np.sqrt(np.dot(dy_dx, np.dot(pcov, dy_dx)))
                delta = tval * se
                ci.append((y_fit[i] - delta, y_fit[i] + delta))

            threshold = (max(y_fit) * 0.5) if auto_thresh else manual_thresh
            t_thresh = inverse_5pl(threshold, a, d, c, b, g)

            st.markdown(f"**{col}**")
            st.write(f"- R²: {r2:.4f}")
            st.write(f"- Threshold: {threshold:.2f} ➜ Time ≈ {t_thresh:.2f} h")
            st.code(f"= {d:.6f} + ({a:.6f} - {d:.6f}) / (1 + (t / {c:.6f})^{b:.6f})^{g:.6f}", language='excel')

            chart_data = pd.DataFrame({
                "Raw": y,
                "Fit": y_fit[:len(y)],
                "CI Low": [ci[i][0] for i in range(len(y))],
                "CI High": [ci[i][1] for i in range(len(y))]
            }, index=t_fit)
            st.line_chart(chart_data)

            for i in range(len(y)):
                export_data.append({
                    "Sample": col,
                    "Time": t_fit[i],
                    "Raw": y[i],
                    "Fit": y_fit[i],
                    "CI Low": ci[i][0],
                    "CI High": ci[i][1]
                })
        except Exception as e:
            st.error(f"❌ Could not fit {col}: {e}")

    if export_data:
        df_export = pd.DataFrame(export_data)
        buffer = BytesIO()
        df_export.to_csv(buffer, index=False)
        st.download_button(
            label="📥 Download Fitted Data (.csv)",
            data=buffer.getvalue(),
            file_name="5PL_fitting_results.csv",
            mime="text/csv"
        )
