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
        server.login('elisa.santovito@cnr.it', "your_email_app_password")  # Replace with actual app password or use env variable
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print("Email notification failed:", e)

# ----- PUBLIC ACCESS -----
if "rerun" in st.session_state and st.session_state.rerun:
    st.session_state.rerun = False
    st.stop()  # allow Streamlit to rerun naturally

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

x_label = st.text_input("X-axis label", value="Time (h)")
y_label = st.text_input("Y-axis label", value="Signal")

import matplotlib.pyplot as plt

st.title("📈 5PL Curve Fitting Web App")
st.markdown("Paste or enter your fluorescence/time data below. First column should be time (in hours), others are samples.")

example_data = pd.DataFrame({
    "Time": np.arange(0, 4.25, 0.25),
    "Sample1": np.linspace(1, 25, 17),
    "Sample2": np.linspace(1.2, 24.5, 17)
})
data = st.data_editor(example_data, use_container_width=True, num_rows="dynamic")

auto_thresh = st.checkbox("Auto threshold (50% of max)", value=True)
manual_thresh = st.number_input("Or enter manual threshold:", min_value=0.0, value=3.0, step=0.1)

def logistic_5pl(t, a, d, c, b, g):
    return d + (a - d) / (1 + (t / c)**b)**g

def inverse_5pl(y, a, d, c, b, g):
    try:
        base = ((a - d) / (y - d))**(1 / g) - 1
        return c * base**(1 / b)
    except:
        return np.nan

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
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.legend()
            ax.grid(False)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Could not fit {col}: {e}")
