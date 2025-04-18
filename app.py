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
if "login_log" not in st.session_state:
    st.session_state.login_log = []

username = st.text_input("Enter your name to begin:")
if st.button("Enter App"):
    if username:
        st.session_state.username = username
        st.session_state.login_log.append((username, str(datetime.datetime.now())))
        send_notification(username)
        st.experimental_rerun()
    else:
        st.error("❌ Please enter your name")

if "username" not in st.session_state or not st.session_state.username:
    st.stop()

# ----- LOGOUT BUTTON -----
with st.sidebar:
    if st.button("🚪 Logout"):
        st.session_state.username = ""
        st.experimental_rerun()

# Display login log (debug/admin tool)
with st.sidebar:
    st.write("### Login Log")
    for entry in st.session_state.login_log:
        st.write(f"{entry[0]} @ {entry[1]}")

# ----- APP LOGIC -----
# (App logic remains unchanged from this point on)
