import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from io import BytesIO
import smtplib
from email.mime.text import MIMEText
import datetime

# ----- Simple 5PL logistic function for fitting -----
def logistic_5pl(t, a, d, c, b, g):
    return d + (a - d) / (1 + (t / c)**b)**g

# ----- Inverse 5PL function for calculating threshold time -----
def inverse_logistic_5pl(y, a, d, c, b, g):
    return c * (((a - d) / (y - d)) ** (1 / g) - 1) ** (1 / b)

# ----- Function to calculate the threshold time (Tt) -----
def calculate_threshold_time(threshold, popt):
    a, d, c, b, g = popt
    if not (d < threshold < a):
        raise ValueError(f"Threshold must be within the range of the 5PL function (d={d}, a={a}).")
    return c * (((a - d) / (threshold - d)) ** (1 / g) - 1) ** (1 / b)

# ----- EMAIL NOTIFICATION FUNCTION -----
def send_email_notification(username):
    try:
        msg = MIMEText(f"User '{username}' just accessed the 5PL web app on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
        msg['Subject'] = 'New 5PL App Session'
        msg['From'] = 'elisa.santovito@cnr.it'
        msg['To'] = 'elisa.santovito@cnr.it'

        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login('elisa.santovito@cnr.it', "your_email_app_password")
        server.send_message(msg)
        server.quit()
    except Exception as e:
        st.error(f"Error sending email: {e}")

# ----- STREAMLIT APP LOGIC -----
st.title("📈 5PL Curve Fitting Web App")
st.markdown("Paste or enter your fluorescence/time data below. First column should be time (in hours), others are samples.")

# Public access: user enters their name
user_name = st.text_input("Enter your name:")
if user_name:
    st.session_state.username = user_name
    send_email_notification(user_name)
    st.success(f"Welcome, {user_name}!")
else:
    st.warning("Please enter your name to begin.")

# Input labels
x_label = st.text_input("X-axis label", value="Time (h)")
y_label = st.text_input("Y-axis label", value="Signal")

# Dynamic sample data
num_samples = st.number_input("How many samples do you want to enter?", min_value=1, max_value=20, value=2, step=1)
sample_data = {"Time": np.arange(0, 4.25, 0.25)}  # Example time data
labels = []

for i in range(1, num_samples + 1):
    label = st.text_input(f"Label for Sample {i}", value=f"Sample{i}")
    labels.append(label)
    sample_data[label] = np.linspace(1 + i, 25 - i, len(sample_data["Time"]))

example_data = pd.DataFrame(sample_data)

# File upload or manual data entry
uploaded_file = st.file_uploader("Upload a CSV file (wide format, first column = Time)", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Data loaded from file.")
else:
    data = st.data_editor(example_data, use_container_width=True, num_rows="dynamic")

# Threshold and plot settings
manual_thresh = st.number_input("Enter manual threshold:", min_value=0.0, value=3.0, step=0.1)
fmt = st.selectbox("Select image format for download", options=["png", "jpeg", "svg", "pdf"], index=0)
dpi = st.slider("Image resolution (DPI)", min_value=100, max_value=600, value=300, step=50)

# Storage for results
fitting_results = []

# 5PL Fitting Process
if st.button("Run Analysis"):
    st.subheader("📊 Results")
    time = data.iloc[:, 0].dropna().values

    if len(time) == 0:
        st.error("❌ No valid time values found in the first column.")
    else:
        for col in data.columns[1:]:
            y = data[col].dropna().values
            t_fit = time[:len(y)]

            if len(t_fit) != len(y):
                st.error(f"❌ Sample '{col}' has mismatched time and data lengths.")
                continue

            try:
                initial_guess = [np.max(y), np.min(y), np.median(t_fit), 1, 1]
                popt, pcov = curve_fit(logistic_5pl, t_fit, y, p0=initial_guess, maxfev=10000)
                y_fit = logistic_5pl(t_fit, *popt)
                r2 = np.corrcoef(y, y_fit)[0, 1]**2

                # Confidence intervals using Monte Carlo simulation
                n_simulations = 1000
                param_samples = np.random.multivariate_normal(popt, pcov, size=n_simulations)
                y_simulations = np.array([logistic_5pl(t_fit, *params) for params in param_samples])
                ci_low = np.percentile(y_simulations, 2.5, axis=0)
                ci_high = np.percentile(y_simulations, 97.5, axis=0)

                # Threshold time
                try:
                    Tt = calculate_threshold_time(manual_thresh, popt)
                except ValueError as ve:
                    st.error(f"❌ {ve}")
                    Tt = np.nan

                # Save results
                fitting_results.append({
                    "Sample": col,
                    "Parameters": popt,
                    "R²": r2,
                    "Threshold Time (Tt)": Tt
                })

                # Create Plotly Plot
                fig = go.Figure()

                # Add raw data
                fig.add_trace(go.Scatter(x=t_fit, y=y, mode='markers', name="Raw Data", marker=dict(color='black')))

                # Add fitted curve
                fig.add_trace(go.Scatter(x=t_fit, y=y_fit, mode='lines', name="5PL Fit", line=dict(color='blue')))

                # Add 95% Confidence Interval
                fig.add_trace(go.Scatter(x=t_fit, y=ci_low, mode='lines', name="95% CI Low", line=dict(color='red', dash='dot')))
                fig.add_trace(go.Scatter(x=t_fit, y=ci_high, mode='lines', name="95% CI High", line=dict(color='red', dash='dot')))

                # Add threshold line
                fig.add_trace(go.Scatter(x=t_fit, y=np.full_like(t_fit, manual_thresh), mode='lines', name="Threshold", line=dict(color='green', dash='solid')))

                # Update layout
                fig.update_layout(
                    title=f"{col} Fit",
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    hovermode="closest",
                    showlegend=True
                )

                st.plotly_chart(fig)

                # Display fitting parameters
                st.write(f"Fitting Results for {col}:")
                st.write(f"- R²: {r2:.4f}")
                st.write(f"- Parameters: {popt}")
                st.write(f"- Threshold Time (Tt): {Tt:.2f} hours")

            except Exception as e:
                st.error(f"❌ Could not fit {col}: {e}")

        # Display parameters table
        st.write("### Parameters Table")
        params_table = pd.DataFrame(
            [{"Sample": res["Sample"], "R²": res["R²"], "Threshold Time (Tt)": res["Threshold Time (Tt)"], **dict(zip(["a", "d", "c", "b", "g"], res["Parameters"]))} for res in fitting_results]
        )
        st.dataframe(params_table)

        # Export fitting data
        st.download_button(
            label="📥 Export Fitting Data",
            data=params_table.to_csv(index=False),
            file_name="fitting_results.csv",
            mime="text/csv"
        )
