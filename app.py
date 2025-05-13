import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import t
from io import BytesIO
import datetime
import plotly.graph_objects as go

# SESSION STATE INIT
for key in ['fits', 'ci', 'summary_rows', 'model']: 
    if key not in st.session_state:
        st.session_state[key] = {}

# ----- MODEL DEFINITIONS -----
def logistic_5pl(x, a, d, c, b, g):
    return d + (a - d) / (1 + (x / c) ** b) ** g

def logistic_4pl(x, a, d, c, b):
    return d + (a - d) / (1 + (x / c) ** b)

def sigmoid(x, L ,x0, k):
    return L / (1 + np.exp(-k * (x - x0)))

def inverse_threshold_5pl(y, a, d, c, b, g):
    try:
        base = ((a - d) / (y - d)) ** (1 / g) - 1
        return c * base ** (1 / b)
    except:
        return np.nan

# ----- USER INPUT -----
st.title("📈 TT Finder - Curve Fitting Tool")
x_label = st.text_input("X-axis label", value="Time (h)")
y_label = st.text_input("Y-axis label", value="Signal")
manual_thresh = st.number_input("Manual threshold", 0.0, 100.0, 3.0, 0.1)
model_choice = st.selectbox("Choose curve model", ["5PL", "4PL", "Sigmoid", "Linear"])
fmt = st.selectbox("Image format", ["png", "jpeg", "svg", "pdf"])
dpi = st.slider("DPI", 100, 600, 300, 50)

uploaded = st.file_uploader("Upload CSV (first col = time)", type=["csv"])
if uploaded:
    data = pd.read_csv(uploaded)
else:
    st.stop()

st.session_state.summary_rows.clear()

if st.button("Run Analysis"):
    time_vals = data.iloc[:, 0].dropna().values
    st.sidebar.markdown("### Fit Errors")

    for col in data.columns[1:]:
        y_vals = data[col].dropna().values
        x_vals = time_vals[:len(y_vals)]
        thresh_time, ci_thresh = None, (None, None)
        model_func, p0 = None, None

        # MODEL SELECTION
        if model_choice == "5PL":
            model_func = logistic_5pl
            p0 = [min(y_vals), max(y_vals), np.median(x_vals), 1, 1]
        elif model_choice == "4PL":
            model_func = logistic_4pl
            p0 = [min(y_vals), max(y_vals), np.median(x_vals), 1]
        elif model_choice == "Sigmoid":
            model_func = sigmoid
            p0 = [max(y_vals), np.median(x_vals), 1]

        with st.expander(f"{col} – Analysis"):
            try:
                if model_choice == "Linear":
                    coef = np.polyfit(x_vals, y_vals, 1)
                    y_fit = np.polyval(coef, x_vals)
                    r2 = r2_score(y_vals, y_fit)
                    popt = coef

                    # CI (linear)
                    n = len(x_vals)
                    x_mean = np.mean(x_vals)
                    Sxx = np.sum((x_vals - x_mean) ** 2)
                    residuals = y_vals - y_fit
                    sigma2 = np.sum(residuals ** 2) / (n - 2)
                    tval = t.ppf(0.975, n - 2)
                    ci_low, ci_high = [], []
                    for xi, yhat in zip(x_vals, y_fit):
                        se = np.sqrt(sigma2 * (1/n + (xi - x_mean) ** 2 / Sxx))
                        delta = tval * se
                        ci_low.append(yhat - delta)
                        ci_high.append(yhat + delta)
                    y_ci = (np.array(ci_low), np.array(ci_high))

                else:
                    popt, pcov = curve_fit(model_func, x_vals, y_vals, p0=p0, maxfev=10000)
                    y_fit = model_func(x_vals, *popt)
                    r2 = r2_score(y_vals, y_fit)

                    # CI band
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

                # Estimate threshold time and its CI (5PL only for now)
                if model_choice == "5PL":
                    thresh_time = inverse_threshold_5pl(manual_thresh, *popt)

                # PLOTLY PLOT
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers', name='Raw Data'))
                fig.add_trace(go.Scatter(x=x_vals, y=y_fit, mode='lines', name=f'{model_choice} Fit'))
                fig.add_trace(go.Scatter(x=x_vals, y=y_ci[0], fill=None, mode='lines', line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=x_vals, y=y_ci[1], fill='tonexty', mode='lines', name='95% CI', line=dict(width=0)))
                fig.add_hline(y=manual_thresh, line_dash="dash", line_color="green")
                fig.update_layout(title=f"{col} – {model_choice} Fit", xaxis_title=x_label, yaxis_title=y_label)
                st.plotly_chart(fig, use_container_width=True)

                # Store
                st.session_state.fits[col] = (x_vals, y_fit)
                st.session_state.ci[col] = y_ci
                st.session_state.summary_rows.append({
                    'Sample': col,
                    'Model': model_choice,
                    'R²': round(r2, 4),
                    'Threshold Time': thresh_time
                })

            except Exception as e:
                st.sidebar.error(f"❌ {col}: {str(e)}")
                continue

    st.subheader("Summary Table")
    st.dataframe(pd.DataFrame(st.session_state.summary_rows))

    # DOWNLOAD CSV
    st.download_button(
        "Download Summary CSV",
        data=pd.DataFrame(st.session_state.summary_rows).to_csv(index=False).encode(),
        file_name=f"summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
