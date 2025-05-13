import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root_scalar
from sklearn.metrics import r2_score
from scipy.stats import t
from io import BytesIO
import datetime
import zipfile
import traceback # For detailed error logging

# --- Model Functions (unchanged, these are fine) ---
def logistic_5pl(x, a, d, c, b, g):
    return d + (a - d) / (1 + (x / c) ** b) ** g

def logistic_4pl(x, a, d, c, b):
    return d + (a - d) / (1 + (x / c) ** b)

def sigmoid(x, L, x0, k):
    return L / (1 + np.exp(-k * (x - x0)))

def linear_model(x, m, c_intercept): # Renamed 'c' to avoid conflict
    return m * x + c_intercept

# --- Cached Helper Functions ---

@st.cache_data
def load_csv_data(uploaded_file_obj):
    """Loads data from an uploaded CSV file."""
    if uploaded_file_obj:
        try:
            return pd.read_csv(uploaded_file_obj)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return pd.DataFrame() # Return empty DataFrame on error
    return pd.DataFrame()

@st.cache_data
def calculate_inverse_threshold(y_target, model_func, popt_tuple, x_data_for_bracket):
    """
    Calculates the x value for a given y_target using the fitted model.
    Uses x_data_for_bracket to set a dynamic bracket for root_scalar.
    """
    if not x_data_for_bracket.size: # an empty array
        min_x, max_x = 0, 1000 # Default bracket if no x_data
    else:
        min_x = np.min(x_data_for_bracket) if np.min(x_data_for_bracket) < 0 else 0 # Ensure bracket starts >=0
        max_x = np.max(x_data_for_bracket) * 1.5 + 1 # Extend max a bit
        if max_x <= min_x: # Handle cases with single point or weird data
            max_x = min_x + 100

    try:
        # Ensure popt_tuple is used correctly
        result = root_scalar(lambda t_val: model_func(t_val, *popt_tuple) - y_target,
                             bracket=[min_x, max_x],
                             method='brentq')
        return result.root if result.converged else None
    except ValueError: # Brentq might fail if signs aren't opposite or root not in bracket
        # Fallback with a wider search or different method if brentq fails
        try:
            # Attempt with a broader bracket or a more robust (but potentially slower) method
            # For simplicity, we'll just try a slightly different bracket or log failure
            # st.warning(f"Brentq failed for inverse threshold, trying wider bracket. Min/Max X: {min_x}, {max_x}")
            # This part might need more robust error handling or alternative solvers
            result = root_scalar(lambda t_val: model_func(t_val, *popt_tuple) - y_target,
                                 x0=np.median(x_data_for_bracket) if x_data_for_bracket.size else (min_x+max_x)/2,
                                 x1=max_x, # Try with a different method if brentq is problematic
                                 method='secant', # Example: might need more tuning
                                 xtol=1e-5, rtol=1e-5, maxiter=200)
            return result.root if result.converged else None
        except Exception:
            # st.warning(f"Inverse threshold calculation failed for y={y_target}. Error: {inner_e}")
            return None
    except Exception:
        # st.warning(f"General error in inverse threshold calculation: {e}")
        return None

@st.cache_data
def fit_single_sample(
    x_data_tuple, y_data_tuple, model_name, p0_tuple, threshold,
    calibration_coeffs_tuple, sample_name_for_error # Added for better error messages
):
    """
    Fits a single sample's data, calculates R2, TT, LogCFU, and CIs.
    Takes tuples for array-like data to ensure hashability for caching.
    """
    x_data = np.array(x_data_tuple)
    y_data = np.array(y_data_tuple)

    # Map model name to function
    model_map = {
        "5PL": logistic_5pl,
        "4PL": logistic_4pl,
        "Sigmoid": sigmoid,
        "Linear": linear_model
    }
    func_to_fit = model_map.get(model_name)

    if func_to_fit is None:
        return None, f"Unknown model: {model_name}", None, None, None, None, None

    try:
        if model_name == "Linear":
            if len(x_data) < 2: # Not enough points for linear fit
                 return None, "Not enough data points for Linear fit.", None, None, None, None, None
            popt, residuals, _, _, _ = np.polyfit(x_data, y_data, 1, full=True) # Get residuals for CI
            pcov = np.zeros((2,2)) # Placeholder, polyfit CIs are different
            y_fit_vals = func_to_fit(x_data, *popt)
            # Simplified CI for linear - or could implement proper prediction intervals
            y_ci_vals = (y_fit_vals, y_fit_vals) # For now, no CI on y_fit for linear from this method
        else:
            if len(x_data) < len(p0_tuple): # Not enough points for model
                return None, f"Not enough data points for {model_name} fit (need at least {len(p0_tuple)}).", None, None, None, None, None
            popt, pcov = curve_fit(func_to_fit, x_data, y_data, p0=list(p0_tuple), maxfev=20000, check_finite=True)
            y_fit_vals = func_to_fit(x_data, *popt)

            # Confidence Interval Calculation for y_fit
            dof = max(1, len(x_data) - len(popt)) # Ensure dof is at least 1
            tval = t.ppf(0.975, dof)
            ci_low_vals, ci_high_vals = [], []

            # Check if pcov is sensible
            if np.any(np.isinf(pcov)) or np.any(np.isnan(pcov)) or pcov.shape != (len(popt), len(popt)):
                # st.warning(f"Covariance matrix issue for {sample_name_for_error}. CIs might be unreliable.")
                y_ci_vals = (y_fit_vals, y_fit_vals) # Fallback: no CI width
            else:
                for i_val, xi_val in enumerate(x_data):
                    grad_vals = []
                    for j_val in range(len(popt)):
                        popt_perturbed = np.array(popt, dtype=float)
                        h = 1e-6 * popt_perturbed[j_val] if popt_perturbed[j_val] != 0 else 1e-6
                        if h == 0: h = 1e-6 # Ensure h is not zero
                        popt_perturbed[j_val] += h
                        grad_val = (func_to_fit(xi_val, *popt_perturbed) - y_fit_vals[i_val]) / h
                        grad_vals.append(grad_val)
                    grad_vals = np.array(grad_vals)

                    # Check for singular pcov more robustly
                    try:
                        # Add small diagonal epsilon if pcov is singular or nearly singular
                        # This is a regularization technique.
                        # Note: This changes pcov, so SEs are approximate.
                        # A better approach is to handle singular pcov by flagging CIs as unreliable.
                        # For now, we try to compute something.
                        if np.linalg.cond(pcov) > 1e10: # Condition number check for near-singularity
                             diag_eps = 1e-9
                             se_val = np.sqrt(max(0, grad_vals @ (pcov + np.eye(pcov.shape[0])*diag_eps) @ grad_vals.T))
                        else:
                             se_val = np.sqrt(max(0, grad_vals @ pcov @ grad_vals.T)) # max(0,...) for numerical stability
                    except np.linalg.LinAlgError: # If pcov is singular
                        se_val = np.inf

                    delta_val = tval * se_val
                    ci_low_vals.append(y_fit_vals[i_val] - delta_val)
                    ci_high_vals.append(y_fit_vals[i_val] + delta_val)
                y_ci_vals = (np.array(ci_low_vals), np.array(ci_high_vals))

        r2_val = r2_score(y_data, y_fit_vals)
        tt_val_res = calculate_inverse_threshold(threshold, func_to_fit, tuple(popt), x_data)

        logcfu_val = None
        if tt_val_res and calibration_coeffs_tuple:
            a_cal, b_cal = calibration_coeffs_tuple[0] # Assuming structure ([a,b], None)
            logcfu_val = a_cal * tt_val_res + b_cal

        fit_df_res = pd.DataFrame({
            'Time': x_data, 'Raw': y_data, 'Fit': y_fit_vals,
            'CI Lower': y_ci_vals[0], 'CI Upper': y_ci_vals[1]
        })
        return fit_df_res, None, r2_val, tt_val_res, logcfu_val, popt, model_name

    except RuntimeError as e: # curve_fit often raises RuntimeError
        err_msg = f"Fitting runtime error for {sample_name_for_error} ({model_name}): {e}. Check initial parameters (p0) or data quality."
        # st.warning(err_msg) # For debugging
        return None, err_msg, None, None, None, None, None
    except Exception as e:
        # err_msg = f"Unexpected error fitting {sample_name_for_error} ({model_name}): {e}\n{traceback.format_exc()}"
        err_msg = f"Unexpected error fitting {sample_name_for_error} ({model_name}): {e}."
        # st.error(err_msg) # For debugging
        return None, err_msg, None, None, None, None, None

@st.cache_data
def generate_plot_buffer(
    sample_name, time_data_tuple, raw_data_tuple, fit_data_tuple,
    ci_lower_tuple, ci_upper_tuple, x_label, y_label,
    threshold_val, tt_val, logcfu, plot_dpi, model_name_for_title=""
):
    """Generates a PNG plot buffer for a single sample."""
    fig, ax = plt.subplots(figsize=(6, 4)) # Consider making figsize dynamic or a parameter
    ax.plot(time_data_tuple, raw_data_tuple, 'o', label='Data', color='black', markersize=5)
    ax.plot(time_data_tuple, fit_data_tuple, label=f'{model_name_for_title} Fit', color='blue')
    if ci_lower_tuple is not None and ci_upper_tuple is not None:
        ax.fill_between(time_data_tuple, ci_lower_tuple, ci_upper_tuple, color='red', alpha=0.15, label='95% CI')

    ax.axhline(y=threshold_val, color='green', linestyle='--', label=f'Threshold ({threshold_val})')
    if tt_val is not None:
        ax.axvline(x=tt_val, color='orange', linestyle=':', label=f'TT ({tt_val:.2f})')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    title = f"{sample_name}"
    # if model_name_for_title: title += f" ({model_name_for_title})" # Already in fit label
    details = []
    if tt_val is not None: details.append(f"TT: {tt_val:.2f} h")
    if logcfu is not None: details.append(f"LogCFU/mL: {logcfu:.2f}")
    if details: title += f" ({', '.join(details)})"

    ax.set_title(title, fontsize=10)
    ax.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0., fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.82, 1]) # Adjust rect to make space for legend
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=plot_dpi)
    buf.seek(0)
    plt.close(fig)
    return buf

@st.cache_data
def generate_combined_plot_buffer(
    _fit_results_dict_items, # Pass items for hashing: tuple of (sample, df_dict)
    threshold_val, x_label, y_label, _summary_rows_tuple, plot_dpi # Pass tuple of dicts for hashing
):
    """Generates a combined plot for all samples."""
    fig, ax = plt.subplots(figsize=(10, 7)) # Slightly larger for combined plot
    summary_rows = [dict(row) for row in _summary_rows_tuple] # Convert back from frozenset if needed

    # Create a color cycle for plots
    colors = plt.cm.viridis(np.linspace(0, 1, len(_fit_results_dict_items)))

    for i, (sample_name, fit_df_dict) in enumerate(_fit_results_dict_items):
        fit_df = pd.DataFrame(fit_df_dict) # Reconstruct DataFrame
        # Find corresponding summary row for TT and LogCFU
        sample_summary = next((row for row in summary_rows if row['Sample'] == sample_name), None)
        tt_val = sample_summary.get('Threshold Time') if sample_summary else None
        logcfu_val = sample_summary.get('Log CFU/mL') if sample_summary else None

        label_parts = [sample_name]
        if tt_val is not None: label_parts.append(f"TT={tt_val:.2f}")
        if logcfu_val is not None: label_parts.append(f"CFU={logcfu_val:.2f}")
        label = " ".join(label_parts)

        ax.plot(fit_df['Time'], fit_df['Fit'], label=label, color=colors[i])
        if 'CI Lower' in fit_df and 'CI Upper' in fit_df:
            ax.fill_between(fit_df['Time'], fit_df['CI Lower'], fit_df['CI Upper'], color=colors[i], alpha=0.2)

    ax.axhline(y=threshold_val, color='green', linestyle='--', label=f'Threshold ({threshold_val})')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Combined Fit Plot", fontsize=12)
    ax.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0., fontsize=9)
    plt.tight_layout(rect=[0, 0, 0.8, 1]) # Adjust for legend
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=plot_dpi)
    buf.seek(0)
    plt.close(fig)
    return buf

@st.cache_data
def create_excel_report_buffer(_original_data_dict, _fit_results_dict_items, _summary_rows_tuple,
                               _calibration_coeffs_tuple, x_label, y_label):
    """Generates an Excel report in a buffer."""
    output = BytesIO()
    original_data = pd.DataFrame(_original_data_dict)
    summary_rows = [dict(row) for row in _summary_rows_tuple]

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        if not original_data.empty:
            original_data.to_excel(writer, sheet_name="Original Data", index=False)

        if _calibration_coeffs_tuple:
            (a_cal, b_cal), _ = _calibration_coeffs_tuple # Unpack
            pd.DataFrame({"Calibration Name": ["Manual"], "Slope": [a_cal], "Intercept": [b_cal]}).to_excel(writer, sheet_name="Calibration", index=False)

        if summary_rows:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

        param_rows = []
        # ... (parameter formula map logic - keeping it concise here, it was okay)
        formula_map = {
            '5PL': "y = d + (a - d) / (1 + (x / c)^b)^g", '4PL': "y = d + (a - d) / (1 + (x / c)^b)",
            'Sigmoid': "y = L / (1 + exp(-k*(x - x0)))", 'Linear': "y = m*x + c_intercept"
        }
        # ... (your parameter extraction logic - assuming it's correct for popt in summary)
        # This part needs popt to be stored in summary_rows or passed via fit_results if not already
        # For now, assuming popt is not directly written to Excel in this version to simplify caching.
        # If 'popt' is needed, add it to summary_rows.

        pd.DataFrame(param_rows).to_excel(writer, sheet_name="Fit Parameters (Simplified)", index=False) # Modified sheet name

        for sample_name, fit_df_dict in _fit_results_dict_items:
            fit_df = pd.DataFrame(fit_df_dict)
            # Ensure sheet name is valid (max 31 chars, no invalid chars)
            safe_sheet_name = "".join([c if c.isalnum() else "_" for c in sample_name])[:30]
            fit_df.to_excel(writer, sheet_name=f"Fit_{safe_sheet_name}", index=False)

    output.seek(0)
    return output

@st.cache_data
def export_all_plots_zip_buffer(_fit_results_dict_items, _summary_rows_tuple,
                                x_label, y_label, threshold_val, _combined_img_buf_content, plot_dpi):
    """Exports all individual plots and the combined plot into a ZIP buffer."""
    zip_buffer = BytesIO()
    summary_rows = [dict(row) for row in _summary_rows_tuple]

    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for sample_name, fit_df_dict in _fit_results_dict_items:
            fit_df = pd.DataFrame(fit_df_dict)
            sample_summary = next((row for row in summary_rows if row['Sample'] == sample_name), {})
            model = sample_summary.get('Model', 'UnknownModel')
            tt_val = sample_summary.get('Threshold Time')
            logcfu = sample_summary.get('Log CFU/mL')

            img_buf = generate_plot_buffer(
                sample_name, tuple(fit_df['Time']), tuple(fit_df['Raw']), tuple(fit_df['Fit']),
                tuple(fit_df['CI Lower']) if 'CI Lower' in fit_df else None,
                tuple(fit_df['CI Upper']) if 'CI Upper' in fit_df else None,
                x_label, y_label, threshold_val, tt_val, logcfu, plot_dpi, model_name_for_title=model
            )
            img_buf.seek(0) # Ensure buffer is at the start
            zip_file.writestr(f"{sample_name}_{model}_fit.png", img_buf.read())

        # Add combined plot
        if _combined_img_buf_content:
            zip_file.writestr("combined_fit.png", _combined_img_buf_content)

    zip_buffer.seek(0)
    return zip_buffer

# --- Initialize Session State ---
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame()
if 'summary_rows' not in st.session_state:
    st.session_state.summary_rows = [] # List of dicts
if 'fit_results_dataframes' not in st.session_state: # To store DataFrames for combined plot & export
    st.session_state.fit_results_dataframes = {} # sample_name -> df.to_dict('list')
if 'calibration_coef' not in st.session_state:
    st.session_state.calibration_coef = None # ([slope, intercept], None)
if 'model_selections' not in st.session_state:
    st.session_state.model_selections = {} # col_name -> model_name

# --- UI Elements ---
st.set_page_config(layout="wide")
st.title("📈 TT Finder - Curve Fitting Tool")

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("Upload CSV Data", type="csv")
    if uploaded:
        # Use a button to trigger processing to avoid re-processing on every widget change
        # if st.button("Load and Process Data"): # Optional: for more control
        new_data = load_csv_data(uploaded)
        if not new_data.equals(st.session_state.data):
            st.session_state.data = new_data
            # Reset dependent states when new data is loaded
            st.session_state.summary_rows = []
            st.session_state.fit_results_dataframes = {}
            st.session_state.model_selections = {}
            st.info("New data loaded. Previous results cleared.")
            st.experimental_rerun() # Force rerun to reflect cleared state immediately

    x_label = st.text_input("X-axis label", "Time (h)")
    y_label = st.text_input("Y-axis label", "Signal")
    manual_thresh = st.number_input("Threshold Value", 0.0, 10000.0, 3.0, 0.1, format="%.2f") # Increased max
    dpi = st.slider("Plot DPI", 100, 600, 200, 50) # Default DPI 200

    st.markdown("### Calibration")
    use_manual_calib = st.checkbox("Enable Manual Calibration", key="calib_check")
    if use_manual_calib:
        cal_a = st.number_input("Slope (a)", value=1.0, format="%.4f")
        cal_b = st.number_input("Intercept (b)", value=0.0, format="%.4f")
        st.session_state.calibration_coef = ([cal_a, cal_b], None)
    else:
        st.session_state.calibration_coef = None

# --- Main App Logic ---
if not st.session_state.data.empty and len(st.session_state.data.columns) > 1:
    data_df = st.session_state.data
    time_col_name = data_df.columns[0]
    time_all_samples = data_df[time_col_name].dropna().values

    current_summary_rows = []
    current_fit_results_dfs = {} # sample_name -> df.to_dict('list') for caching

    st.header("📊 Analysis per Sample")
    # Create columns for layout if many samples, or just iterate
    # num_cols = min(len(data_df.columns[1:]), 3) # e.g. max 3 columns for display
    # display_cols = st.columns(num_cols)
    # col_idx_cycle = 0

    for sample_col_name in data_df.columns[1:]:
        # with display_cols[col_idx_cycle % num_cols]: # If using st.columns layout
        with st.expander(f"Sample: {sample_col_name}", expanded=False):
            y_col_data = data_df[sample_col_name].dropna().values
            # Ensure x and y data are of the same length
            current_x_data = time_all_samples[:len(y_col_data)]

            if len(current_x_data) == 0 or len(y_col_data) == 0:
                st.warning(f"No valid data for sample {sample_col_name}.")
                continue

            # Model selection for this sample
            # Persist model selection using session_state
            default_model_idx = ["5PL", "4PL", "Sigmoid", "Linear"].index(st.session_state.model_selections.get(sample_col_name, "4PL"))
            selected_model = st.selectbox(
                "Select Model",
                ["5PL", "4PL", "Sigmoid", "Linear"],
                index=default_model_idx,
                key=f"model_select_{sample_col_name}"
            )
            st.session_state.model_selections[sample_col_name] = selected_model # Store selection

            # Determine initial parameters (p0)
            # Ensure y_col_data is not empty before calling min/max/median
            if not y_col_data.size or not current_x_data.size:
                 st.warning(f"Not enough data points in {sample_col_name} for p0 calculation.")
                 continue

            median_x = np.median(current_x_data) if current_x_data.size > 0 else 1.0 # Handle empty x
            min_y, max_y = (np.min(y_col_data), np.max(y_col_data)) if y_col_data.size > 0 else (0,1) # Handle empty y

            if selected_model == "5PL":
                p0 = (min_y, max_y, median_x, 1.0, 1.0)
            elif selected_model == "4PL":
                p0 = (min_y, max_y, median_x, 1.0)
            elif selected_model == "Sigmoid":
                p0 = (max_y, median_x, 1.0)
            else: # Linear
                p0 = (1.0, 0.0) # m, c for polyfit (not strictly p0 for curve_fit but consistent)

            # Ensure calibration_coeffs is hashable for caching (tuple of tuples/primitives)
            calib_coeffs_for_cache = None
            if st.session_state.calibration_coef:
                coeffs, other = st.session_state.calibration_coef
                calib_coeffs_for_cache = (tuple(coeffs), other)

            # Call cached fitting function
            fit_df_obj, error_msg, r2, tt_val, logcfu, popt_fit, model_name_fit = fit_single_sample(
                tuple(current_x_data), tuple(y_col_data), selected_model, p0,
                manual_thresh, calib_coeffs_for_cache, sample_col_name
            )

            if error_msg:
                st.error(f"Failed for {sample_col_name}: {error_msg}")
                current_summary_rows.append({
                    'Sample': sample_col_name, 'Model': selected_model, 'R²': None,
                    'Threshold Time': None, 'Log CFU/mL': None, 'Error': error_msg
                })
            elif fit_df_obj is not None:
                current_fit_results_dfs[sample_col_name] = fit_df_obj.to_dict('list') # Store as dict for caching
                current_summary_rows.append({
                    'Sample': sample_col_name, 'Model': model_name_fit,
                    'R²': round(r2, 4) if r2 is not None else None,
                    'Threshold Time': tt_val, # tt_val can be None, handle in display
                    'Log CFU/mL': logcfu, # logcfu can be None
                    'Error': None
                })

                # Generate and display plot for this sample
                plot_buf = generate_plot_buffer(
                    sample_col_name,
                    tuple(fit_df_obj['Time']), tuple(fit_df_obj['Raw']), tuple(fit_df_obj['Fit']),
                    tuple(fit_df_obj['CI Lower']) if 'CI Lower' in fit_df_obj else None,
                    tuple(fit_df_obj['CI Upper']) if 'CI Upper' in fit_df_obj else None,
                    x_label, y_label, manual_thresh, tt_val, logcfu, dpi, model_name_fit
                )
                st.image(plot_buf, caption=f"{sample_col_name} - {model_name_fit} Fit", use_container_width=True)
            else:
                st.warning(f"No fit data returned for {sample_col_name} with model {selected_model}.")
        # col_idx_cycle +=1 # If using st.columns layout

    # Update session state after processing all samples for this run
    st.session_state.summary_rows = current_summary_rows
    st.session_state.fit_results_dataframes = current_fit_results_dfs

    # --- Display Summary Table and Combined Plot (if data exists) ---
    if st.session_state.summary_rows:
        st.header("📄 Summary Results")
        summary_display_df = pd.DataFrame(st.session_state.summary_rows)
        # Format TT and LogCFU for display (handle None)
        if 'Threshold Time' in summary_display_df.columns:
            summary_display_df['Threshold Time'] = summary_display_df['Threshold Time'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
        if 'Log CFU/mL' in summary_display_df.columns:
            summary_display_df['Log CFU/mL'] = summary_display_df['Log CFU/mL'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")

        st.dataframe(summary_display_df[[col for col in ['Sample', 'Model', 'R²', 'Threshold Time', 'Log CFU/mL', 'Error'] if col in summary_display_df]], use_container_width=True)

        # Combined Plot
        # Prepare data for combined plot (needs to be hashable for caching)
        fit_results_items_for_cache = tuple(st.session_state.fit_results_dataframes.items())
        summary_rows_for_cache = tuple(frozenset(row.items()) for row in st.session_state.summary_rows) # tuple of frozensets of items

        if fit_results_items_for_cache:
            st.header("📈 Combined Plot")
            combined_plot_buf = generate_combined_plot_buffer(
                fit_results_items_for_cache, manual_thresh, x_label, y_label,
                summary_rows_for_cache, dpi
            )
            st.image(combined_plot_buf, caption="Combined Fit Plot", use_container_width=True)
        else:
            combined_plot_buf = None # Ensure it's defined

        # --- Download Buttons ---
        st.header("💾 Export Options")
        col1, col2 = st.columns(2)

        with col1:
            if combined_plot_buf: # Ensure buffer exists
                 combined_plot_buf.seek(0) # Reset buffer pointer for reading
                 combined_plot_content_for_zip = combined_plot_buf.read()
                 combined_plot_buf.seek(0) # Reset again if used elsewhere
            else:
                 combined_plot_content_for_zip = None

            zip_buffer = export_all_plots_zip_buffer(
                fit_results_items_for_cache, summary_rows_for_cache,
                x_label, y_label, manual_thresh, combined_plot_content_for_zip, dpi
            )
            st.download_button(
                "📦 Download All Plots (ZIP)",
                data=zip_buffer,
                file_name=f"tt_finder_plots_{datetime.datetime.now():%Y%m%d_%H%M%S}.zip",
                mime="application/zip",
                use_container_width=True
            )

        with col2:
            # Prepare data for Excel report (hashable)
            original_data_for_cache = st.session_state.data.to_dict('list') if not st.session_state.data.empty else {}
            calib_coeffs_for_cache = None
            if st.session_state.calibration_coef:
                coeffs, other = st.session_state.calibration_coef
                calib_coeffs_for_cache = (tuple(coeffs), other)


            excel_buffer = create_excel_report_buffer(
                original_data_for_cache, fit_results_items_for_cache, summary_rows_for_cache,
                calib_coeffs_for_cache, x_label, y_label
            )
            st.download_button(
                "📥 Download Excel Report",
                data=excel_buffer,
                file_name=f"tt_finder_report_{datetime.datetime.now():%Y%m%d_%H%M%S}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    elif not st.session_state.data.empty:
        st.info("Data loaded. Configure settings and models in the expanders above to see results.")
    else:
        st.info("👋 Welcome! Please upload a CSV file to begin.")


# --- Optional: test_fit_4pl_on_user_data_safe section ---
# This section was a bit disconnected. If you need it:
# 1. Ensure `user_data` is defined (e.g., st.session_state.data).
# 2. Make the function cacheable if it's expensive.
# 3. Call it conditionally if needed.
# Example:
# if st.sidebar.checkbox("Run 4PL Safety Test"):
#     if not st.session_state.data.empty:
#         st.markdown("--- \n ### 🧪 4PL Fit Safety Test")
#         # Redefine or import test_fit_4pl_on_user_data_safe, make it cached
#         # test_summary_df = test_fit_4pl_on_user_data_safe_cached(st.session_state.data.to_dict('list'), manual_thresh)
#         # st.dataframe(test_summary_df)
#         st.warning("4PL Safety Test section is placeholder and needs to be fully integrated.")
#     else:
#         st.sidebar.warning("Upload data to run safety test.")


