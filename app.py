import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import t
from io import BytesIO
import datetime
import plotly.graph_objects as go
import xlsxwriter
import tempfile
import os

# SESSION STATE INIT
for key in ['fits', 'ci', 'summary_rows', 'model_choices', 'calibration_coef']:
    if key not in st.session_state:
        st.session_state[key] = {}

# [Same model functions and user input code as before]

# Combined plot image generation
combined_fig_path = None
if not summary_df.empty:
    combined_fig.update_layout(title="Combined Model Fits", xaxis_title=x_label, yaxis_title=y_label)
    st.plotly_chart(combined_fig, use_container_width=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{fmt}") as tmp_img:
        combined_fig.write_image(tmp_img.name, format=fmt, scale=dpi/100)
        tmp_img.seek(0)
        combined_fig_bytes = tmp_img.read()
        combined_fig_path = tmp_img.name
        st.download_button("Download Combined Plot Image", combined_fig_bytes, file_name=f"combined_plot.{fmt}", mime=f"image/{fmt if fmt != 'svg' else 'svg+xml'}")

    # Excel export
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False, sheet_name="Raw Data")
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        if manual_calib:
            calib_df = pd.DataFrame({
                'Calibration Name': [calib_name],
                'Slope': [cal_slope],
                'Intercept': [cal_intercept]
            })
        elif 'calib_data' in locals():
            calib_df = calib_data
        else:
            calib_df = pd.DataFrame()
        if not calib_df.empty:
            calib_df.to_excel(writer, index=False, sheet_name="Calibration")
        for name, df in fit_results.items():
            df.to_excel(writer, index=False, sheet_name=name[:31])
        combined_data = pd.concat(fit_results.values(), keys=fit_results.keys()).reset_index()
        combined_data.to_excel(writer, index=False, sheet_name="Combined Fits")

        # Save combined plot image to Excel
        if combined_fig_path:
            workbook = writer.book
            worksheet = workbook.add_worksheet("Combined Plot")
            writer.sheets["Combined Plot"] = worksheet
            worksheet.insert_image("B2", combined_fig_path)
    excel_buffer.seek(0)
    st.download_button("Download All Results (Excel)", data=excel_buffer.read(), file_name="tt_finder_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if combined_fig_path:
        os.remove(combined_fig_path)

# [Show selected models as before]
