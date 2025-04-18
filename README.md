# 5PL Web App – Time-Resolved Fluorescence Analyzer

This is a web-based tool built in Streamlit for fitting 5-parameter logistic (5PL) models to time-resolved fluorescence data.

---

## 🔧 Features
- 5PL curve fitting with R² and confidence intervals
- Automatic or manual threshold detection
- Calculates time to threshold
- Export results as CSV
- Email notifications to elisa.santovito@cnr.it when someone logs in
- Public access: users just enter their name

---

## 📦 Files
- `app.py`: Streamlit application
- `requirements.txt`: Python dependencies
- `README.md`: This file

---

## 🚀 Deploy on Streamlit Cloud
1. Create a GitHub repository and upload all files
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **New app**
4. Select your repo
5. Set **main file** to `app.py`
6. Click **Deploy**

---

## 📧 Email Notification
The app notifies `elisa.santovito@cnr.it` via Gmail SMTP. You must replace `"your_email_app_password"` in `app.py` with a real Gmail **App Password**.

---

## 🧪 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 👩‍🔬 Author
Elisa Santovito – CNR  
