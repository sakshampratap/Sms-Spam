# 🛡️ SMS GUARD — Action Guide & Presentation Manual

## What is This?
This document is your **practical guide** for running, demonstrating, and submitting the **SMS GUARD** project on your laptop.

---

## 🚀 Quick Start (Running the Application)

### Step 1: Open VS Code
1. Open **Visual Studio Code**.
2. Click **File > Open Folder...** and select:
   `C:\Users\Saksham Singh\.gemini\antigravity\scratch\SMS_GUARD`

### Step 2: Open Terminal
- Press `` Ctrl + ` `` (backtick) or click **Terminal > New Terminal**.

### Step 3: Run the App
Type this command and press Enter:
```powershell
streamlit run app.py
```
Your browser will open automatically at `http://localhost:8501`.

---

## 🎯 How to Demo Each Page to Your Faculty

### Page 1: Main Dashboard
- Point out the **4 Metric Cards**: Total Messages Analysed, Spam Detected, Normal Messages, and 98.26% Model Accuracy.
- Emphasize that these statistics update dynamically from the **SQLite database** and **trained model metadata**.

### Page 2: SMS Classifier (Most Important)
- Click the sample buttons to demonstrate:
  - Click **"🏆 Free Prize Scam"** -> Click **Classify Message** -> Shows Red **SPAM DETECTED** banner (~99% confidence).
  - Click **"🎓 College Lecture"** -> Click **Classify Message** -> Shows Green **NORMAL MESSAGE** banner.
- Expand the **Linguistic Features** tab to show that URLs, currency symbols, and uppercase ratios are detected.

### Page 3: Message History
- Show that past classifications are saved in the local SQLite table.
- Demonstrate **Search**, **Filter (SPAM/HAM)**, and the **Download CSV** button.

### Page 4: Model Performance
- Show the **Confusion Matrix Heatmap** ($N = 1,035$ test samples: 900 TN, 117 TP, only 4 FP).
- Explain **Accuracy (98.26%)**, **Precision (96.69%)**, **Recall (89.31%)**, and **F1-Score (92.86%)**.

### Page 5: About Project
- Shows the problem statement, technology stack, and architecture flowchart.

---

## 🧪 How to Run Automated Tests
In the terminal, run:
```powershell
python -m pytest tests -v
```
All **22 tests** will run and pass in green.

---

## 📦 How to Submit the Project
1. Open File Explorer to:
   `C:\Users\Saksham Singh\.gemini\antigravity\scratch\`
2. Right-click the `SMS_GUARD` folder.
3. Select **Compress to ZIP file**.
4. Submit the generated `.zip` file.
