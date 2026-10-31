# SMS GUARD — Complete Installation & Setup Guide

This guide is designed for **first-year Computer Science and Engineering students**. Even if you have never configured a Python machine-learning application before, following these step-by-step instructions will get SMS GUARD running cleanly on your personal laptop.

---

## 1. Prerequisites Checklist

Ensure your computer has the following tools installed:

1. **Python 3.10, 3.11, or 3.12:**
   - Download from [python.org/downloads](https://www.python.org/downloads/).
   - ⚠️ **CRITICAL ON WINDOWS:** Check the box that says **"Add python.exe to PATH"** during installation!
2. **Visual Studio Code (VS Code):**
   - Download from [code.visualstudio.com](https://code.visualstudio.com/).
3. **VS Code Python Extension:**
   - In VS Code, press `Ctrl+Shift+X` (or `Cmd+Shift+X` on Mac), search for **Python** by Microsoft, and click **Install**.

---

## 2. Step-by-Step Installation Procedure

### Step 1: Open the Project in VS Code
1. Launch VS Code.
2. Click **File > Open Folder...**
3. Select the `SMS_GUARD` directory.

---

### Step 2: Open the Integrated Terminal
In VS Code, press `` Ctrl + ` `` (backtick) or select **Terminal > New Terminal** from the top menu.

---

### Step 3: Create an Isolated Virtual Environment
A virtual environment ensures that the packages you install for SMS GUARD do not conflict with other Python applications on your computer.

#### On Windows (PowerShell or Command Prompt):
```powershell
python -m venv .venv
```

#### On macOS or Linux:
```bash
python3 -m venv .venv
```

---

### Step 4: Activate the Virtual Environment

#### On Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```
*(If PowerShell displays an `Execution_Policies` restriction error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and retry, or use Command Prompt: `.venv\Scripts\activate.bat`)*

#### On macOS or Linux:
```bash
source .venv/bin/activate
```

When activated, you will see `(.venv)` displayed at the beginning of your terminal prompt.

---

### Step 5: Install Project Dependencies
Run pip to install all required libraries specified in `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- `streamlit` (Frontend web interface)
- `pandas` & `numpy` (Dataset processing)
- `scikit-learn` (Machine learning algorithms)
- `plotly` & `matplotlib` (Visualizations)
- `joblib` (Model serialization)
- `pytest` (Automated testing suite)

---

### Step 6: Dataset Placement
The UCI SMS Spam Collection dataset is automatically provided in the `data/` folder:
`data/sms_dataset.csv`

If you ever need to reset or re-download the dataset, execute:
```bash
python -c "
import urllib.request, zipfile
url = 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'
urllib.request.urlretrieve(url, 'data/sms.zip')
with zipfile.ZipFile('data/sms.zip', 'r') as z:
    z.extract('SMSSpamCollection', 'data/')
import pandas as pd
df = pd.read_csv('data/SMSSpamCollection', sep='\t', names=['label', 'message'], quoting=3)
df.to_csv('data/sms_dataset.csv', index=False)
print('Dataset ready!')
"
```

---

### Step 7: Train the Machine Learning Model
Before starting the web application, train the Scikit-Learn TF-IDF + Naive Bayes pipeline on the dataset:

```bash
python -m src.train_model
```

You will see output detailing dataset distribution, train/test split (80/20), test accuracy (~98.26%), precision, recall, and confusion matrix. The trained pipeline will be saved to `models/spam_classifier.pkl`.

---

### Step 8: Run the Automated Unit Test Suite
Verify that all 22 automated unit tests pass:

```bash
python -m pytest tests -v
```

All tests should show `PASSED` in green.

---

### Step 9: Launch the SMS GUARD Application
Launch the Streamlit web dashboard:

```bash
streamlit run app.py
```

Streamlit will print a local URL in your terminal:
```
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Your web browser will automatically open to this URL. You can now classify messages, view history, inspect performance metrics, and review the project documentation!

To stop the running application, press `Ctrl + C` in the terminal.

---

## 3. Troubleshooting Common Errors

### 1. `python` is not recognized as an internal or external command
- **Cause:** Python was installed without checking the "Add Python to PATH" box.
- **Solution:** Re-run the Python installer, select **Modify**, and ensure **"Add Python to environment variables"** is checked. Alternatively, use `py -m venv .venv`.

### 2. `cannot be loaded because running scripts is disabled on this system` (PowerShell)
- **Cause:** Windows PowerShell script execution policy prevents unsigned script activation.
- **Solution:** Run this command in your PowerShell terminal:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```
  Then run `.venv\Scripts\Activate.ps1` again.

### 3. `ModuleNotFoundError: No module named 'src'`
- **Cause:** The command was run from inside a subfolder rather than the project root directory.
- **Solution:** Ensure your terminal's current directory is `SMS_GUARD`:
  ```bash
  cd SMS_GUARD
  python -m src.train_model
  ```

### 4. Port 8501 is already in use
- **Cause:** Another Streamlit application is running on port 8501.
- **Solution:** Run Streamlit on an alternative port:
  ```bash
  streamlit run app.py --server.port 8502
  ```
