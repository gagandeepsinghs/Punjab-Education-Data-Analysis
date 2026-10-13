# How to Run the Punjab Education Data Analysis Project

This guide provides clear, step-by-step instructions to run every component of the **Punjab Education Data Analysis (1989–2022)** project, including the **interactive Streamlit dashboard**, the **end-to-end Machine Learning pipeline**, and the **PDF documentation generator**.

---

## ⚡ Quick Reference (One-Liner Commands)

If you already have the repository open in Windows PowerShell:

| Task | Command |
| :--- | :--- |
| **Run Web Dashboard** | `.\.venv\Scripts\streamlit.exe run punjab_dashboard\app.py` |
| **Open in Browser** | Visit **[http://localhost:8501](http://localhost:8501)** |
| **Run ML & Data Pipeline** | `.\.venv\Scripts\python.exe punjab_enrollment_pipeline.py` |
| **Generate PDF Documentation**| `.\.venv\Scripts\python.exe generate_pdf_docs.py` |

---

## 📋 Table of Contents
1. [Prerequisites & System Requirements](#1-prerequisites--system-requirements)
2. [Environment Setup & Activation](#2-environment-setup--activation)
3. [Running the Interactive Web Dashboard](#3-running-the-interactive-web-dashboard)
4. [Running the Machine Learning & Data Pipeline](#4-running-the-machine-learning--data-pipeline)
5. [Regenerating the PDF Documentation](#5-regenerating-the-pdf-documentation)
6. [Project Structure Overview](#6-project-structure-overview)
7. [Troubleshooting & Common Issues](#7-troubleshooting--common-issues)

---

## 1. Prerequisites & System Requirements

- **Operating System:** Windows, macOS, or Linux
- **Python Version:** Python 3.9 to 3.12 recommended
- **Hardware:** Standard CPU (runs locally in under 30 seconds)
- **Web Browser:** Google Chrome, Microsoft Edge, Firefox, or Safari

---

## 2. Environment Setup & Activation

### Option A: Using the Existing Virtual Environment (Recommended)
This workspace already contains a configured `.venv` directory.

#### On Windows (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```

> **Note on PowerShell Execution Policy:**
> If you get an error stating *“running scripts is disabled on this system”*, run this once in your PowerShell terminal:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\.venv\Scripts\Activate.ps1
> ```

#### On Windows (Command Prompt / CMD):
```cmd
.venv\Scripts\activate.bat
```

#### On Linux / macOS (Bash / Zsh):
```bash
source .venv/bin/activate
```

---

### Option B: Fresh Installation (From Scratch on a New Machine)
If setting up on a clean environment or machine:

1. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

2. **Activate the environment:**
   ```powershell
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 3. Running the Interactive Web Dashboard

The web application is built with **Streamlit** and **Plotly** to provide an interactive exploration interface.

### Step 1: Launch Streamlit
Run the following command from the project root directory:

```powershell
# With activated virtual environment:
streamlit run punjab_dashboard\app.py

# Or directly using the virtual environment path:
.\.venv\Scripts\streamlit.exe run punjab_dashboard\app.py
```

### Step 2: Access in Browser
Once launched, Streamlit will display local URLs:
- **Local URL:** [http://localhost:8501](http://localhost:8501)
- **Network URL:** `http://<your-local-ip>:8501`

Click the link or paste `http://localhost:8501` into your browser.

### Key Dashboard Views Available:
- 📊 **Punjab Overview:** Statewide long-term trends (1989–2022) and Gender Parity Index (GPI) evolution.
- 🔍 **District Comparison:** Side-by-side comparison across any combination of Punjab's 23 districts.
- 🎯 **Single District Deep-Dive:** Detailed metrics, missing data handling, and raw CSV download.
- 🗺️ **GPI Heatmap:** Comprehensive heatmap with a toggle for Continuous 12 districts vs. All 23 districts.
- 🤖 **ML Anomalies & Forecasting:** Interactive display of 2023–2027 ARIMA projections (with 95% Confidence Intervals), K-Means district clusters, and data duplicate findings.
- 📝 **Data Quality Notes:** Transparent documentation on historical data anomalies.

### How to Stop the Dashboard:
Press `Ctrl + C` in the terminal where Streamlit is running.

---

## 4. Running the Machine Learning & Data Pipeline

To re-process raw government data, detect anomalies, run district clustering, and compute ARIMA forecasts:

```powershell
# With activated environment:
python punjab_enrollment_pipeline.py

# Or directly:
.\.venv\Scripts\python.exe punjab_enrollment_pipeline.py
```

### What this pipeline does:
1. **Data Cleaning & Reshaping:** Cleans raw CSV files (`1-5boys.csv`, `1-5girls.csv`, `6-8boys.csv`, `6-8girls.csv`), aligns historical boundary changes, and handles missing years (e.g., 2008).
2. **Feature Engineering & GPI:** Calculates Gender Parity Index ($\text{GPI} = \frac{\text{Girls}}{\text{Boys}}$), YoY percentage shifts, and rolling statistics.
3. **Anomaly & Duplicate Auditing:** Runs scikit-learn's `IsolationForest` and detects duplicate reporting years (such as identical values in 2017 & 2018).
4. **District Typology (K-Means):** Clusters districts into 3 archetypes based on enrollment stability, scale, and gender parity.
5. **ARIMA Time Series Forecasting:** Compares Naive ARIMA vs. Anomaly-Aware ARIMA, reducing forecast error (RMSE) and projecting enrollment for 2023–2027.
6. **Artifact Generation:** Automatically updates the `outputs/` folder with 7 high-resolution charts and 7 CSV datasets.

---

## 5. Regenerating the PDF Documentation

The project includes an automated PDF generation script powered by **ReportLab**:

```powershell
# With activated environment:
python generate_pdf_docs.py

# Or directly:
.\.venv\Scripts\python.exe generate_pdf_docs.py
```

This compiles all project findings, methodologies, charts, and mathematical formulations into:
- [PROJECT_DOCUMENTATION.pdf](file:///e:/My%20Projects/punjab%20education%20data%20analysis%20project/PROJECT_DOCUMENTATION.pdf)

---

## 6. Project Structure Overview

```text
punjab education data analysis project/
│
├── .venv/                              # Python virtual environment
├── requirements.txt                    # Python package dependencies
├── HOW_TO_RUN.md                       # Execution guide (this file)
├── README.md                           # Main analytical & technical documentation
├── PROJECT_DOCUMENTATION.pdf           # Comprehensive report (PDF)
│
├── punjab_enrollment_pipeline.py       # ML & data processing pipeline script
├── generate_pdf_docs.py                # ReportLab script to generate documentation PDF
│
├── outputs/                            # Exported charts, CSVs, and model forecasts
│   ├── 01_statewide_trend.png
│   ├── 02_state_gpi.png
│   ├── 03_gpi_heatmap.png
│   ├── 04_anomalies_by_year.png
│   ├── 05_cluster_scatter.png
│   ├── 06_forecast_validation.png
│   ├── 07_final_forecast.png
│   ├── consolidated_long.csv
│   ├── derived_gpi_table.csv
│   ├── all_flagged_anomalies.csv
│   ├── district_clusters.csv
│   └── forecast_2023_2027.csv
│
└── punjab_dashboard/                   # Streamlit web application
    ├── app.py                          # Streamlit UI & interactive code
    ├── README.md                       # Dashboard specific overview
    ├── data/                           # Synchronized data for the UI
    │   ├── punjab_district_long.csv
    │   └── punjab_state_level.csv
    └── 1-5boys.csv, 1-5girls.csv, ...  # Raw government datasets
```

---

## 7. Troubleshooting & Common Issues

### Issue 1: Script execution is disabled in PowerShell
**Cause:** Default Windows PowerShell restriction on unsigned scripts.  
**Solution:** Run PowerShell as Current User with bypass flag:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Issue 2: Port 8501 is already in use
**Cause:** Another instance of Streamlit is already running.  
**Solution:** Specify an alternative port:
```powershell
.\.venv\Scripts\streamlit.exe run punjab_dashboard\app.py --server.port 8502
```
Then visit `http://localhost:8502`.

### Issue 3: Missing packages or `ModuleNotFoundError`
**Cause:** Running with system Python instead of the virtual environment.  
**Solution:** Ensure you use `.\.venv\Scripts\python.exe` or `.\.venv\Scripts\streamlit.exe`, or reinstall requirements:
```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```
