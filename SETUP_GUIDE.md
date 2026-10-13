# 🎓 Punjab Education Data Analysis — Setup Guide

> **Full project:** District-wise, gender-disaggregated school enrollment analysis across all 23 districts of Punjab (1989–2022), including an interactive Streamlit dashboard, an ML pipeline, and automated PDF documentation.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Prerequisites & System Requirements](#2-prerequisites--system-requirements)
3. [Getting the Project](#3-getting-the-project)
4. [Environment Setup](#4-environment-setup)
5. [Project Structure](#5-project-structure)
6. [Running the Components](#6-running-the-components)
7. [Quick-Reference Command Table](#7-quick-reference-command-table)
8. [Dashboard Features](#8-dashboard-features)
9. [Python Dependencies](#9-python-dependencies)
10. [Troubleshooting](#10-troubleshooting)
11. [Verification Checklist](#11-verification-checklist)

---

## 1. Project Overview

This project is a complete end-to-end **Machine Learning & Data Analytics** solution analysing 34 years of school enrollment records from Punjab, India.

| Component | File | Description |
|:---|:---|:---|
| **ML Pipeline** | `punjab_enrollment_pipeline.py` | Data cleaning → Feature engineering → Anomaly detection (Isolation Forest) → District clustering (K-Means) → ARIMA forecasting |
| **Interactive Dashboard** | `punjab_dashboard/app.py` | Streamlit + Plotly web app for interactive exploration of all results |
| **PDF Generator** | `generate_pdf_docs.py` | Auto-generates a formatted project report using ReportLab |

### Core Objectives

- 📈 **Historical Trend Analysis** — Track enrollment for Classes 1–5 and 6–8 (boys & girls) from 1989–2022
- ⚖️ **Gender Parity Index (GPI)** — Measure & visualise progress toward gender equity across all 23 districts
- 🤖 **Anomaly Detection** — Identify data quality issues and reporting errors using `IsolationForest`
- 🗺️ **District Clustering** — Group the 23 districts into typology profiles with K-Means (k=4)
- 🔮 **5-Year Forecast** — Project enrollment for 2023–2027 using Anomaly-Aware ARIMA

---

## 2. Prerequisites & System Requirements

| Requirement | Details |
|:---|:---|
| **Operating System** | Windows 10/11, macOS 12+, or Linux (Ubuntu 20.04+) |
| **Python Version** | **3.9 – 3.12** (3.11 recommended) |
| **RAM** | 2 GB minimum (4 GB recommended) |
| **Disk Space** | ~500 MB (including `.venv`) |
| **Web Browser** | Chrome, Edge, Firefox, or Safari |
| **Internet** | Required only for fresh dependency installation |

### Check your Python version

```powershell
python --version
```

Expected output: `Python 3.9.x` through `Python 3.12.x`

> ⚠️ **Note:** Python **3.13+** may have compatibility issues with `statsmodels` and `scikit-learn`. Use 3.9–3.12 for a smooth experience.

---

## 3. Getting the Project

If you are copying the project to a new machine, make sure these critical files and folders exist at the project root before proceeding:

```
punjab education data analysis project/
├── punjab_enrollment_pipeline.py   ← Main ML pipeline (REQUIRED)
├── generate_pdf_docs.py            ← PDF generator
├── requirements.txt                ← Dependency list (REQUIRED)
├── punjab_dashboard/
│   ├── app.py                      ← Dashboard (REQUIRED)
│   ├── data/
│   │   ├── punjab_district_long.csv
│   │   └── punjab_state_level.csv
│   ├── 1-5boys.csv                 ← Raw data files (REQUIRED)
│   ├── 1-5girls.csv
│   ├── 6-8boys.csv
│   └── 6-8girls.csv
└── outputs/                        ← Auto-generated (created by the pipeline)
```

> 💡 The `outputs/` folder is created automatically when you first run the ML pipeline. You do **not** need to create it manually.

---

## 4. Environment Setup

All commands must be run from the **project root directory** — the folder that contains `requirements.txt` and `punjab_enrollment_pipeline.py`.

---

### Option A — Use the Existing `.venv` (Recommended)

The project ships with a pre-configured `.venv` virtual environment. Simply activate it:

#### Windows — PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

#### Windows — Command Prompt (CMD)

```cmd
.venv\Scripts\activate.bat
```

#### macOS / Linux — Bash or Zsh

```bash
source .venv/bin/activate
```

After activation, your terminal prompt changes to show `(.venv)` at the beginning — confirming the environment is active.

---

> ⚠️ **PowerShell Execution Policy Error?**
> If you see _"running scripts is disabled on this system"_, run this **once** in your PowerShell session:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\.venv\Scripts\Activate.ps1
> ```
> This only affects the current terminal session — it does **not** make permanent system changes.

---

### Option B — Fresh Install on a New Machine

Use this path if the `.venv` folder does not exist.

#### Step 1 — Create a virtual environment

```powershell
python -m venv .venv
```

#### Step 2 — Activate the environment

```powershell
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Windows CMD:
.venv\Scripts\activate.bat

# macOS / Linux:
source .venv/bin/activate
```

#### Step 3 — Upgrade pip

```powershell
python -m pip install --upgrade pip
```

#### Step 4 — Install all dependencies

```powershell
pip install -r requirements.txt
```

Installation typically takes **2–5 minutes** depending on internet speed.

#### Step 5 — Verify the installation

```powershell
python -c "import pandas, numpy, sklearn, statsmodels, matplotlib, seaborn, streamlit, plotly, reportlab; print('All dependencies installed successfully!')"
```

If no errors appear and you see `All dependencies installed successfully!`, you are ready.

---

## 5. Project Structure

```text
punjab education data analysis project/
│
├── .venv/                              # Python virtual environment (do not modify)
├── requirements.txt                    # Locked package dependencies
├── SETUP_GUIDE.md                      # ← You are here
├── HOW_TO_RUN.md                       # Quick execution reference
├── README.md                           # Full analytical & technical documentation
├── PROJECT_DOCUMENTATION.pdf           # Auto-generated PDF report
│
├── punjab_enrollment_pipeline.py       # ★ Main ML & data pipeline script
├── generate_pdf_docs.py                # ReportLab PDF generation script
│
├── outputs/                            # Auto-generated artifacts (run pipeline first)
│   ├── 01_statewide_trend.png          # Historical enrollment trend chart
│   ├── 02_state_gpi.png                # Gender Parity Index evolution chart
│   ├── 03_gpi_heatmap.png              # District × Year GPI heatmap
│   ├── 04_anomalies_by_year.png        # Detected anomaly counts per year
│   ├── 05_cluster_scatter.png          # K-Means district typology scatter plot
│   ├── 06_forecast_validation.png      # Naive vs Anomaly-Aware ARIMA comparison
│   ├── 07_final_forecast.png           # 5-Year enrollment forecast with 95% CI
│   ├── consolidated_long.csv           # Cleaned, long-format master dataset
│   ├── derived_gpi_table.csv           # Wide table with computed GPI values
│   ├── all_flagged_anomalies.csv       # Full Isolation Forest scored output
│   ├── district_clusters.csv           # K-Means labels & district profiles
│   ├── forecast_2023_2027.csv          # ARIMA 5-year projections
│   ├── new_duplicate_findings.csv      # ML-discovered duplicate-year anomalies
│   └── punjab_state_level.csv          # Aggregated state-level figures
│
└── punjab_dashboard/                   # Streamlit interactive web application
    ├── app.py                          # ★ Streamlit application entrypoint
    ├── README.md                       # Dashboard-specific usage guide
    ├── 1-5boys.csv                     # Raw data — Primary Boys (Classes 1–5)
    ├── 1-5girls.csv                    # Raw data — Primary Girls (Classes 1–5)
    ├── 6-8boys.csv                     # Raw data — Upper Primary Boys (Classes 6–8)
    ├── 6-8girls.csv                    # Raw data — Upper Primary Girls (Classes 6–8)
    └── data/                           # Pre-processed data for the dashboard
        ├── punjab_district_long.csv    # Cleaned long-format district records
        └── punjab_state_level.csv      # Cleaned state-level aggregation
```

---

## 6. Running the Components

> ⚠️ **Always activate the virtual environment first** (see Section 4) before running any command. Your terminal prompt should show `(.venv)`.

---

### Step 1 — ML & Data Pipeline

Run this **first** to generate all charts, CSV tables, and ML model outputs in the `outputs/` folder.

```powershell
python punjab_enrollment_pipeline.py
```

Or directly without activation:

```powershell
.\.venv\Scripts\python.exe punjab_enrollment_pipeline.py
```

**Pipeline stages:**

| # | Stage | Output |
|:---:|:---|:---|
| 1 | **Load & Clean** — Reads 4 raw CSVs, strips formatting, reshapes to long format | `consolidated_long.csv` |
| 2 | **Feature Engineering** — Computes GPI, YoY % change, duplicate flags, rolling z-scores | Adds columns to master dataset |
| 3 | **Anomaly Detection** — `IsolationForest` (300 trees, 5% contamination) | `all_flagged_anomalies.csv`, `new_duplicate_findings.csv` |
| 4 | **EDA Visualisations** — 4 static publication-quality charts | `01_` to `04_*.png` |
| 5 | **District Clustering** — K-Means (k=4) with Silhouette Score evaluation | `district_clusters.csv`, `05_cluster_scatter.png` |
| 6 | **ARIMA Forecasting** — Naive vs Anomaly-Aware comparison + 2023–2027 projections | `forecast_2023_2027.csv`, `06_` and `07_*.png` |

**Expected runtime:** Under 30 seconds on a standard laptop.

**Key results you will see in the terminal:**

```
Recall on known 2018 carry-forward:  70.1%
Naive ARIMA MAPE:                    65.6%
Anomaly-Aware ARIMA MAPE:             5.7%
Pipeline complete. All outputs saved to ./outputs/
```

---

### Step 2 — Interactive Web Dashboard

Launch the Streamlit web application. The ML pipeline must have been run at least once before this step.

```powershell
streamlit run punjab_dashboard\app.py
```

Or directly without activation:

```powershell
.\.venv\Scripts\streamlit.exe run punjab_dashboard\app.py
```

**Once launched**, Streamlit prints:

```
  Local URL:    http://localhost:8501
  Network URL:  http://192.168.x.x:8501
```

Open **http://localhost:8501** in your browser. The dashboard loads automatically.

**To stop the dashboard:** Press `Ctrl + C` in the terminal.

**To run on a different port** (if 8501 is already in use):

```powershell
.\.venv\Scripts\streamlit.exe run punjab_dashboard\app.py --server.port 8502
```

Then visit `http://localhost:8502`.

---

### Step 3 — PDF Documentation Generator (Optional)

Regenerate the full project report as a PDF at any time:

```powershell
python generate_pdf_docs.py
```

This creates/overwrites `PROJECT_DOCUMENTATION.pdf` in the project root with an up-to-date formatted report including all findings, charts, methodology, and mathematical formulations.

---

## 7. Quick-Reference Command Table

| Task | Command (with activated `.venv`) | Command (direct path, no activation needed) |
|:---|:---|:---|
| **Activate environment** | `.\.venv\Scripts\Activate.ps1` | _(n/a)_ |
| **Run ML pipeline** | `python punjab_enrollment_pipeline.py` | `.\.venv\Scripts\python.exe punjab_enrollment_pipeline.py` |
| **Launch dashboard** | `streamlit run punjab_dashboard\app.py` | `.\.venv\Scripts\streamlit.exe run punjab_dashboard\app.py` |
| **Generate PDF docs** | `python generate_pdf_docs.py` | `.\.venv\Scripts\python.exe generate_pdf_docs.py` |
| **Install dependencies** | `pip install -r requirements.txt` | `.\.venv\Scripts\pip.exe install -r requirements.txt` |
| **Upgrade pip** | `python -m pip install --upgrade pip` | `.\.venv\Scripts\python.exe -m pip install --upgrade pip` |

---

## 8. Dashboard Features

The Streamlit dashboard provides **6 interactive views**, selectable from the left sidebar panel.

| View | What You Can Explore |
|:---|:---|
| 📊 **Punjab Overview** | State-wide Boys vs Girls enrollment trends (1989–2022), GPI evolution over time, class-group (1–5 vs 6–8) breakdown |
| 🔍 **District Comparison** | Side-by-side multi-district enrollment & GPI line charts; select any combination of the 23 districts |
| 🎯 **Single District Deep-Dive** | Full trend chart, GPI chart, raw data table with missing-year notes, and CSV download button |
| 🗺️ **GPI Heatmap** | Interactive `RdYlGn` colour heatmap of Gender Parity Index — toggle between 12 continuous-history districts or all 23 |
| 🤖 **ML Anomalies & Forecasting** | ARIMA 2023–2027 forecast with 95% confidence interval, K-Means cluster profiles, candidate duplicate anomaly findings |
| 📝 **Data Quality Notes** | Documented known irregularities (2017=2018 duplication, 2011–2013 spike, missing 2008, new districts); missing-data % table |

### Sidebar Controls

| Control | Options | Effect |
|:---|:---|:---|
| **Class Group** | Both (1–8), Classes 1–5, Classes 6–8 | Filters all charts and tables to the selected class group |
| **Year Range** | Slider: 1989 – 2022 | Restricts the visible time window across all views |
| **View** | 6 options | Switches the main panel to the selected view |

---

## 9. Python Dependencies

All packages are pinned in `requirements.txt`. The table below describes what each library does in this project:

| Library | Min Version | Role in This Project |
|:---|:---:|:---|
| `pandas` | ≥ 3.0 | Reading raw CSVs, reshaping to long format, `groupby`, `pivot_table`, rolling statistics |
| `numpy` | ≥ 2.0 | Linear regression slopes (`polyfit`), MAPE calculation, NaN handling |
| `scikit-learn` | ≥ 1.9 | `IsolationForest` anomaly detection, `KMeans` clustering, `StandardScaler`, Silhouette Score |
| `statsmodels` | ≥ 0.15 | `ARIMA(1,1,1)` fitting, 5-year projections, 95% confidence intervals |
| `matplotlib` | ≥ 3.11 | Static PNG chart generation saved to `outputs/` |
| `seaborn` | ≥ 0.13 | `RdYlGn` divergent heatmaps, publication-quality themes |
| `streamlit` | ≥ 1.65 | Interactive web dashboard framework |
| `plotly` | ≥ 7.1 | Interactive charts within the dashboard (tooltips, zoom, pan) |
| `reportlab` | ≥ 5.0 | Programmatic PDF report generation |

---

## 10. Troubleshooting

### ❌ "running scripts is disabled on this system" (PowerShell)

**Cause:** Windows blocks unsigned PowerShell scripts by default.

**Fix:** Run once in the current PowerShell session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

---

### ❌ `ModuleNotFoundError` when running scripts

**Cause:** The script is using system Python instead of the virtual environment.

**Fix:** Ensure `(.venv)` is visible in your prompt. If not, activate first. Or use the explicit path:

```powershell
.\.venv\Scripts\python.exe punjab_enrollment_pipeline.py
```

To reinstall all packages:

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

---

### ❌ Port 8501 already in use

**Cause:** Another Streamlit instance or application is using port 8501.

**Fix:** Specify an alternative port:

```powershell
.\.venv\Scripts\streamlit.exe run punjab_dashboard\app.py --server.port 8502
```

Then open `http://localhost:8502`.

---

### ❌ Dashboard shows empty charts or "No data" messages

**Cause:** The ML pipeline has not been run yet — `outputs/` and `data/` CSV files do not exist.

**Fix:** Run the pipeline first:

```powershell
python punjab_enrollment_pipeline.py
```

---

### ❌ `pip install` fails with a C compiler / build error

**Cause:** `statsmodels` or `scikit-learn` require Microsoft C++ Build Tools on Windows.

**Fix:** Download and install from:
`https://visualstudio.microsoft.com/visual-cpp-build-tools/`

Select **"Desktop development with C++"**, install, then retry:

```powershell
pip install -r requirements.txt
```

---

### ❌ `python` command not found

**Cause:** Python is not installed or not added to the system `PATH`.

**Fix:**
1. Download Python 3.11 from [python.org/downloads](https://www.python.org/downloads/)
2. During installation, tick **"Add Python to PATH"**
3. Restart your terminal and verify: `python --version`

---

## 11. Verification Checklist

Use this checklist to confirm the project is set up and working correctly:

- [ ] `python --version` returns `3.9.x` – `3.12.x`
- [ ] Virtual environment activated — terminal prompt shows `(.venv)`
- [ ] `python punjab_enrollment_pipeline.py` completes without errors
- [ ] `outputs/` folder contains **7 PNG files** and **7 CSV files**
- [ ] `streamlit run punjab_dashboard\app.py` opens successfully at `http://localhost:8501`
- [ ] All **6 dashboard views** load without error messages in the browser

---

*For full analytical documentation and findings, see [README.md](README.md).*
*For a formatted PDF report, open [PROJECT_DOCUMENTATION.pdf](PROJECT_DOCUMENTATION.pdf).*
