# Punjab School Enrollment Data Analysis (1989–2022)

An end-to-end Machine Learning and Data Analytics project analyzing 34 years of district-wise, gender-disaggregated school enrollment records (Classes 1–5 and 6–8) in Punjab, India.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Libraries & Technologies Used and Their Purpose](#libraries--technologies-used-and-their-purpose)
3. [Project Architecture & Pipeline Workflow](#project-architecture--pipeline-workflow)
4. [Key Analytical Findings & Results](#key-analytical-findings--results)
5. [Data Quality Anomalies & Mitigation](#data-quality-anomalies--mitigation)
6. [Project Structure](#project-structure)
7. [Installation & Execution Guide](#installation--execution-guide)
8. [PDF Documentation](#pdf-documentation)

---

## 1. Project Overview

The primary objective of this project is to explore, audit, cluster, and forecast school enrollment dynamics across all 23 districts of Punjab from 1989 to 2022.

### Core Objectives:
- **Historical Enrollment Analysis**: Track long-term educational trends for Primary (Classes 1–5) and Upper Primary (Classes 6–8) cohorts for both boys and girls.
- **Gender Parity Index (GPI) Evaluation**: Calculate and visualize the ratio of female to male enrollment ($\text{GPI} = \frac{\text{Girls}}{\text{Boys}}$) to measure progress toward gender equity.
- **Unsupervised Anomaly Detection**: Detect reporting anomalies, administrative shifts, and duplicated data points using Machine Learning.
- **District Typology Clustering**: Group districts with similar demographic behaviors and educational volatility using K-Means clustering.
- **Anomaly-Aware Time Series Forecasting**: Produce robust 5-year future enrollment projections (2023–2027) using ARIMA, demonstrating how handling structural anomalies dramatically reduces forecast error.
- **Interactive Exploration Dashboard**: Provide policymakers and researchers with an interactive Streamlit web dashboard.

---

## 2. Libraries & Technologies Used and Their Purpose

Each library in this project was selected for a specific data-engineering, machine-learning, statistical, or visualization requirement:

| Library | Version | Category | Specific Role & Purpose in This Project |
| :--- | :--- | :--- | :--- |
| **`pandas`** | `>= 3.0` | Data Manipulation | **Data Ingestion & Transformation**: Reads raw wide-format CSVs (`1-5boys.csv`, etc.), unpivots them to long/tidy format (`melt`), computes group aggregations (`groupby`), builds pivot tables (`pivot_table`), handles missing values, and calculates rolling statistics. |
| **`numpy`** | `>= 2.0` | Numerical Computing | **Mathematical & Array Operations**: Calculates linear regression slopes (`polyfit`) for enrollment and GPI trajectories, computes Mean Absolute Percentage Error (MAPE), and safely manages NaN/infinite values. |
| **`scikit-learn`** | `>= 1.9` | Machine Learning | **Unsupervised Learning & Clustering**: Powers the **Isolation Forest** (`IsolationForest`) for anomaly detection; performs **K-Means clustering** (`KMeans`) to segment districts; standardizes features with **StandardScaler**; and evaluates cluster cohesion via **Silhouette Score**. |
| **`statsmodels`** | `>= 0.15` | Econometrics & Time Series | **ARIMA Forecasting**: Builds and fits Autoregressive Integrated Moving Average (`ARIMA(1,1,1)`) models. Generates 5-year projections (2023–2027) along with formal analytical 95% confidence intervals (`conf_int`). |
| **`matplotlib`** | `>= 3.11` | Visualization | **Static Chart Rendering**: Produces high-resolution publication-quality PNG figures for trends, anomaly distributions, cluster distributions, and ARIMA validation/forecast curves saved in `./outputs/`. |
| **`seaborn`** | `>= 0.13` | Statistical Plotting | **Heatmaps & Theming**: Formats divergent color heatmaps (`RdYlGn`) for the District $\times$ Year Gender Parity Index matrix, and creates publication-grade bubble scatter plots with styled themes (`whitegrid`). |
| **`streamlit`** | `>= 1.65` | Web Application | **Interactive Web Dashboard**: Powers the interactive web application (`punjab_dashboard/app.py`), enabling dynamic filtering by class group, year ranges, district selections, and views. |
| **`plotly`** | `>= 7.1` | Interactive Visualization | **Interactive Charts in UI**: Powers rich Plotly charts (`px.line`, `go.Scatter`, `px.imshow`) within Streamlit, providing tooltips, interactive zooming, panning, and responsive legends. |
| **`reportlab`** | `>= 5.0` | Document Generation | **Automated PDF Reporting**: Programmatically compiles the project documentation, methodology, library roles, and findings into a standalone PDF report (`PROJECT_DOCUMENTATION.pdf`). |

---

## 3. Project Architecture & Pipeline Workflow

The project is structured into two complementary layers: the **Analytical ML Pipeline** and the **Interactive Web Application**.

```
[Raw CSVs (4 files)] 
       │
       ▼
[Data Cleaning & Long-format Reshaping] (pandas)
       │
       ├─────────────────────────────────────────────┐
       ▼                                             ▼
[Feature Engineering]                         [Dashboard Data Export]
• YoY % Growth                                • punjab_district_long.csv
• Duplicate Flag                              • punjab_state_level.csv
• 5-Year Rolling Z-Score                             │
       │                                             ▼
       ▼                                     [Streamlit Web App]
[Isolation Forest Anomaly Detection]          • Overview & Trends
• Flags 2011-2013 Volatility                  • District Comparison
• Flags 2018 Carry-Forward Duplicates         • Single District Deep Dive
• Discovers Unknown Duplicate Years           • GPI Heatmap Matrix
       │                                      • ML Forecasts & Clusters
       ├──────────────────────┬──────────────────────┤
       ▼                      ▼                      ▼
[EDA Visualizations]   [K-Means Clustering]   [Anomaly-Aware ARIMA]
• 4 Primary Charts     • Silhouette Scoring   • Naive ARIMA: 65.6% MAPE
• Saved to /outputs/   • 4 District Profiles  • Anomaly-Aware: 5.7% MAPE
                       • Scatter Visualization• 2023-2027 Forecast
```

### Detailed Pipeline Stages:
1. **Load, Clean, Reshape (`load_all`)**: Ingests wide CSV files, strips text artifacts and commas, replaces 0 placeholders with NaNs, filters statewide totals, and melts into long records.
2. **Derived Fields (`build_derived`)**: Computes Total Enrollment ($\text{Boys} + \text{Girls}$), Girls Share ($\%$) and GPI ($\frac{\text{Girls}}{\text{Boys}}$).
3. **Feature Engineering (`make_features`)**: Computes year-over-year rate of change, exact value duplication flags (`dup_prev`), and centered 5-year rolling standard deviation z-scores (`roll_z`).
4. **Isolation Forest (`detect_anomalies`)**: Trains an ensemble of 300 isolation trees with a contamination threshold of 5% to detect outlier behavior across multivariate features.
5. **Validation (`validate`)**: Tests model recall against known data anomalies (such as the 2017→2018 carry-forward) and outputs candidate duplicate findings outside 2018.
6. **Exploratory Data Analysis (`eda`)**: Renders state-wide enrollment trajectories, class-wise GPI evolution, district GPI heatmaps, and anomaly counts per year.
7. **District Clustering (`cluster`)**: Normalizes district summary statistics (average GPI, GPI change, GPI slope, enrollment slope, and enrollment volatility) to cluster districts into 4 distinct groups using K-Means.
8. **Anomaly-Aware Forecasting (`forecasting`)**: Demonstrates how interpolating artificial reporting anomalies (2011–2013) improves ARIMA forecast accuracy by over **10x** during validation (2015–2019), and outputs 2023–2027 projections.

---

## 4. Key Analytical Findings & Results

### 1. Forecast Validation Accuracy (2015–2019)
- **Naive ARIMA(1,1,1)**: `65.6% MAPE` (heavily biased by the reporting spike in 2011–2013).
- **Anomaly-Aware ARIMA(1,1,1)**: `5.7% MAPE` (accurately tracks underlying structural trend).

### 2. State-Wide 5-Year Enrollment Forecast (2023–2027)
| Year | Projected Enrollment | 95% Confidence Interval (Lower) | 95% Confidence Interval (Upper) |
| :---: | :---: | :---: | :---: |
| **2023** | **1,732,471** | 1,402,997 | 2,061,945 |
| **2024** | **1,732,369** | 1,258,083 | 2,206,655 |
| **2025** | **1,732,313** | 1,144,341 | 2,320,286 |
| **2026** | **1,732,283** | 1,047,535 | 2,417,032 |
| **2027** | **1,732,267** | 961,965 | 2,502,569 |

### 3. District Typology Clusters (K-Means, $k=4$)
- **Cluster 0 (Core Industrial Plains)**: *Amritsar, Fatehgarh Sahib, Gurdaspur, Hoshiarpur, Jalandhar, Kapurthala, Ludhiana, Moga, Rupnagar, SBS Nagar*. Characterized by stable enrollment and steady GPI improvement.
- **Cluster 1 (High Volatility / Fast Developing)**: *Barnala, Fazilka, SAS Nagar, Sangrur, Tarn Taran*. Moderate GPI with higher volatility due to rapid urbanization and reorganization.
- **Cluster 2 (Rapid GPI Catch-Up)**: *Bathinda, Faridkot, Ferozepur, Mansa, Patiala, Sri Muktsar Sahib*. Historically lower baseline GPI in 1989 that exhibited the steepest gains over three decades.
- **Cluster 3 (Sub-Mountainous Outlier)**: *Pathankot*. Smallest geographic volume, carved out in 2011, distinct demographic profile.

---

## 5. Data Quality Anomalies & Mitigation

The project explicitly audits and documents real-world reporting anomalies rather than silently smoothing them:
1. **2017 = 2018 Carry-Forward Duplication**: In 98.9% of district/class/gender rows, 2017 numbers were repeated verbatim in 2018 at the source. The Isolation Forest model achieved **70.1% recall** on identifying these rows.
2. **2011–2013 Volatility Spike**: A dramatic artificial spike in 2011 and 2013 with a sharp trough in 2012 across almost all districts simultaneously, consistent with changes in reporting methods rather than real demographic swings.
3. **Missing Year 2008**: Completely missing across all districts, genders, and class groups in source records. Handled via gap interpolation in ARIMA and excluded from trend lines.
4. **Administrative District Carve-Outs**: Districts like Malerkotla (2021), Pathankot (2011), Fazilka (2011), SAS Nagar (2006) did not exist in earlier years. The pipeline and dashboard preserve them as `NaN` rather than misleading `0` values.

---

## 6. Project Structure

```text
punjab education data analysis project/
│
├── .venv/                              # Python virtual environment
├── requirements.txt                    # Locked dependencies
├── HOW_TO_RUN.md                       # Quick & detailed execution guide
├── README.md                           # Markdown project documentation
├── PROJECT_DOCUMENTATION.pdf           # Generated PDF project documentation
│
├── punjab_enrollment_pipeline.py       # Main data processing & ML pipeline
│
├── outputs/                            # Exported charts, tables, and models
│   ├── 01_statewide_trend.png          # Historical enrollment plot
│   ├── 02_state_gpi.png                # Gender parity index plot
│   ├── 03_gpi_heatmap.png              # District x Year GPI heatmap
│   ├── 04_anomalies_by_year.png        # Bar chart of detected anomalies
│   ├── 05_cluster_scatter.png          # K-Means district typology plot
│   ├── 06_forecast_validation.png      # Naive vs Anomaly-Aware ARIMA validation
│   ├── 07_final_forecast.png           # 5-Year forecast curve with 95% CI
│   ├── all_flagged_anomalies.csv       # Complete scored anomalies
│   ├── consolidated_long.csv           # Cleaned long-format dataset
│   ├── derived_gpi_table.csv           # Wide table with computed GPI
│   ├── district_clusters.csv           # Cluster labels and district profiles
│   ├── forecast_2023_2027.csv          # Future forecast table
│   ├── new_duplicate_findings.csv      # Discovered duplicate anomalies
│   └── punjab_state_level.csv          # Aggregated state figures
│
└── punjab_dashboard/                   # Streamlit Interactive Explorer
    ├── app.py                          # Streamlit application script
    ├── README.md                       # Dashboard usage guide
    ├── 1-5boys.csv                     # Raw data: Primary Boys
    ├── 1-5girls.csv                    # Raw data: Primary Girls
    ├── 6-8boys.csv                     # Raw data: Upper Primary Boys
    ├── 6-8girls.csv                    # Raw data: Upper Primary Girls
    └── data/                           # Synchronized data for dashboard
        ├── punjab_district_long.csv    # Cleaned long records
        └── punjab_state_level.csv      # Cleaned state records
```

---

## 7. Installation & Execution Guide

> 📖 **Full Guide:** For detailed step-by-step instructions, troubleshooting tips, and options, see [HOW_TO_RUN.md](HOW_TO_RUN.md).

### Step 1: Activate Environment
```powershell
# In Windows PowerShell:
.\.venv\Scripts\Activate.ps1
```

If setting up on a new machine:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 2: Run the Machine Learning Pipeline
```bash
python punjab_enrollment_pipeline.py
```
This executes the data cleaning, feature engineering, Isolation Forest detection, K-Means clustering, and ARIMA forecasting, automatically generating all plots and CSV files in `outputs/`.

### Step 3: Run the Streamlit Web Application
```bash
streamlit run punjab_dashboard/app.py
```
Open your browser at `http://localhost:8501`.

---

## 8. PDF Documentation

A complete, formatted PDF version of this documentation is generated and saved as:
[PROJECT_DOCUMENTATION.pdf](file:///e:/My%20Projects/punjab%20education%20data%20analysis%20project/PROJECT_DOCUMENTATION.pdf)

To regenerate the PDF at any time, run:
```bash
python generate_pdf_docs.py
```
