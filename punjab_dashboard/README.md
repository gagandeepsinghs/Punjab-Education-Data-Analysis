# Punjab Education Enrollment Explorer

An interactive Streamlit dashboard for exploring 1989–2022 district-wise, gender-disaggregated school enrollment data (Classes 1–5 and 6–8) for Punjab, India.

## Contents
- `app.py` — The interactive Streamlit dashboard
- `data/punjab_district_long.csv` — Cleaned, long-format district-level enrollment data
- `data/punjab_state_level.csv` — Cleaned, state-aggregated data with pre-computed Gender Parity Index (GPI)

## Quick Start

### 1. Activate Environment
Using the project's virtual environment:
```powershell
# In PowerShell (Windows):
.\.venv\Scripts\Activate.ps1
```

Or install dependencies with:
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
From the root directory or inside `punjab_dashboard`:
```bash
streamlit run punjab_dashboard/app.py
```
Or from within the `punjab_dashboard` folder:
```bash
streamlit run app.py
```

Your browser will automatically open at `http://localhost:8501`.

## Available Views
- **Punjab Overview** — State-wide enrollment and GPI trends across 1989–2022
- **District Comparison** — Compare multiple districts side by side without distorted missing-data values
- **Single District Deep-Dive** — Detailed enrollment and GPI trends, complete underlying table, and CSV download
- **GPI Heatmap** — District × Year Gender Parity Index heatmap with toggle for Continuous 12 districts vs All 23 districts
- **ML Anomalies & Forecasting** — Interactive display of 2023–2027 ARIMA projections with 95% CI, K-Means district clusters, and candidate duplicate findings
- **Data Quality Notes** — Documented, known irregularities in the source government data

## Documented Data Issues
- **2017 = 2018 duplication**: In 98.9% of district/class-group/gender rows, the 2017 and 2018 figures are identical.
- **2011–2013 volatility**: Sharp spike in 2011 and 2013 with a matching dip in 2012 across districts.
- **2008 missing year**: Completely absent across all districts in source records; omitted rather than zeroed.
- **New districts**: Carved out over time (Malerkotla 2021, Pathankot 2011, etc.), correctly preserved as missing data prior to formation.
