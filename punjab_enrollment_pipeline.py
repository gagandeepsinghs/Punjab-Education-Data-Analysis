"""
ML-based Anomaly Detection, District Clustering and Anomaly-Aware Forecasting
for Punjab School Enrollment Data (1989-2022)

Steps:
  1. Load and clean 4 wide CSV files -> convert to tidy long format
  2. Compute derived metrics (Total enrollment, Girls %, Gender Parity Index)
  3. Feature engineering (YoY % change, duplicate flag, rolling z-score)
  4. Isolation Forest anomaly detection & validation
  5. Exploratory Data Analysis (EDA) visualizations
  6. District clustering using K-Means
  7. Anomaly-aware ARIMA forecasting for 2023-2027
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
DATA_DIR = "punjab_dashboard"
OUT_DIR = "outputs"
DASHBOARD_DATA_DIR = os.path.join("punjab_dashboard", "data")

FILES = {
    ("1-5", "Boys"): "1-5boys.csv",
    ("1-5", "Girls"): "1-5girls.csv",
    ("6-8", "Boys"): "6-8boys.csv",
    ("6-8", "Girls"): "6-8girls.csv",
}

# New districts created over time (their enrollment was recorded in parent districts earlier)
NEW_DISTRICTS = [
    "Pathankot", "Fazilka", "SAS Nagar", "Tarn Taran", "Barnala",
    "Moga", "SBS Nagar", "Sri Muktsar Sahib", "Malerkotla",
    "Fatehgarh Sahib", "Mansa"
]

EXCLUDE_FROM_MODEL = ["Malerkotla"]  # Created in 2021, only 1 year of data
TRAIN_END = 2014
TEST_START, TEST_END = 2015, 2019
FORECAST_YEARS = 5                   # 2023 to 2027
K_CLUSTERS = 4

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DASHBOARD_DATA_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


def save_plot(filename):
    """Save current matplotlib figure to outputs directory."""
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
# 1. LOAD, CLEAN, RESHAPE
# ----------------------------------------------------------------------------
def load_all():
    """Load 4 raw CSVs, clean numeric strings, and reshape to long format."""
    frames = []
    totals = []

    for (band, gender), fname in FILES.items():
        filepath = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(filepath, na_values=["NA", "N/A", "-", ""])

        # Clean district and column names
        df = df.rename(columns={df.columns[0]: "District"})
        df["District"] = df["District"].astype(str).str.strip()
        df.columns = [str(c).strip() for c in df.columns]

        # Convert numbers: remove commas and treat 0 as missing placeholder
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
            df[col] = df[col].where(df[col] > 0)

        # Separate state summary row from district rows
        is_total = df["District"].str.lower().str.contains("total|punjab|state")
        long_df = df.melt(id_vars="District", var_name="Year", value_name="Enrollment")
        long_df["Year"] = long_df["Year"].astype(int)
        long_df["ClassGroup"] = band
        long_df["Gender"] = gender
        long_df["IsNewDistrict"] = long_df["District"].isin(NEW_DISTRICTS)

        district_data = long_df[~long_df["District"].isin(df.loc[is_total, "District"])]
        total_data = long_df[long_df["District"].isin(df.loc[is_total, "District"])]

        frames.append(district_data)
        totals.append(total_data)

    data = pd.concat(frames, ignore_index=True)
    src_totals = pd.concat(totals, ignore_index=True)
    return data, src_totals


def build_derived(data):
    """Compute Total, Girls Share %, and Gender Parity Index (GPI)."""
    wide = data.pivot_table(
        index=["District", "Year", "ClassGroup"],
        columns="Gender",
        values="Enrollment"
    ).reset_index()

    wide["Total"] = wide["Boys"] + wide["Girls"]
    wide["GirlsShare"] = (wide["Girls"] / wide["Total"]) * 100
    wide["GPI"] = wide["Girls"] / wide["Boys"]
    return wide


# ----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ----------------------------------------------------------------------------
def make_features(data):
    """Compute YoY % change, duplicate value flag, and rolling z-scores."""
    records = []
    keys = ["District", "ClassGroup", "Gender"]

    for _, group in data.dropna(subset=["Enrollment"]).sort_values("Year").groupby(keys):
        group = group.copy()
        prev = group["Enrollment"].shift(1)

        # YoY percentage growth
        group["yoy_pct"] = (group["Enrollment"] - prev) / prev * 100

        # Flag if current year exactly duplicates previous year
        group["dup_prev"] = (group["Enrollment"] == prev).astype(int)

        # 5-year centered rolling z-score to spot spikes/dips
        roll = group["Enrollment"].rolling(5, center=True, min_periods=3)
        roll_std = roll.std().replace(0, np.nan)
        group["roll_z"] = (group["Enrollment"] - roll.mean()) / roll_std

        records.append(group)

    return pd.concat(records, ignore_index=True)


# ----------------------------------------------------------------------------
# 3. ISOLATION FOREST ANOMALY DETECTION
# ----------------------------------------------------------------------------
def detect_anomalies(feats):
    """Train Isolation Forest on engineered features to detect anomalies."""
    features = ["yoy_pct", "dup_prev", "roll_z"]
    model_df = feats[~feats["District"].isin(EXCLUDE_FROM_MODEL)].dropna(subset=features).copy()
    X = model_df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    model = IsolationForest(n_estimators=300, contamination=0.05, random_state=42)
    model.fit(X)

    model_df["anomaly"] = (model.predict(X) == -1).astype(int)
    model_df["score"] = -model.score_samples(X)

    print(f"[Isolation Forest] Scored {len(model_df)} rows, Flagged: {model_df['anomaly'].sum()} anomalies")
    return model_df


def validate(model_df):
    """Validate model results against documented known historical issues."""
    flagged = model_df[model_df["anomaly"] == 1]
    print("\n=== VALIDATION AGAINST KNOWN ISSUES ===")

    # Check 2011-2013 reporting volatility block
    blk = model_df[model_df["Year"].between(2011, 2013)]
    print(f"2011-2013 volatility: {blk['anomaly'].sum()} / {len(blk)} rows flagged")

    # Check 2017/2018 carry-forward duplicates
    dup1718 = model_df[(model_df["Year"] == 2018) & (model_df["dup_prev"] == 1)]
    recall = dup1718["anomaly"].mean() if len(dup1718) else 0.0
    print(f"2017/2018 carry-over duplicates: {len(dup1718)} rows, Recall: {recall:.1%}")

    # Check newly discovered duplicates outside 2018
    new_dups = flagged[(flagged["dup_prev"] == 1) & (flagged["Year"] != 2018)]
    print(f"Candidate duplicates outside 2018: {len(new_dups)}")

    new_dups.to_csv(os.path.join(OUT_DIR, "new_duplicate_findings.csv"), index=False)
    flagged.sort_values("score", ascending=False).to_csv(
        os.path.join(OUT_DIR, "all_flagged_anomalies.csv"), index=False
    )


# ----------------------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------------------------------------------------------------
def eda(data, wide, model_df):
    """Generate and save the 4 primary EDA charts."""
    # 1. State-wide enrollment trend (Boys vs Girls)
    st = data.groupby(["Year", "Gender"])["Enrollment"].sum(min_count=1).unstack().dropna()
    anomaly_counts = model_df[model_df.anomaly == 1].groupby("Year").size()

    plt.figure(figsize=(11, 5))
    plt.plot(st.index, st["Boys"], label="Boys", color="#2563eb", lw=2)
    plt.plot(st.index, st["Girls"], label="Girls", color="#db2777", lw=2)
    for y in anomaly_counts.index:
        if y in st.index:
            plt.axvline(y, color="red", alpha=0.08 + 0.02 * min(anomaly_counts[y], 10))
    plt.title("Punjab State-Wide Enrollment (Red lines = Anomaly years)")
    plt.xlabel("Year")
    plt.ylabel("Enrollment")
    plt.legend()
    save_plot("01_statewide_trend.png")

    # 2. State-wide Gender Parity Index by class group
    sg = data.groupby(["Year", "ClassGroup", "Gender"])["Enrollment"].sum(min_count=1).unstack()
    sg["GPI"] = sg["Girls"] / sg["Boys"]
    sg = sg.reset_index().dropna(subset=["GPI"])

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=sg, x="Year", y="GPI", hue="ClassGroup", palette={"1-5": "#16a34a", "6-8": "#ea580c"})
    plt.axhline(1.0, ls="--", color="gray", label="Parity (1.0)")
    plt.title("State-Wide Gender Parity Index by Class Band")
    plt.xlabel("Year")
    plt.ylabel("GPI (Girls / Boys)")
    plt.legend()
    save_plot("02_state_gpi.png")

    # 3. Continuous history districts GPI Heatmap
    comb = data.groupby(["District", "Year", "Gender"])["Enrollment"].sum(min_count=1).unstack()
    comb["GPI"] = comb["Girls"] / comb["Boys"]
    hm = comb["GPI"].unstack("Year")
    # Only districts with complete records (ignoring 2008 which is missing everywhere)
    continuous_districts = hm.dropna(axis=0, thresh=hm.shape[1] - 1)

    plt.figure(figsize=(14, 6))
    sns.heatmap(continuous_districts, cmap="RdYlGn", center=0.9, cbar_kws={'label': 'GPI'})
    plt.title("District x Year Gender Parity Index (Districts with Full 1989-2022 History)")
    plt.xlabel("Year")
    plt.ylabel("District")
    save_plot("03_gpi_heatmap.png")

    # 4. Number of flagged anomalies per year
    all_years = sorted(model_df["Year"].unique())
    full_counts = anomaly_counts.reindex(all_years, fill_value=0)

    plt.figure(figsize=(11, 4))
    full_counts.plot(kind="bar", color="firebrick")
    plt.title("Machine-Learning Flagged Anomalies by Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Anomalies")
    save_plot("04_anomalies_by_year.png")


# ----------------------------------------------------------------------------
# 5. DISTRICT CLUSTERING (K-MEANS)
# ----------------------------------------------------------------------------
def get_slope(y, x):
    """Linear regression slope helper."""
    mask = ~np.isnan(y)
    return np.polyfit(x[mask], y[mask], 1)[0] if mask.sum() >= 3 else np.nan


def district_features(data, model_df):
    """Create summary features per district for clustering."""
    comb = data.groupby(["District", "Year", "Gender"])["Enrollment"].sum(min_count=1).unstack()
    comb["Total"] = comb["Boys"] + comb["Girls"]
    comb["GPI"] = comb["Girls"] / comb["Boys"]

    rows = []
    for district, group in comb.groupby(level=0):
        group = group.droplevel(0).sort_index().dropna(subset=["GPI", "Total"])
        if len(group) < 6 or district in EXCLUDE_FROM_MODEL:
            continue

        years = group.index.values.astype(float)
        normalized_total = group["Total"].values / group["Total"].mean()

        rows.append({
            "District": district,
            "avg_gpi": group["GPI"].mean(),
            "gpi_change": group["GPI"].iloc[-3:].mean() - group["GPI"].iloc[:3].mean(),
            "gpi_slope": get_slope(group["GPI"].values, years),
            "enroll_slope_norm": get_slope(normalized_total, years),
            "enroll_volatility": pd.Series(group["Total"].values).pct_change().std(),
            "n_anomalies": int(model_df[(model_df.District == district) & (model_df.anomaly == 1)].shape[0]),
        })

    return pd.DataFrame(rows).set_index("District")


def cluster(feat):
    """Cluster districts using K-Means and evaluate silhouette scores."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feat.fillna(feat.mean()))

    print("\n=== SILHOUETTE SCORES ===")
    for k in range(2, 7):
        labels = KMeans(k, n_init=20, random_state=42).fit_predict(X_scaled)
        print(f"k={k}: {silhouette_score(X_scaled, labels):.3f}")

    km = KMeans(K_CLUSTERS, n_init=20, random_state=42).fit(X_scaled)
    feat = feat.copy()
    feat["cluster"] = km.labels_
    feat.to_csv(os.path.join(OUT_DIR, "district_clusters.csv"))

    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=feat, x="avg_gpi", y="enroll_volatility", hue="cluster",
                    size="n_anomalies", palette="tab10", sizes=(40, 300))
    for district, row in feat.iterrows():
        plt.text(row.avg_gpi, row.enroll_volatility, district, fontsize=7)
    plt.title("District Typology Clustering (K-Means)")
    plt.xlabel("Average GPI")
    plt.ylabel("Enrollment Volatility")
    save_plot("05_cluster_scatter.png")
    return feat


# ----------------------------------------------------------------------------
# 6. ANOMALY-AWARE FORECASTING
# ----------------------------------------------------------------------------
def calculate_mape(actual, forecast):
    """Calculate Mean Absolute Percentage Error (MAPE)."""
    a, f = np.asarray(actual), np.asarray(forecast)
    return np.mean(np.abs((a - f) / a)) * 100


def forecasting(data, model_df):
    """Compare naive ARIMA vs anomaly-aware ARIMA, then forecast 2023-2027."""
    # State-wide total enrollment series
    total_series = data.groupby("Year")["Enrollment"].sum(min_count=1)
    all_years = range(int(total_series.index.min()), int(total_series.index.max()) + 1)
    total_series = total_series.reindex(all_years).interpolate(limit_direction="both")

    # Replace anomalous block (2011-2013) with interpolation for anomaly-aware model
    anomaly_years = [2011, 2012, 2013]
    clean_series = total_series.copy()
    clean_series.loc[anomaly_years] = np.nan
    clean_series = clean_series.interpolate(limit_direction="both")

    # Train / test split (Validation 2015-2019)
    train_naive = total_series[total_series.index <= TRAIN_END]
    train_aware = clean_series[clean_series.index <= TRAIN_END]
    test_actual = total_series[(total_series.index >= TEST_START) & (total_series.index <= TEST_END)]
    horizon = len(test_actual)

    naive_forecast = ARIMA(train_naive, order=(1, 1, 1)).fit().forecast(horizon)
    aware_forecast = ARIMA(train_aware, order=(1, 1, 1)).fit().forecast(horizon)

    mape_naive = calculate_mape(test_actual, naive_forecast)
    mape_aware = calculate_mape(test_actual, aware_forecast)

    print(f"\n=== VALIDATION MAPE (2015-2019) ===")
    print(f"Naive ARIMA(1,1,1):         {mape_naive:.1f}%")
    print(f"Anomaly-aware ARIMA(1,1,1): {mape_aware:.1f}%")

    # Validation plot
    plt.figure(figsize=(11, 5))
    plt.plot(total_series.index, total_series.values, label="Actual Enrollment", color="black")
    plt.plot(test_actual.index, naive_forecast.values, "--", label=f"Naive ARIMA (MAPE: {mape_naive:.1f}%)")
    plt.plot(test_actual.index, aware_forecast.values, "--", label=f"Anomaly-Aware (MAPE: {mape_aware:.1f}%)")
    plt.axvspan(2010.5, 2013.5, color="red", alpha=0.1, label="Anomalous Block (2011-2013)")
    plt.title("Forecast Validation: Naive vs Anomaly-Aware ARIMA")
    plt.xlabel("Year")
    plt.ylabel("Enrollment")
    plt.legend()
    save_plot("06_forecast_validation.png")

    # Final forecast for 2023-2027
    model_full = ARIMA(clean_series, order=(1, 1, 1)).fit()
    forecast_obj = model_full.get_forecast(FORECAST_YEARS)
    ci = forecast_obj.conf_int(alpha=0.05)

    last_year = int(clean_series.index.max())
    future_years = range(last_year + 1, last_year + 1 + FORECAST_YEARS)
    forecast_df = pd.DataFrame({
        "Year": future_years,
        "Forecast": forecast_obj.predicted_mean.values,
        "Lower95": ci.iloc[:, 0].values,
        "Upper95": ci.iloc[:, 1].values
    })

    print("\n=== FINAL FORECAST (2023-2027) ===")
    print(forecast_df.round(0).to_string(index=False))
    forecast_df.to_csv(os.path.join(OUT_DIR, "forecast_2023_2027.csv"), index=False)

    plt.figure(figsize=(11, 5))
    plt.plot(clean_series.index, clean_series.values, label="Cleaned History", color="black")
    plt.plot(forecast_df["Year"], forecast_df["Forecast"], "r--", label="5-Year Forecast")
    plt.fill_between(forecast_df["Year"], forecast_df["Lower95"], forecast_df["Upper95"],
                     color="red", alpha=0.2, label="95% Confidence Interval")
    plt.title("Punjab State-Wide Enrollment Forecast (2023-2027)")
    plt.xlabel("Year")
    plt.ylabel("Enrollment")
    plt.legend()
    save_plot("07_final_forecast.png")


# ----------------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------------
def main():
    print("Starting Punjab Enrollment Data Pipeline...")
    data, src_totals = load_all()
    print(f"Loaded: {data.shape[0]} records across {data['District'].nunique()} districts")

    # Export cleaned long format dataset for dashboard and analysis
    data.to_csv(os.path.join(OUT_DIR, "consolidated_long.csv"), index=False)
    data.to_csv(os.path.join(DASHBOARD_DATA_DIR, "punjab_district_long.csv"), index=False)

    # Export state-level summary
    state_level = data.groupby(["Year", "ClassGroup", "Gender"])["Enrollment"].sum(min_count=1).unstack().reset_index()
    state_level["Total"] = state_level["Boys"] + state_level["Girls"]
    state_level["GPI"] = state_level["Girls"] / state_level["Boys"]
    state_level = state_level.dropna(subset=["Total"])
    state_level.to_csv(os.path.join(OUT_DIR, "punjab_state_level.csv"), index=False)
    state_level.to_csv(os.path.join(DASHBOARD_DATA_DIR, "punjab_state_level.csv"), index=False)

    wide = build_derived(data)
    wide.to_csv(os.path.join(OUT_DIR, "derived_gpi_table.csv"), index=False)

    # Run ML Anomaly Detection
    feats = make_features(data)
    model_df = detect_anomalies(feats)
    validate(model_df)

    # Run EDA
    eda(data, wide, model_df)

    # Run Clustering
    dfeat = district_features(data, model_df)
    cluster(dfeat)

    # Run Forecasting
    forecasting(data, model_df)

    print(f"\nAll operations completed successfully! Outputs saved to '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
