"""
Punjab Education Enrollment Explorer
An interactive Streamlit dashboard for exploring 1989-2022 district-wise,
gender-disaggregated school enrollment data (Classes 1-5 and 6-8) for Punjab.

Run with:  streamlit run app.py
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ROOT_DIR = BASE_DIR.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"

st.set_page_config(page_title="Punjab Education Enrollment Explorer",
                   page_icon="📚", layout="wide")

# ------------------------------------------------------------------
# Data loading (cached so it only reads from disk once per session)
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    district_path = DATA_DIR / "punjab_district_long.csv"
    state_path = DATA_DIR / "punjab_state_level.csv"

    # Fallback to outputs folder if not found in data
    if not district_path.is_file() and (OUTPUTS_DIR / "consolidated_long.csv").is_file():
        district_path = OUTPUTS_DIR / "consolidated_long.csv"
    if not state_path.is_file() and (OUTPUTS_DIR / "punjab_state_level.csv").is_file():
        state_path = OUTPUTS_DIR / "punjab_state_level.csv"

    district = pd.read_csv(district_path)
    state = pd.read_csv(state_path)
    return district, state

district_df, state_df = load_data()

ALL_DISTRICTS = sorted(district_df["District"].dropna().unique())
NEW_DISTRICTS = {
    "Pathankot": 2011, "Fazilka": 2011, "SAS Nagar": 2006,
    "Tarn Taran": 2006, "Barnala": 2006, "Moga": 1995,
    "SBS Nagar": 1995, "Sri Muktsar Sahib": 1995, "Malerkotla": 2021,
    "Fatehgarh Sahib": 1992, "Mansa": 1992,
}

# ------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------
st.sidebar.title("📚 Filters")

class_group = st.sidebar.radio(
    "Class group", ["Both (1–8)", "Classes 1–5", "Classes 6–8"], index=0
)
group_map = {"Both (1–8)": None, "Classes 1–5": "1-5", "Classes 6–8": "6-8"}
selected_group = group_map[class_group]

valid_years = district_df["Year"].dropna().astype(int)
year_min, year_max = int(valid_years.min()), int(valid_years.max())
year_range = st.sidebar.slider("Year range", year_min, year_max, (year_min, year_max))

view_mode = st.sidebar.radio(
    "View", ["Punjab Overview", "District Comparison", "Single District Deep-Dive",
             "GPI Heatmap", "ML Anomalies & Forecasting", "Data Quality Notes"]
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data source: District-wise school enrollment records, Punjab, "
    "Classes 1–5 and 6–8, boys/girls, 1989–2022."
)

# ------------------------------------------------------------------
# Helper to filter by class group + year range
# ------------------------------------------------------------------
def filter_district(df, group=None, districts=None, yrange=None):
    out = df.copy()
    if group:
        out = out[out["ClassGroup"] == group]
    if districts:
        out = out[out["District"].isin(districts)]
    if yrange:
        out = out[(out["Year"] >= yrange[0]) & (out["Year"] <= yrange[1])]
    return out

def filter_state(df, group=None, yrange=None):
    out = df.copy()
    if group:
        out = out[out["ClassGroup"] == group]
    if yrange:
        out = out[(out["Year"] >= yrange[0]) & (out["Year"] <= yrange[1])]
    return out

st.title("Punjab School Enrollment Explorer (1989–2022)")
st.caption("Classes 1–5 and 6–8 · Boys & Girls · 23 Districts")

# ==================================================================
# VIEW 1: Punjab Overview
# ==================================================================
if view_mode == "Punjab Overview":
    sdf = filter_state(state_df, selected_group, year_range)
    if sdf.empty:
        st.warning(f"No state-level data available for year range {year_range[0]}–{year_range[1]}. (Note: 2008 has no recorded data across Punjab).")
    else:
        agg = sdf.groupby("Year")[["Boys", "Girls"]].sum(min_count=1).reset_index()
        agg = agg.dropna(subset=["Boys", "Girls"])

        if agg.empty:
            st.warning("No complete records found for the selected filter range.")
        else:
            agg["Total"] = agg["Boys"] + agg["Girls"]
            agg["GPI"] = agg["Girls"] / agg["Boys"]

            c1, c2, c3, c4 = st.columns(4)
            if len(agg) > 1:
                base_yr = int(agg["Year"].iloc[0])
                c1.metric("Latest Total Enrollment", f"{agg['Total'].iloc[-1]:,.0f}",
                           f"{agg['Total'].iloc[-1] - agg['Total'].iloc[0]:,.0f} vs {base_yr}")
                c2.metric("Latest GPI", f"{agg['GPI'].iloc[-1]:.3f}",
                           f"{agg['GPI'].iloc[-1] - agg['GPI'].iloc[0]:+.3f} vs {base_yr}")
            else:
                c1.metric("Latest Total Enrollment", f"{agg['Total'].iloc[-1]:,.0f}")
                c2.metric("Latest GPI", f"{agg['GPI'].iloc[-1]:.3f}")

            c3.metric("Boys (latest)", f"{agg['Boys'].iloc[-1]:,.0f}")
            c4.metric("Girls (latest)", f"{agg['Girls'].iloc[-1]:,.0f}")

            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=agg["Year"], y=agg["Boys"], name="Boys",
                                          line=dict(color="#2563eb", width=2.5)))
                fig.add_trace(go.Scatter(x=agg["Year"], y=agg["Girls"], name="Girls",
                                          line=dict(color="#db2777", width=2.5)))
                fig.update_layout(title="Total Enrollment: Boys vs Girls", xaxis_title="Year",
                                   yaxis_title="Enrollment", height=420)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=agg["Year"], y=agg["GPI"], name="GPI",
                                           line=dict(color="#16a34a", width=2.5)))
                fig2.add_hline(y=1.0, line_dash="dash", line_color="gray",
                                annotation_text="Parity (1.0)")
                fig2.update_layout(title="Gender Parity Index (GPI) Over Time",
                                    xaxis_title="Year", yaxis_title="GPI (Girls / Boys)", height=420)
                st.plotly_chart(fig2, use_container_width=True)

            if selected_group is None:
                st.subheader("GPI by Class Group")
                gsub = filter_state(state_df, None, year_range).dropna(subset=["GPI"])
                if not gsub.empty:
                    fig3 = px.line(gsub, x="Year", y="GPI", color="ClassGroup", markers=True,
                                    color_discrete_map={"1-5": "#16a34a", "6-8": "#ea580c"})
                    fig3.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Parity")
                    fig3.update_layout(height=400)
                    st.plotly_chart(fig3, use_container_width=True)

# ==================================================================
# VIEW 2: District Comparison
# ==================================================================
elif view_mode == "District Comparison":
    default_selection = [d for d in ["Ludhiana", "Amritsar", "Bathinda", "Patiala"] if d in ALL_DISTRICTS]
    districts = st.multiselect("Select districts to compare", ALL_DISTRICTS,
                                default=default_selection)
    if not districts:
        st.info("Select at least one district from the box above.")
    else:
        ddf = filter_district(district_df, selected_group, districts, year_range)
        # Using min_count=1 ensures all-NaN historical periods do not convert into misleading 0s
        agg = ddf.groupby(["District", "Year", "Gender"])["Enrollment"].sum(min_count=1).reset_index()
        valid_agg = agg.dropna(subset=["Enrollment"])

        if valid_agg.empty:
            st.warning("No data found for the selected districts within the chosen year range.")
        else:
            total = valid_agg.groupby(["District", "Year"])["Enrollment"].sum(min_count=1).reset_index()

            fig = px.line(total, x="Year", y="Enrollment", color="District", markers=True)
            fig.update_layout(title="Total Enrollment by District", height=450)
            st.plotly_chart(fig, use_container_width=True)

            # GPI comparison
            piv = valid_agg.pivot_table(index=["District", "Year"], columns="Gender",
                                        values="Enrollment").reset_index()
            for g_col in ["Boys", "Girls"]:
                if g_col not in piv.columns:
                    piv[g_col] = np.nan

            piv["GPI"] = piv["Girls"] / piv["Boys"]
            valid_gpi = piv.dropna(subset=["GPI"])

            if not valid_gpi.empty:
                fig2 = px.line(valid_gpi, x="Year", y="GPI", color="District", markers=True)
                fig2.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Parity")
                fig2.update_layout(title="GPI by District", height=450)
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Latest-Year Snapshot")
            latest_yr = int(total["Year"].max())
            snap = total[total["Year"] == latest_yr].sort_values("Enrollment", ascending=False)
            st.dataframe(snap, use_container_width=True, hide_index=True)

# ==================================================================
# VIEW 3: Single District Deep-Dive
# ==================================================================
elif view_mode == "Single District Deep-Dive":
    default_dist = "Ludhiana" if "Ludhiana" in ALL_DISTRICTS else ALL_DISTRICTS[0]
    dist = st.selectbox("Select a district", ALL_DISTRICTS, index=ALL_DISTRICTS.index(default_dist))
    ddf = filter_district(district_df, selected_group, [dist], year_range)

    # Use min_count=1 to preserve missing values instead of creating zeros
    agg = ddf.groupby(["Year", "Gender"])["Enrollment"].sum(min_count=1).reset_index()
    piv = agg.pivot_table(index="Year", columns="Gender", values="Enrollment").reset_index()

    for col in ["Boys", "Girls"]:
        if col not in piv.columns:
            piv[col] = np.nan

    piv["Total"] = piv["Boys"] + piv["Girls"]
    piv["GPI"] = piv["Girls"] / piv["Boys"]

    if dist in NEW_DISTRICTS:
        st.info(f"ℹ️ {dist} was carved out as a separate administrative district "
                f"around {NEW_DISTRICTS[dist]}. Earlier years show missing data "
                f"because its enrollment was previously recorded under its parent district.")

    valid_records = piv.dropna(subset=["Total"])
    if valid_records.empty:
        st.warning(f"No enrollment recorded for {dist} in the range {year_range[0]}–{year_range[1]}.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=valid_records["Year"], y=valid_records["Boys"], name="Boys", line=dict(color="#2563eb", width=2.5)))
            fig.add_trace(go.Scatter(x=valid_records["Year"], y=valid_records["Girls"], name="Girls", line=dict(color="#db2777", width=2.5)))
            fig.update_layout(title=f"{dist}: Enrollment Trend", xaxis_title="Year", yaxis_title="Enrollment", height=420)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = go.Figure()
            valid_gpi = valid_records.dropna(subset=["GPI"])
            fig2.add_trace(go.Scatter(x=valid_gpi["Year"], y=valid_gpi["GPI"], name="GPI", line=dict(color="#16a34a", width=2.5)))
            fig2.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Parity")
            fig2.update_layout(title=f"{dist}: GPI Trend", xaxis_title="Year", yaxis_title="GPI (Girls / Boys)", height=420)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Underlying data")
        st.dataframe(piv, use_container_width=True, hide_index=True)
        st.download_button("Download this district's data as CSV",
                            piv.to_csv(index=False), file_name=f"{dist}_enrollment.csv")

# ==================================================================
# VIEW 4: GPI Heatmap
# ==================================================================
elif view_mode == "GPI Heatmap":
    full_hist = ["Amritsar", "Bathinda", "Faridkot", "Ferozepur", "Gurdaspur",
                 "Hoshiarpur", "Jalandhar", "Kapurthala", "Ludhiana", "Patiala",
                 "Rupnagar", "Sangrur"]

    heatmap_scope = st.radio(
        "Districts to include",
        ["Continuous history districts only (12 districts)", "All districts (23 districts)"],
        horizontal=True,
    )
    if heatmap_scope.startswith("Continuous"):
        st.caption("Showing only the 12 continuous-history districts (1989–2022) for a clean, gap-free heatmap.")
        selected_districts = full_hist
    else:
        st.caption("Showing all 23 districts. Newer districts show empty cells prior to their administrative establishment.")
        selected_districts = ALL_DISTRICTS

    ddf = filter_district(district_df, selected_group, selected_districts, year_range)
    agg = ddf.groupby(["District", "Year", "Gender"])["Enrollment"].sum(min_count=1).reset_index()
    piv = agg.pivot_table(index=["District", "Year"], columns="Gender", values="Enrollment").reset_index()

    for col in ["Boys", "Girls"]:
        if col not in piv.columns:
            piv[col] = np.nan

    piv["GPI"] = piv["Girls"] / piv["Boys"]
    heat = piv.pivot_table(index="District", columns="Year", values="GPI")

    if not heat.empty:
        fig = px.imshow(heat, color_continuous_scale="RdYlGn", zmin=0.7, zmax=1.1,
                         aspect="auto", labels=dict(color="GPI"))
        fig.update_layout(title="District × Year Gender Parity Index", height=550)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available to construct the heatmap.")

# ==================================================================
# VIEW 5: ML Anomalies & Forecasting
# ==================================================================
elif view_mode == "ML Anomalies & Forecasting":
    st.subheader("Machine Learning Insights & Projections")
    st.markdown("""
    This view displays outputs produced by the ML analysis pipeline (`punjab_enrollment_pipeline.py`),
    featuring **Isolation Forest** anomaly detection, **K-Means** district typology clustering,
    and **Anomaly-Aware ARIMA** forecasting for 2023–2027.
    """)

    tab1, tab2, tab3 = st.tabs(["📈 ARIMA Forecasting", "🔍 District Typology (K-Means)", "🚩 Candidate Anomalies"])

    with tab1:
        st.markdown("#### State-Wide Total Enrollment Forecast (2023–2027)")
        forecast_csv = ROOT_DIR / "forecast_2023_2027.csv"
        if not forecast_csv.is_file():
            forecast_csv = OUTPUTS_DIR / "forecast_2023_2027.csv"

        if forecast_csv.is_file():
            f_df = pd.read_csv(forecast_csv)
            col_f1, col_f2 = st.columns([1, 2])
            with col_f1:
                st.write("**Forecast Projections (with 95% Confidence Interval)**")
                st.dataframe(f_df.round(0), hide_index=True)
            with col_f2:
                fig_fc = go.Figure()
                # Historical state line
                state_tot = state_df.groupby("Year")["Total"].sum().reset_index()
                fig_fc.add_trace(go.Scatter(x=state_tot["Year"], y=state_tot["Total"], name="Historical Actual", line=dict(color="#2563eb", width=2)))
                # Forecast line
                fig_fc.add_trace(go.Scatter(x=f_df["Year"], y=f_df["Forecast"], name="ARIMA Forecast", line=dict(color="#dc2626", dash="dash", width=2.5)))
                # Confidence interval
                fig_fc.add_trace(go.Scatter(x=f_df["Year"].tolist() + f_df["Year"].tolist()[::-1],
                                            y=f_df["Upper95"].tolist() + f_df["Lower95"].tolist()[::-1],
                                            fill="toself", fillcolor="rgba(220, 38, 38, 0.15)",
                                            line=dict(color="rgba(255,255,255,0)"),
                                            hoverinfo="skip", showlegend=True, name="95% CI"))
                fig_fc.update_layout(title="Historical Enrollment & Anomaly-Aware 5-Year Forecast", xaxis_title="Year", yaxis_title="Total Enrollment", height=420)
                st.plotly_chart(fig_fc, use_container_width=True)
        else:
            st.info("Run `python punjab_enrollment_pipeline.py` to generate the forecast tables.")

    with tab2:
        st.markdown("#### District Typology Clustering (k=4 K-Means)")
        cluster_csv = ROOT_DIR / "district_clusters.csv"
        if not cluster_csv.is_file():
            cluster_csv = OUTPUTS_DIR / "district_clusters.csv"

        if cluster_csv.is_file():
            c_df = pd.read_csv(cluster_csv)
            st.dataframe(c_df.round(3), hide_index=True)
            fig_cl = px.scatter(c_df, x="avg_gpi", y="enroll_volatility", color="cluster",
                                size="n_anomalies", hover_name="District", text="District",
                                title="District Typology: Average GPI vs Enrollment Volatility",
                                labels={"avg_gpi": "Average GPI (1989-2022)", "enroll_volatility": "Enrollment Volatility", "cluster": "Cluster"})
            fig_cl.update_traces(textposition='top center')
            fig_cl.update_layout(height=480)
            st.plotly_chart(fig_cl, use_container_width=True)
        else:
            st.info("Run `python punjab_enrollment_pipeline.py` to generate cluster profiles.")

    with tab3:
        st.markdown("#### Candidate Duplicate & Reporting Anomalies")
        dup_csv = ROOT_DIR / "new_duplicate_findings.csv"
        if not dup_csv.is_file():
            dup_csv = OUTPUTS_DIR / "new_duplicate_findings.csv"

        if dup_csv.is_file():
            d_df = pd.read_csv(dup_csv)
            st.caption("Duplicates flagged by the model outside of the known 2018 carry-over:")
            st.dataframe(d_df, hide_index=True)
        else:
            st.info("Run `python punjab_enrollment_pipeline.py` to view candidate duplicate findings.")

# ==================================================================
# VIEW 6: Data Quality Notes
# ==================================================================
elif view_mode == "Data Quality Notes":
    st.subheader("Known data quality issues in the source data")
    st.markdown("""
    This dashboard is built on real government-reported data, which has some
    genuine irregularities. These are documented here rather than silently
    corrected, in line with good data-analytics practice.
    """)
    st.warning(
        "**2017 vs 2018 duplication:** In 98.9% of district/class-group/gender "
        "combinations, the 2017 and 2018 values are exactly identical — strongly "
        "suggesting the 2017 report was carried forward as the 2018 figure in the "
        "source. Treat year-over-year comparisons across 2017→2018 with caution."
    )
    st.warning(
        "**2011–2013 volatility:** Total enrollment shows a sharp, implausible "
        "spike in 2011 and 2013 with a matching dip in 2012 across many districts "
        "simultaneously — consistent with a reporting/administrative anomaly "
        "rather than a genuine demographic swing. Flagged for exclusion or "
        "correction before any forecasting model is trained."
    )
    st.info(
        "**New district creation:** " +
        ", ".join(f"{d} (~{y})" for d, y in sorted(NEW_DISTRICTS.items(), key=lambda x: x[1])) +
        " — these districts show missing data before their creation year because "
        "their enrollment was recorded under their parent district at the time."
    )
    st.info(
        "**2008 is entirely missing:** every district, both class groups, and both "
        "genders show no data for 2008 in the source records — this year is dropped "
        "from all trend lines and computations rather than shown as zero."
    )

    st.subheader("Missing-data percentage by district")
    miss = (district_df[~district_df["District"].eq("Malerkotla")]
            .groupby("District")["Enrollment"]
            .apply(lambda s: s.isna().mean() * 100)
            .sort_values(ascending=False)
            .reset_index(name="Missing %"))
    st.dataframe(miss, use_container_width=True, hide_index=True)
