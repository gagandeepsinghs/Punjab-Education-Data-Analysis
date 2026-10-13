"""
PDF Documentation Generator for Punjab Education Data Analysis Project
Uses ReportLab to produce a publication-grade PDF report.
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
PDF_PATH = BASE_DIR / "PROJECT_DOCUMENTATION.pdf"

def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1e3a8a'), # Deep Navy
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#4b5563'),
        spaceAfter=15,
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#1e3a8a'),
        spaceBefore=12,
        spaceAfter=8,
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#0f766e'), # Teal
        spaceBefore=8,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#1f2937'),
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        'BulletText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor('#374151'),
        leftIndent=15,
        spaceAfter=3,
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.white,
        alignment=0,
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#1f2937'),
    )

    table_code_style = ParagraphStyle(
        'TableCode',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#1d4ed8'),
    )

    story = []

    # Title & Header
    story.append(Paragraph("Punjab School Enrollment Analysis", title_style))
    story.append(Paragraph("Comprehensive Project Documentation & Technical Architecture (1989–2022)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1e3a8a'), spaceAfter=15))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        "This project analyzes 34 years of district-wise, gender-disaggregated school enrollment records "
        "(Classes 1–5 and Classes 6–8) in Punjab, India. The workflow integrates clean automated data engineering, "
        "unsupervised Machine Learning for reporting anomaly detection and district clustering, anomaly-aware "
        "ARIMA econometric time-series forecasting, and an interactive Streamlit web explorer.",
        body_style
    ))
    story.append(Spacer(1, 8))

    # Section 2: Libraries & Technologies Used and Their Purpose
    story.append(Paragraph("2. Libraries and Technologies Used in the Project", h1_style))
    story.append(Paragraph(
        "Each library was deliberately chosen to serve a specific role within the analysis pipeline or user dashboard:",
        body_style
    ))

    lib_data = [
        [
            Paragraph("Library", table_header_style),
            Paragraph("Category", table_header_style),
            Paragraph("Specific Role and Purpose in This Project", table_header_style)
        ],
        [
            Paragraph("pandas", table_code_style),
            Paragraph("Data Processing", table_cell_style),
            Paragraph("Ingests wide CSV tables; unpivots to tidy long-format records (melt); handles missing records; computes derived aggregates (Total, Girls %, GPI); builds rolling window statistics.", table_cell_style)
        ],
        [
            Paragraph("numpy", table_code_style),
            Paragraph("Numerical Computing", table_cell_style),
            Paragraph("Performs fast vectorized array computations, polyfit regression slopes for long-term district trend calculation, and Mean Absolute Percentage Error (MAPE) mathematical formulas.", table_cell_style)
        ],
        [
            Paragraph("scikit-learn", table_code_style),
            Paragraph("Machine Learning", table_cell_style),
            Paragraph("Powers Isolation Forest (300 estimators) for unsupervised anomaly scoring; K-Means clustering (k=4) for district typology; StandardScaler for feature normalization; and Silhouette Score validation.", table_cell_style)
        ],
        [
            Paragraph("statsmodels", table_code_style),
            Paragraph("Time Series Modeling", table_cell_style),
            Paragraph("Fits Autoregressive Integrated Moving Average (ARIMA 1,1,1) models on enrollment time series; validates naive vs anomaly-aware forecasts; generates 2023–2027 forecasts with 95% confidence intervals.", table_cell_style)
        ],
        [
            Paragraph("matplotlib", table_code_style),
            Paragraph("Static Visualization", table_cell_style),
            Paragraph("Renders publication-ready high-resolution static charts (trend curves, anomaly bar charts, ARIMA validation, forecast confidence bands) saved directly into outputs/.", table_cell_style)
        ],
        [
            Paragraph("seaborn", table_code_style),
            Paragraph("Statistical Plotting", table_cell_style),
            Paragraph("Creates divergent color-scaled District x Year Gender Parity Index heatmaps (RdYlGn) and styled multi-attribute cluster scatter plots with bubble scaling.", table_cell_style)
        ],
        [
            Paragraph("streamlit", table_code_style),
            Paragraph("Web Dashboard", table_cell_style),
            Paragraph("Powers the interactive web explorer (punjab_dashboard/app.py) with dynamic filters, multi-district selectors, responsive KPI metric cards, and multi-view navigation tabs.", table_cell_style)
        ],
        [
            Paragraph("plotly", table_code_style),
            Paragraph("Interactive Charts", table_cell_style),
            Paragraph("Embeds interactive charts with custom hover data tooltips, cross-filtering, and dynamic zooming within the Streamlit dashboard.", table_cell_style)
        ],
        [
            Paragraph("reportlab", table_code_style),
            Paragraph("PDF Generation", table_cell_style),
            Paragraph("Compiles and styles this comprehensive documentation directly into a standalone, formatted PDF document.", table_cell_style)
        ],
    ]

    t_lib = Table(lib_data, colWidths=[80, 85, 365])
    t_lib.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_lib)
    story.append(Spacer(1, 12))

    # Page Break for clean multi-page layout
    story.append(PageBreak())

    # Section 3: Technical Architecture & Pipeline
    story.append(Paragraph("3. Technical Architecture & Methodology", h1_style))
    story.append(Paragraph(
        "The project pipeline transforms heterogeneous government data into actionable analytical products:",
        body_style
    ))
    story.append(Paragraph("<b>Step 1: Data Cleaning & Normalization</b> — Converts 4 separate wide tables into 3,128 standardized observations. Missing placeholders ('NA', '-', empty strings) and zero placeholders are converted to NaN.", bullet_style))
    story.append(Paragraph("<b>Step 2: Metric Derivation</b> — Computes Total Enrollment, Girls Share %, and Gender Parity Index (GPI = Girls / Boys). Parity is achieved at 1.0.", bullet_style))
    story.append(Paragraph("<b>Step 3: Feature Engineering</b> — Calculates YoY percentage changes, exact carry-over duplicate flags, and 5-year rolling standard deviation z-scores.", bullet_style))
    story.append(Paragraph("<b>Step 4: Isolation Forest Anomaly Detection</b> — Flags multi-attribute outliers with 300 estimators and 5% contamination, detecting known reporting spikes and discovering duplicate entries.", bullet_style))
    story.append(Paragraph("<b>Step 5: District Typology Clustering</b> — Normalizes long-term GPI trajectories, rates of change, and enrollment volatility, grouping the 22 eligible districts into 4 distinct behavior clusters using K-Means.", bullet_style))
    story.append(Paragraph("<b>Step 6: Anomaly-Aware Time Series Forecasting</b> — Demonstrates that handling the artificial 2011–2013 reporting spike reduces ARIMA validation forecast error from 65.6% down to 5.7%. Projections with 95% confidence intervals are forecasted for 2023–2027.", bullet_style))
    story.append(Spacer(1, 10))

    # Section 4: Key Analytical Findings
    story.append(Paragraph("4. Key Findings & Forecast Projections", h1_style))

    # Forecast table
    fc_data = [
        [
            Paragraph("Year", table_header_style),
            Paragraph("Projected Enrollment", table_header_style),
            Paragraph("95% CI Lower Bound", table_header_style),
            Paragraph("95% CI Upper Bound", table_header_style),
        ],
        [Paragraph("2023", table_cell_style), Paragraph("1,732,471", table_cell_style), Paragraph("1,402,997", table_cell_style), Paragraph("2,061,945", table_cell_style)],
        [Paragraph("2024", table_cell_style), Paragraph("1,732,369", table_cell_style), Paragraph("1,258,083", table_cell_style), Paragraph("2,206,655", table_cell_style)],
        [Paragraph("2025", table_cell_style), Paragraph("1,732,313", table_cell_style), Paragraph("1,144,341", table_cell_style), Paragraph("2,320,286", table_cell_style)],
        [Paragraph("2026", table_cell_style), Paragraph("1,732,283", table_cell_style), Paragraph("1,047,535", table_cell_style), Paragraph("2,417,032", table_cell_style)],
        [Paragraph("2027", table_cell_style), Paragraph("1,732,267", table_cell_style), Paragraph("961,965", table_cell_style), Paragraph("2,502,569", table_cell_style)],
    ]
    t_fc = Table(fc_data, colWidths=[70, 150, 155, 155])
    t_fc.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f766e')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0fdf4')]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_fc)
    story.append(Spacer(1, 10))

    # District clusters summary
    story.append(Paragraph("<b>District Typology Clusters (K-Means, k=4):</b>", h2_style))
    story.append(Paragraph("• <b>Cluster 0 (Core Industrial Plains)</b>: Amritsar, Fatehgarh Sahib, Gurdaspur, Hoshiarpur, Jalandhar, Kapurthala, Ludhiana, Moga, Rupnagar, SBS Nagar. High total enrollment, consistent GPI progress.", bullet_style))
    story.append(Paragraph("• <b>Cluster 1 (High Volatility / Fast Developing)</b>: Barnala, Fazilka, SAS Nagar, Sangrur, Tarn Taran. Dynamic demographic shifts and rapid expansion.", bullet_style))
    story.append(Paragraph("• <b>Cluster 2 (Rapid GPI Catch-Up)</b>: Bathinda, Faridkot, Ferozepur, Mansa, Patiala, Sri Muktsar Sahib. Historically low baseline GPI in 1989 that exhibited the steepest gains over three decades.", bullet_style))
    story.append(Paragraph("• <b>Cluster 3 (Sub-Mountainous Outlier)</b>: Pathankot. Established in 2011 with distinct geographic and scale characteristics.", bullet_style))
    story.append(Spacer(1, 10))

    # Section 5: Data Quality Audit
    story.append(Paragraph("5. Documented Data Irregularities & Handling", h1_style))
    story.append(Paragraph("• <b>2017 = 2018 Carry-Forward:</b> In 98.9% of district/band combinations, 2017 numbers were repeated identically in 2018 in source reports. The Isolation Forest model detected this with 70.1% recall.", bullet_style))
    story.append(Paragraph("• <b>2011–2013 Volatility Artifact:</b> Simultaneous multi-district enrollment spike in 2011 & 2013 with a trough in 2012, reflecting an administrative census/reporting anomaly rather than demographic change.", bullet_style))
    story.append(Paragraph("• <b>Missing Year 2008:</b> Completely absent across all districts in source records. Preserved as NaN and interpolated for continuous time-series modeling.", bullet_style))
    story.append(Paragraph("• <b>Administrative Carve-Outs:</b> New districts (Malerkotla 2021, Pathankot 2011, etc.) are maintained as NaN prior to formation rather than misleading zero enrollment.", bullet_style))
    story.append(Spacer(1, 10))

    # Section 6: How to Run
    story.append(Paragraph("6. Project Execution Guide", h1_style))
    story.append(Paragraph("<b>To run the ML Pipeline:</b> <code>.\\.venv\\Scripts\\python.exe punjab_enrollment_pipeline.py</code>", bullet_style))
    story.append(Paragraph("<b>To launch the Streamlit Dashboard:</b> <code>.\\.venv\\Scripts\\python.exe -m streamlit run punjab_dashboard/app.py</code>", bullet_style))
    story.append(Paragraph("Open <code>http://localhost:8501</code> in your browser to access the interactive web explorer.", bullet_style))

    # Add embedded chart if available
    chart_path = OUTPUTS_DIR / "06_forecast_validation.png"
    if chart_path.is_file():
        story.append(Spacer(1, 10))
        story.append(Paragraph("<b>Key Validation Figure (Naive vs Anomaly-Aware ARIMA):</b>", h2_style))
        story.append(Image(str(chart_path), width=500, height=220))

    doc.build(story)
    print(f"PDF successfully built: {PDF_PATH}")

if __name__ == "__main__":
    build_pdf()
