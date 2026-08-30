import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys
import time
import datetime
import psutil

# Import production backend model loader
import model_loader as loader

# Create directories if they do not exist
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="SANNIF – Semantic AI Network Intrusion Framework",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session States
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "df_validation" not in st.session_state:
    st.session_state.df_validation = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "detection_run" not in st.session_state:
    st.session_state.detection_run = False
if "kpi_data" not in st.session_state:
    st.session_state.kpi_data = None
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# ---------------------------------------------------------
# Custom CSS Styling Injection
# ---------------------------------------------------------
def inject_custom_css():
    st.markdown("""
    <style>
    /* CSS Root Theme Variables */
    :root {
        --primary: #2563EB;
        --accent: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --bg: #0F172A;
        --card: #1E293B;
        --border: #334155;
        --text: #FFFFFF;
        --text-muted: #94A3B8;
    }

    /* Global Overrides */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", sans-serif !important;
    }

    .block-container {
        padding: 2rem 2.5rem 3rem !important;
        max-width: 1400px !important;
    }

    /* Hide standard Streamlit header and footer */
    header[data-testid="stHeader"], footer, .stDeployButton {
        visibility: hidden !important;
        display: none !important;
    }

    /* Sticky Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #334155 !important;
    }
    
    /* Layout Column Spacing */
    [data-testid="stHorizontalBlock"] {
        gap: 1.25rem !important;
    }

    /* Custom Cards */
    .premium-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1.25rem;
        transition: transform 0.22s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.22s ease;
        animation: slideUpFade 0.45s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    .premium-card:hover {
        transform: translateY(-2px);
        border-color: #2563EB;
    }

    /* KPI Typography */
    .kpi-title {
        font-size: 0.72rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 0.3rem;
        letter-spacing: -0.03em;
    }
    .kpi-subtext {
        font-size: 0.72rem;
        color: #94A3B8;
        margin-top: 0.4rem;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* Hero Section Card */
    .hero-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2.25rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        animation: slideUpFade 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 850;
        color: #FFFFFF;
        letter-spacing: -0.04em;
        line-height: 1.1;
        margin-bottom: 0.25rem;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #2563EB;
        font-weight: 600;
        margin-bottom: 1rem;
        letter-spacing: 0.02em;
    }
    .hero-description {
        font-size: 0.92rem;
        color: #94A3B8;
        max-width: 750px;
        line-height: 1.55;
    }

    /* System Health Box in Sidebar */
    .health-container {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .health-title {
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
        letter-spacing: 0.08em;
    }
    .health-item {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.8rem;
        margin-bottom: 0.55rem;
        color: #E2E8F0;
    }
    .health-item:last-child {
        margin-bottom: 0;
    }
    .dot {
        height: 7px;
        width: 7px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-green {
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
    }
    .dot-red {
        background-color: #EF4444;
        box-shadow: 0 0 8px #EF4444;
    }

    /* Table Styling */
    .data-table-container {
        overflow-x: auto;
        border: 1px solid #334155;
        border-radius: 10px;
        background-color: #1E293B;
        margin-bottom: 1.5rem;
    }
    .data-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
    }
    .data-table th {
        background-color: #1E293B;
        color: #94A3B8;
        text-align: left;
        padding: 0.85rem 1rem;
        font-size: 0.72rem;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        border-bottom: 1px solid #334155;
    }
    .data-table td {
        padding: 0.85rem 1rem;
        font-size: 0.82rem;
        color: #E2E8F0;
        border-bottom: 1px solid #334155;
        background-color: #1E293B;
    }
    .data-table tr:last-child td {
        border-bottom: none;
    }
    .data-table tr:hover td {
        background-color: #24324D !important;
    }

    /* Color borders based on threat status */
    tr.status-normal td:first-child {
        border-left: 4px solid #10B981;
    }
    tr.status-suspicious td:first-child {
        border-left: 4px solid #F59E0B;
    }
    tr.status-attack td:first-child {
        border-left: 4px solid #EF4444;
    }

    /* Threat Status Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.15rem 0.55rem;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .badge-green {
        color: #10B981;
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .badge-orange {
        color: #F59E0B;
        background-color: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    .badge-red {
        color: #EF4444;
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.2);
    }

    /* Custom inline Progress Bar */
    .progress-container {
        display: flex;
        align-items: center;
        gap: 10px;
        width: 140px;
    }
    .progress-bg {
        background-color: #334155;
        height: 6px;
        border-radius: 3px;
        flex-grow: 1;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 3px;
        background-color: #2563EB;
        transition: width 0.4s ease-in-out;
    }

    /* Sidebar Navigation buttons customization */
    .stButton > button {
        border-radius: 8px !important;
        padding: 0.55rem 1rem !important;
        font-weight: 550 !important;
        font-size: 0.85rem !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-align: left !important;
        justify-content: flex-start !important;
    }
    /* Inactive Sidebar Buttons */
    div[data-testid="stSidebar"] button[kind="secondary"] {
        background-color: transparent !important;
        color: #94A3B8 !important;
        border: 1px solid transparent !important;
    }
    div[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        border-color: #334155 !important;
    }
    /* Active Sidebar Buttons */
    div[data-testid="stSidebar"] button[kind="primary"] {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        border: 1px solid #2563EB !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Main Content Action Buttons */
    div[data-testid="stMain"] button[kind="primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        padding: 0.65rem 1.8rem !important;
        font-size: 0.92rem !important;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.25) !important;
    }
    div[data-testid="stMain"] button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 14px rgba(37, 99, 235, 0.35) !important;
    }
    
    /* Diagnostics Code blocks override */
    code {
        color: #38BDF8 !important;
        background-color: #0F172A !important;
        border: 1px solid #334155 !important;
        padding: 2px 5px !important;
        border-radius: 4px !important;
    }

    /* Fixed Sidebar Footer */
    .sidebar-footer {
        font-size: 0.68rem;
        color: #475569;
        line-height: 1.45;
        border-top: 1px solid #334155;
        padding-top: 12px;
        margin-top: 2rem;
        padding-left: 0.5rem;
    }
    
    /* Animation definition */
    @keyframes slideUpFade {
        from {
            opacity: 0;
            transform: translateY(12px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Run CSS Injection
inject_custom_css()

# ---------------------------------------------------------
# Sidebar Component
# ---------------------------------------------------------
# Header & Name
st.sidebar.markdown("""
<div style="padding: 1rem 0 0.5rem 0.5rem;">
    <h1 style="color: #FFFFFF; font-size: 1.85rem; margin-bottom: 0.1rem; font-weight: 850; letter-spacing: -0.02em; display: flex; align-items: center; gap: 8px;">
        <span style="color: #2563EB;">🛡️</span> SANNIF
    </h1>
    <p style="color: #64748B; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin: 0;">Semantic AI Network Framework</p>
</div>
""", unsafe_allow_html=True)

# Health Status Checks
model_loaded_dot = "dot-green" if loader.model_loaded else "dot-red"
encoder_loaded_dot = "dot-green" if loader.encoder_loaded else "dot-red"
engine_ready_dot = "dot-green" if loader.prediction_engine_ready else "dot-red"
features_verified_dot = "dot-green" if loader.features_verified else "dot-red"

st.sidebar.markdown(f"""
<div class="health-container">
    <div class="health-title">System Diagnostics</div>
    <div class="health-item">
        <span class="dot {model_loaded_dot}"></span>
        <span>Model Loaded</span>
    </div>
    <div class="health-item">
        <span class="dot {encoder_loaded_dot}"></span>
        <span>Label Encoder Loaded</span>
    </div>
    <div class="health-item">
        <span class="dot {engine_ready_dot}"></span>
        <span>Prediction Engine Ready</span>
    </div>
    <div class="health-item">
        <span class="dot {features_verified_dot}"></span>
        <span>Required Features Verified</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Page Navigation List
nav_pages = {
    "Dashboard": "🏠 Dashboard",
    "Upload Dataset": "📂 Upload Dataset",
    "Detection Results": "🔍 Detection Results",
    "Analytics": "📊 Analytics",
    "Export Report": "⬇ Export Report",
    "Diagnostics": "🩺 Diagnostics",
    "About": "ℹ About"
}

st.sidebar.markdown('<p style="color: #475569; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem; margin-top: 0.5rem; padding-left: 0.5rem;">Navigation</p>', unsafe_allow_html=True)

for page_key, page_label in nav_pages.items():
    is_active = st.session_state.page == page_key
    btn_type = "primary" if is_active else "secondary"
    if st.sidebar.button(page_label, key=f"nav_{page_key}", use_container_width=True, type=btn_type):
        st.session_state.page = page_key
        st.rerun()

# Sidebar Sticky Footer
st.sidebar.markdown("""
<div class="sidebar-footer">
    <strong>SANNIF v1.0</strong><br>
    Powered by XGBoost Baseline<br>
    UNSW-NB15 Network Dataset<br>
    Semantic AI Network Intrusion Framework
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# UI Core Helper Functions
# ---------------------------------------------------------
def metric_card(title, value, subtext="", card_type="neutral"):
    """
    Renders an Apple-inspired visual KPI card.
    """
    border_color = "#334155"
    val_color = "#FFFFFF"
    
    if card_type == "green":
        border_color = "rgba(16, 185, 129, 0.3)"
        val_color = "#10B981"
    elif card_type == "red":
        border_color = "rgba(239, 68, 68, 0.3)"
        val_color = "#EF4444"
    elif card_type == "orange":
        border_color = "rgba(245, 158, 11, 0.3)"
        val_color = "#F59E0B"
        
    st.markdown(f"""
    <div class="premium-card" style="border-color: {border_color};">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value" style="color: {val_color};">{value}</div>
        <div class="kpi-subtext">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)

def get_status_from_attack(attack_label):
    """
    Derives threat level classification for custom table coloring.
    """
    attack_lower = str(attack_label).lower().strip()
    if attack_lower == "normal":
        return "Normal"
    elif attack_lower in ["analysis", "backdoor", "suspicious", "reconnaissance", "shellcode"]:
        return "Suspicious"
    else:
        return "Attack"

# Default Plotly Styling Config
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system, sans-serif", color="#94A3B8", size=11),
    margin=dict(l=0, r=0, t=30, b=10),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.05)",
        tickfont=dict(size=10, color="#94A3B8"),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.05)",
        tickfont=dict(size=10, color="#94A3B8"),
    ),
)

# ---------------------------------------------------------
# Page 1: Dashboard
# ---------------------------------------------------------
if st.session_state.page == "Dashboard":
    # Hero Callout Card
    st.markdown("""
    <div class="hero-card">
        <div class="hero-title">🛡️ SANNIF</div>
        <div class="hero-subtitle">Semantic AI Network Intrusion Framework</div>
        <div class="hero-description">
            AI-Powered Network Intrusion Detection & Threat Intelligence Dashboard. 
            Analyze network traffic streams in real-time using our production-trained XGBoost model to instantly flag cyber anomalies, reconnaissance sweeps, and malicious activity.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load Active KPIs (if predictions have run)
    kpi = st.session_state.get("kpi_data")
    if kpi is None:
        # Default placeholder metrics
        kpi = {
            "total_records": 0,
            "detected_attacks": 0,
            "normal_traffic": 0,
            "avg_confidence": 0.0,
            "highest_risk_attack": "No active dataset",
            "most_frequent_attack": "N/A",
            "detection_time": 0.0
        }

    # First Section: Detection Performance
    st.markdown('<p style="font-size: 1.25rem; font-weight: 700; margin-bottom: 1rem; color: #FFFFFF;">📊 Detection Metrics</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Total Records Analyzed", f"{kpi['total_records']:,}", subtext="Rows currently in detection memory")
        metric_card("Normal Traffic Packets", f"{kpi['normal_traffic']:,}", 
                    subtext=f"{(kpi['normal_traffic']/kpi['total_records']*100) if kpi['total_records'] > 0 else 0:.2f}% of active traffic",
                    card_type="green" if kpi["total_records"] > 0 else "neutral")
    with c2:
        metric_card("Detected Threats/Attacks", f"{kpi['detected_attacks']:,}", 
                    subtext=f"{(kpi['detected_attacks']/kpi['total_records']*100) if kpi['total_records'] > 0 else 0:.2f}% threat ratio",
                    card_type="red" if kpi["detected_attacks"] > 0 else "neutral")
        metric_card("Average Confidence", f"{kpi['avg_confidence']:.2f}%", subtext="Mean prediction confidence score")
    with c3:
        metric_card("Highest Risk Attack", str(kpi['highest_risk_attack']), subtext="Highest confidence threat category")
        metric_card("Detection Processing Time", f"{kpi['detection_time']*1000:.0f} ms" if kpi['detection_time'] > 0 else "0 ms", subtext="Backend inference execution latency")

    # Second Section: Model Profiling
    st.markdown('<p style="font-size: 1.25rem; font-weight: 700; margin-top: 1.5rem; margin-bottom: 1rem; color: #FFFFFF;">🛡️ Model Profile</p>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card("Model Accuracy", "82.01%", subtext="XGBoost Baseline classifier accuracy")
        model_status_val = "Active & Ready" if loader.prediction_engine_ready else "Offline / Loading Error"
        model_status_type = "green" if loader.prediction_engine_ready else "red"
        metric_card("Model Status", model_status_val, subtext="Primary detection engine state", card_type=model_status_type)
    with m2:
        metric_card("Dataset Model", "UNSW-NB15", subtext="Trained on standard network security benchmark")
        metric_card("Required Features Count", f"{len(loader.required_features) if loader.required_features else 34} Columns", subtext="Features validated during CSV parsing")
    with m3:
        metric_card("Total Output Classes", "10 Categories", subtext="9 attack categories + 1 normal category")

# ---------------------------------------------------------
# Page 2: Upload Dataset
# ---------------------------------------------------------
elif st.session_state.page == "Upload Dataset":
    st.markdown('<h2 style="font-weight: 800; letter-spacing: -0.03em;">📂 Upload Traffic Dataset</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 1.5rem;">Upload a CSV file containing network packet features to run threat detection.</p>', unsafe_allow_html=True)
    
    # Check backend integrity
    if not loader.prediction_engine_ready:
        st.error(f"⚠️ Prediction Engine is offline! Model files failed to load: {loader.error_message}. Upload is disabled.")
    else:
        # Drag and drop uploader
        uploaded_file = st.file_uploader("Upload Network Traffic (CSV)", type=["csv"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_df = df
                st.session_state.predictions = None
                st.session_state.detection_run = False
                st.session_state.kpi_data = None
                
                # Perform Schema Validation
                uploaded_cols = set(df.columns)
                required_cols = set(loader.required_features)
                
                missing_cols = list(required_cols - uploaded_cols)
                unexpected_cols = list(uploaded_cols - required_cols)
                is_valid = len(missing_cols) == 0
                
                st.session_state.df_validation = {
                    "is_valid": is_valid,
                    "missing_cols": missing_cols,
                    "unexpected_cols": unexpected_cols,
                    "required_cols": loader.required_features
                }
                
                # Show upload success and trigger balloons
                st.success("🎉 CSV file uploaded and parsed successfully!")
                st.toast("Success: File uploaded and validated.")
                st.balloons()
                
            except Exception as e:
                st.error(f"Failed to read CSV file: {str(e)}")
                st.session_state.uploaded_df = None
                st.session_state.df_validation = None
                
        # If dataset is loaded, display validation details & preview
        if st.session_state.uploaded_df is not None and st.session_state.df_validation is not None:
            val = st.session_state.df_validation
            
            st.markdown('<p style="font-size: 1.2rem; font-weight: 700; margin-top: 1.5rem; margin-bottom: 1rem;">🔍 Schema Validation Summary</p>', unsafe_allow_html=True)
            
            val_col1, val_col2, val_col3 = st.columns(3)
            with val_col1:
                st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                st.markdown('<p style="font-size: 0.95rem; font-weight: 700; color: #EF4444; margin-bottom: 0.5rem;">🛑 Missing Features</p>', unsafe_allow_html=True)
                if val["missing_cols"]:
                    st.markdown(f"**{len(val['missing_cols'])} missing columns:**")
                    for col in val["missing_cols"][:8]:
                        st.markdown(f"- `{col}`")
                    if len(val["missing_cols"]) > 8:
                        st.markdown(f"*and {len(val['missing_cols']) - 8} more...*")
                else:
                    st.markdown("""
                    <div style="display: flex; align-items: center; gap: 8px; color: #10B981; font-weight: 500; font-size: 0.85rem;">
                        <span>✓</span> None! All required columns present.
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with val_col2:
                st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                st.markdown('<p style="font-size: 0.95rem; font-weight: 700; color: #F59E0B; margin-bottom: 0.5rem;">⚠️ Unexpected Columns</p>', unsafe_allow_html=True)
                if val["unexpected_cols"]:
                    st.markdown(f"**{len(val['unexpected_cols'])} extra columns (will be ignored):**")
                    for col in val["unexpected_cols"][:8]:
                        st.markdown(f"- `{col}`")
                    if len(val["unexpected_cols"]) > 8:
                        st.markdown(f"*and {len(val['unexpected_cols']) - 8} more...*")
                else:
                    st.markdown("""
                    <div style="display: flex; align-items: center; gap: 8px; color: #10B981; font-weight: 500; font-size: 0.85rem;">
                        <span>✓</span> None! Perfect schema match.
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with val_col3:
                st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                st.markdown('<p style="font-size: 0.95rem; font-weight: 700; color: #38BDF8; margin-bottom: 0.5rem;">📋 Required Features</p>', unsafe_allow_html=True)
                st.markdown(f"**The model requires {len(val['required_cols'])} features:**")
                for col in val["required_cols"][:8]:
                    st.markdown(f"- `{col}`")
                if len(val["required_cols"]) > 8:
                    st.markdown(f"*and {len(val['required_cols']) - 8} more (view Diagnostics for full list).*")
                st.markdown('</div>', unsafe_allow_html=True)

            if not val["is_valid"]:
                st.error("🛑 Schema validation failed. Please upload a dataset containing all required model columns.")
            else:
                st.info("💡 Validation successful. Head to the **Detection Results** page to run threat detection.")
                
            # Display first 10 rows preview
            st.markdown('<p style="font-size: 1.2rem; font-weight: 700; margin-top: 1.5rem; margin-bottom: 1rem;">📄 Dataset Preview (First 10 Rows)</p>', unsafe_allow_html=True)
            preview_df = st.session_state.uploaded_df.head(10)
            
            # Format numbers to be clean
            preview_formatted = preview_df.copy()
            for col in preview_formatted.columns:
                if pd.api.types.is_float_dtype(preview_formatted[col]):
                    preview_formatted[col] = preview_formatted[col].round(4)
                    
            # Render HTML Table
            headers = "".join(f"<th>{col}</th>" for col in preview_formatted.columns)
            rows = ""
            for idx, row in preview_formatted.iterrows():
                cells = "".join(f"<td>{val}</td>" for val in row.values)
                rows += f"<tr>{cells}</tr>"
                
            st.markdown(f"""
            <div class="data-table-container">
                <table class="data-table">
                    <thead><tr>{headers}</tr></thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Page 3: Detection Results
# ---------------------------------------------------------
elif st.session_state.page == "Detection Results":
    st.markdown('<h2 style="font-weight: 800; letter-spacing: -0.03em;">🔍 Network Intrusion Detection</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 1.5rem;">Execute the XGBoost baseline threat detection engine on the active dataset.</p>', unsafe_allow_html=True)
    
    # 1. State verification
    if st.session_state.uploaded_df is None:
        st.warning("⚠️ No dataset uploaded. Please go to the **Upload Dataset** page and import a CSV file first.")
    elif not loader.prediction_engine_ready:
        st.error("🛑 The XGBoost prediction engine is offline. Model files must be resolved first.")
    elif st.session_state.df_validation is not None and not st.session_state.df_validation["is_valid"]:
        st.error("🛑 Schema mismatch detected. The uploaded dataset lacks required model columns. Execution is blocked.")
    else:
        # Button container
        c_btn, _ = st.columns([1, 4])
        with c_btn:
            run_btn = st.button("🚀 Run Detection", use_container_width=True, type="primary")
            
        if run_btn:
            # Loading Spinner
            with st.spinner("Analyzing Network Traffic..."):
                start_time = time.time()
                # Run actual XGBoost prediction
                try:
                    preds = loader.predict_attacks(st.session_state.uploaded_df)
                    elapsed = time.time() - start_time
                    
                    # Store Results
                    preds["Status"] = preds["Predicted Attack"].apply(get_status_from_attack)
                    st.session_state.predictions = preds
                    st.session_state.detection_run = True
                    
                    # Compute & Save KPIs
                    threats = preds[preds["Status"] != "Normal"]
                    num_threats = len(threats)
                    num_normal = len(preds) - num_threats
                    avg_confidence = preds["Confidence"].mean() * 100.0
                    
                    # Determine risk metrics
                    if num_threats > 0:
                        highest_risk_row = threats.loc[threats["Confidence"].idxmax()]
                        highest_risk_attack = highest_risk_row["Predicted Attack"]
                        most_frequent_attack = threats["Predicted Attack"].value_counts().index[0]
                    else:
                        highest_risk_attack = "None"
                        most_frequent_attack = "None"
                        
                    st.session_state.kpi_data = {
                        "total_records": len(preds),
                        "detected_attacks": num_threats,
                        "normal_traffic": num_normal,
                        "avg_confidence": avg_confidence,
                        "highest_risk_attack": highest_risk_attack,
                        "most_frequent_attack": most_frequent_attack,
                        "detection_time": elapsed
                    }
                    
                    # Write to disk automatically in reports/
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    report_path = os.path.join("reports", f"prediction_{timestamp}.csv")
                    # Save combined dataset (input features + predictions)
                    combined_df = pd.concat([st.session_state.uploaded_df, preds], axis=1)
                    combined_df.to_csv(report_path, index=False)
                    
                    st.success("🎉 Threat detection completed successfully! Report archived.")
                    st.toast("Success: Detection complete.", icon="🛡️")
                except Exception as e:
                    st.error(f"Inference failed: {str(e)}")
                    st.session_state.predictions = None
                    st.session_state.detection_run = False
                    st.session_state.kpi_data = None
                    
        # Render results if completed
        if st.session_state.detection_run and st.session_state.predictions is not None:
            kpi = st.session_state.kpi_data
            pred_df = st.session_state.predictions
            
            # Post-Prediction Summary Cards
            st.markdown('<p style="font-size: 1.2rem; font-weight: 700; margin-top: 1.5rem; margin-bottom: 1rem;">📊 Analysis Summary</p>', unsafe_allow_html=True)
            k_col1, k_col2, k_col3 = st.columns(3)
            with k_col1:
                metric_card("Total Rows Scanned", f"{kpi['total_records']:,}", subtext="Complete traffic sequence length")
                metric_card("Normal Traffic", f"{kpi['normal_traffic']:,}", 
                            subtext=f"{(kpi['normal_traffic']/kpi['total_records']*100):.2f}% of traffic packets", card_type="green")
            with k_col2:
                metric_card("Threats Detected", f"{kpi['detected_attacks']:,}", 
                            subtext=f"{(kpi['detected_attacks']/kpi['total_records']*100):.2f}% attack signatures flagged", card_type="red" if kpi['detected_attacks'] > 0 else "neutral")
                metric_card("Average Confidence", f"{kpi['avg_confidence']:.2f}%", subtext="Mean certainty across classification outputs")
            with k_col3:
                metric_card("Highest Risk Threat", str(kpi['highest_risk_attack']), subtext="Threat class matching maximum probability")
                metric_card("Execution Latency", f"{kpi['detection_time']*1000:.1f} ms", subtext=f"Average {(kpi['detection_time']/kpi['total_records']*1000000):.3f} μs per packet")

            # Table Preview with Pagination
            st.markdown('<p style="font-size: 1.2rem; font-weight: 700; margin-top: 1.5rem; margin-bottom: 1rem;">📄 Predicted Intrusion Logs</p>', unsafe_allow_html=True)
            
            # Define pagination setup
            rows_per_page = 20
            total_records = len(pred_df)
            total_pages = max(1, (total_records + rows_per_page - 1) // rows_per_page)
            
            pag_col1, pag_col2, pag_col3 = st.columns([1, 2, 1])
            with pag_col2:
                page_selection = st.number_input(
                    f"Showing page of {total_pages}",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    step=1,
                    key="predictions_page_picker"
                )
                
            start_row = (page_selection - 1) * rows_per_page
            end_row = min(start_row + rows_per_page, total_records)
            
            sliced_df = pred_df.iloc[start_row:end_row]
            
            # Build Table HTML
            table_rows = ""
            for index, row in sliced_df.iterrows():
                attack_type = row["Predicted Attack"]
                confidence_score = row["Confidence"] * 100.0
                threat_status = row["Status"]
                
                # Check status class & badge
                if threat_status == "Normal":
                    row_class = "status-normal"
                    badge_html = '<span class="badge badge-green">Normal</span>'
                elif threat_status == "Suspicious":
                    row_class = "status-suspicious"
                    badge_html = '<span class="badge badge-orange">Suspicious</span>'
                else:
                    row_class = "status-attack"
                    badge_html = '<span class="badge badge-red">Attack</span>'
                    
                # Format progress bar HTML
                progress_html = f"""
                <div class="progress-container">
                    <div class="progress-bg">
                        <div class="progress-fill" style="width: {confidence_score}%;"></div>
                    </div>
                    <span style="font-size: 0.8rem; font-weight: 500; min-width: 45px;">{confidence_score:.1f}%</span>
                </div>
                """
                
                table_rows += f"""
                <tr class="{row_class}">
                    <td><strong>Row {index + 1}</strong></td>
                    <td><strong>{attack_type}</strong></td>
                    <td>{progress_html}</td>
                    <td>{badge_html}</td>
                </tr>
                """
                
            st.markdown(f"""
            <div class="data-table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Packet Index</th>
                            <th>Predicted Attack Category</th>
                            <th>Classification Confidence</th>
                            <th>Threat Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Page 4: Analytics
# ---------------------------------------------------------
elif st.session_state.page == "Analytics":
    st.markdown('<h2 style="font-weight: 800; letter-spacing: -0.03em;">📊 Threat Intelligence Analytics</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 1.5rem;">Visual breakdown of classification distributions and prediction parameters.</p>', unsafe_allow_html=True)
    
    if not st.session_state.detection_run or st.session_state.predictions is None:
        st.warning("⚠️ No analytical data available. Please run threat detection on a valid CSV file first.")
    else:
        pred_df = st.session_state.predictions
        kpi = st.session_state.kpi_data
        
        # Section A: Distributions
        col_a1, col_a2 = st.columns(2)
        
        with col_a1:
            # 1. Attack Distribution Pie Chart
            pie_counts = pred_df["Predicted Attack"].value_counts().reset_index()
            pie_counts.columns = ["Category", "Count"]
            
            fig_pie = px.pie(
                pie_counts,
                names="Category",
                values="Count",
                hole=0.42,
                color="Category",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#1E293B', width=2)))
            fig_pie.update_layout(**PLOT_LAYOUT)
            
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown('<p class="kpi-title" style="margin-bottom: 0.1rem;">Attack Profile Distribution</p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 0.72rem; color: #64748B; margin-bottom: 1rem;">Share of network events assigned to specific categories</p>', unsafe_allow_html=True)
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_a2:
            # 2. Confidence Distribution Histogram (Pre-aggregated server-side for speed)
            counts, bins = np.histogram(pred_df["Confidence"], bins=20, range=(0.0, 1.0))
            bin_labels = [f"{bins[i]*100:.0f}-{bins[i+1]*100:.0f}%" for i in range(len(counts))]
            
            hist_df = pd.DataFrame({
                "Confidence Range": bin_labels,
                "Count": counts
            })
            
            fig_hist = px.bar(
                hist_df,
                x="Confidence Range",
                y="Count",
                color_discrete_sequence=["#2563EB"]
            )
            fig_hist.update_layout(**PLOT_LAYOUT)
            fig_hist.update_xaxes(title="Confidence Range")
            fig_hist.update_yaxes(gridcolor="rgba(255,255,255,0.05)", title="Event Count")
            
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown('<p class="kpi-title" style="margin-bottom: 0.1rem;">Confidence Score Distribution</p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 0.72rem; color: #64748B; margin-bottom: 1rem;">Server-side aggregated frequency of prediction confidence scores</p>', unsafe_allow_html=True)
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Section B: Frequencies
        col_b1, col_b2 = st.columns(2)
        
        with col_b1:
            # 3. Overall Frequencies
            fig_bar = px.bar(
                pie_counts,
                x="Category",
                y="Count",
                color="Category",
                color_discrete_sequence=px.colors.qualitative.Safe,
                text_auto=True
            )
            fig_bar.update_layout(**PLOT_LAYOUT)
            fig_bar.update_xaxes(title="Traffic Classification")
            fig_bar.update_yaxes(title="Frequency")
            
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown('<p class="kpi-title" style="margin-bottom: 0.1rem;">Intrusion Category Frequency</p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 0.72rem; color: #64748B; margin-bottom: 1rem;">Absolute counts for all detected categories</p>', unsafe_allow_html=True)
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_b2:
            # 4. Top Attack Categories (excluding Normal)
            threats_only = pred_df[pred_df["Predicted Attack"] != "Normal"]
            if not threats_only.empty:
                threat_counts = threats_only["Predicted Attack"].value_counts().reset_index()
                threat_counts.columns = ["Attack Category", "Count"]
                
                fig_threats = px.bar(
                    threat_counts,
                    y="Attack Category",
                    x="Count",
                    orientation="h",
                    color="Attack Category",
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    text_auto=True
                )
                fig_threats.update_layout(**PLOT_LAYOUT)
                fig_threats.update_xaxes(title="Flagged Counts")
                fig_threats.update_yaxes(title="Malicious Category")
                
                st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                st.markdown('<p class="kpi-title" style="margin-bottom: 0.1rem;">Top Flagged Intrusion Vectors</p>', unsafe_allow_html=True)
                st.markdown('<p style="font-size: 0.72rem; color: #64748B; margin-bottom: 1rem;">Detailed view of malicious threats (excluding normal packets)</p>', unsafe_allow_html=True)
                st.plotly_chart(fig_threats, use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="premium-card" style="height: 100%; display: flex; align-items: center; justify-content: center; min-height: 380px;">', unsafe_allow_html=True)
                st.markdown("""
                <div style="text-align: center; color: #10B981;">
                    <h3 style="font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">🟢 Zero Threats Detected</h3>
                    <p style="font-size: 0.85rem; color: #94A3B8; max-width: 300px; margin: 0 auto;">No malicious or suspicious signatures were detected in this traffic stream.</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Page 5: Export Report
# ---------------------------------------------------------
elif st.session_state.page == "Export Report":
    st.markdown('<h2 style="font-weight: 800; letter-spacing: -0.03em;">⬇ Export Prediction Reports</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 1.5rem;">Download local database copies or archive predictions compiled on the platform.</p>', unsafe_allow_html=True)
    
    if not st.session_state.detection_run or st.session_state.predictions is None:
        st.warning("⚠️ No threat detection report available. Please run detection on a valid CSV file first.")
    else:
        pred_df = st.session_state.predictions
        combined_df = pd.concat([st.session_state.uploaded_df, pred_df], axis=1)
        
        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown('### 📄 Download Data Report', unsafe_allow_html=True)
            st.write("Download the comprehensive CSV report. The export includes all original uploaded columns matched to predicted categories and confidence scores.")
            
            # Export CSV button
            csv_data = combined_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV Report",
                data=csv_data,
                file_name=f"sannif_threat_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c_exp2:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown('### 📕 Enterprise Document Export', unsafe_allow_html=True)
            st.write("Generate a formal executive threat report in PDF format containing summary tables, risk scoring, diagnostics, and graphical charts.")
            
            # PDF Placeholder button
            st.button("Download PDF Report (Coming Soon)", disabled=True, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        # File System Archives
        st.markdown('<p style="font-size: 1.25rem; font-weight: 700; margin-top: 1.5rem; margin-bottom: 1rem;">🗄️ Local Server Archives</p>', unsafe_allow_html=True)
        st.write("The platform automatically archives all run predictions into the `reports/` directory on the server. Below are the historical logs:")
        
        # List files in reports/
        reports_dir = "reports"
        if os.path.exists(reports_dir):
            files = [f for f in os.listdir(reports_dir) if f.startswith("prediction_") and f.endswith(".csv")]
            files.sort(reverse=True) # newest first
            
            if files:
                archived_rows = ""
                for index, file in enumerate(files):
                    file_path = os.path.join(reports_dir, file)
                    file_size = os.path.getsize(file_path) / 1024 # KB
                    created_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
                    
                    archived_rows += f"""
                    <tr>
                        <td><strong>Archive #{len(files) - index}</strong></td>
                        <td><code>{file}</code></td>
                        <td>{created_time}</td>
                        <td>{file_size:.2f} KB</td>
                    </tr>
                    """
                    
                st.markdown(f"""
                <div class="data-table-container">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Index</th>
                                <th>Filename</th>
                                <th>Creation Timestamp</th>
                                <th>File Size</th>
                            </tr>
                        </thead>
                        <tbody>
                            {archived_rows}
                        </tbody>
                    </table>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No server reports found in storage.")
        else:
            st.info("Reports folder empty or initializing.")

# ---------------------------------------------------------
# Page 6: Diagnostics
# ---------------------------------------------------------
elif st.session_state.page == "Diagnostics":
    st.markdown('<h2 style="font-weight: 800; letter-spacing: -0.03em;">🩺 Platform Diagnostics</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 1.5rem;">Environmental variables, package versions, resource metrics, and model feature configuration.</p>', unsafe_allow_html=True)
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('### ⚙️ System Environment', unsafe_allow_html=True)
        
        # Get python versions
        py_version = sys.version.split()[0]
        st.text(f"Python Version: {py_version}")
        st.text(f"Streamlit Version: {st.__version__}")
        
        # Get xgboost version
        try:
            import xgboost as xgb
            xgb_ver = xgb.__version__
        except ImportError:
            xgb_ver = "Not Installed"
        st.text(f"XGBoost Version: {xgb_ver}")
        
        # Memory Info
        try:
            process = psutil.Process()
            mem_use = process.memory_info().rss / (1024 * 1024)
            st.text(f"Process Memory Usage: {mem_use:.2f} MB")
        except Exception:
            st.text("Process Memory Usage: Unavailable")
            
        # Prediction time
        pred_time = st.session_state.get("kpi_data")
        if pred_time is not None:
            st.text(f"Last Inference Time: {pred_time['detection_time'] * 1000:.2f} ms")
        else:
            st.text("Last Inference Time: No run recorded")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Backend States
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('### 🛡️ Backend Deserialization Status', unsafe_allow_html=True)
        
        status_yes = '<span class="badge badge-green">VERIFIED</span>'
        status_no = '<span class="badge badge-red">FAILED</span>'
        
        st.markdown(f"""
        - Model Deserialized: {status_yes if loader.model_loaded else status_no}
        - Label Encoder Deserialized: {status_yes if loader.encoder_loaded else status_no}
        - Schema Checker Active: {status_yes if loader.features_verified else status_no}
        - Prediction Engine Ready: {status_yes if loader.prediction_engine_ready else status_no}
        - Dataset Uploaded: {status_yes if st.session_state.uploaded_df is not None else '<span class="badge badge-orange">WAITING</span>'}
        """, unsafe_allow_html=True)
        
        # Show specific loader error if any
        if loader.error_message:
            st.error(f"Engine Log: {loader.error_message}")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_d2:
        st.markdown('<div class="premium-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown('### 📋 Model Required Features', unsafe_allow_html=True)
        st.write("The production model expects the following features in order. CSV inputs will be subsetted and re-ordered to match this schema:")
        
        if loader.required_features:
            feat_df = pd.DataFrame({"Feature Name": loader.required_features})
            
            # Simple custom feature table preview
            f_rows = ""
            for idx, feature in enumerate(loader.required_features):
                f_rows += f"""
                <tr>
                    <td><code>{idx + 1}</code></td>
                    <td><strong>{feature}</strong></td>
                </tr>
                """
            st.markdown(f"""
            <div class="data-table-container" style="max-height: 380px; overflow-y: auto;">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Index</th>
                            <th>Feature Name</th>
                        </tr>
                    </thead>
                    <tbody>
                        {f_rows}
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.write("⚠️ Model required features are empty. Model is not loaded or does not support automatic feature extraction.")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Page 7: About
# ---------------------------------------------------------
elif st.session_state.page == "About":
    st.markdown('<h2 style="font-weight: 800; letter-spacing: -0.03em;">ℹ About SANNIF</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 1.5rem;">Project details, architecture description, and model validation details.</p>', unsafe_allow_html=True)
    
    col_ab1, col_ab2 = st.columns(2)
    with col_ab1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('### 🛡️ Semantic AI Network Intrusion Framework', unsafe_allow_html=True)
        st.write("""
        **SANNIF** is an enterprise-grade cyber-threat analytics framework designed to analyze packet streams and detect malicious traffic signatures.
        
        By incorporating advanced machine learning classifiers, SANNIF bridges the gap between deep model reasoning and security operations center (SOC) visualization. The platform is designed with a premium, clutter-free Apple-inspired interface to prioritize rapid decision-making under high-pressure scenarios.
        
        **Core Objectives:**
        - **Semantic Scannability:** Instantly distinguish between benign, suspicious, and severe intrusions using clear visual cues.
        - **Model Resiliency:** Automatic feature schema validation ensures the frontend never crashes due to format deviations.
        - **High-Velocity Analytics:** Deep processing statistics and sub-millisecond execution analysis.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_ab2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('### 🧱 Model & Dataset Specifications', unsafe_allow_html=True)
        
        st.markdown("""
        **Detection Engine Properties:**
        - **Underlying Classifier:** XGBoost Baseline
        - **Dataset Reference:** UNSW-NB15
        - **Model Accuracy Score:** 82.01%
        - **Total Training Features:** 34
        - **Target Intrusion Categories:** 10 (9 Threat Classes + 1 Normal Class)
        
        **Target Classification Categories:**
        1. **Normal:** Standard secure network transactions.
        2. **Analysis:** Contains spam, port sweeps, or HTML file transfers.
        3. **Backdoor:** Rogue entry points designed to bypass firewalls.
        4. **DoS (Denial of Service):** Exhausting memory/connection pools.
        5. **Exploits:** Exploding software bugs to run unauthorized scripts.
        6. **Fuzzers:** Automated inputs designed to crash code pathways.
        7. **Generic:** Non-specific anomalous packet structures.
        8. **Reconnaissance:** Port scanners, OS sweeps, and system mapping.
        9. **Shellcode:** Small injection scripts designed to launch terminal shells.
        10. **Worms:** Replicating software agents designed to spread.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
