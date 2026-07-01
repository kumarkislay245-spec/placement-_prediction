import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import time

# ==================================================================
# PAGE CONFIGURATION
# ==================================================================
st.set_page_config(
    page_title="Campus Placement Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================================================================
# ENTERPRISE DARK MODE DESIGN SYSTEM
# ==================================================================
st.markdown("""
    <style>
    :root {
        --brand-primary: #6366F1;
        --brand-hover: #4F46E5;
        --bg-main: #0B0F19;
        --card-bg: #151B2C;
        --border-color: #242F49;
        --text-main: #F9FAFB;
        --text-muted: #9CA3AF;

        --success: #10B981;
        --success-bg: rgba(16, 185, 129, 0.1);
        --warning: #F59E0B;
        --warning-bg: rgba(245, 158, 11, 0.1);
    }

    /* Core Layout Architecture */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-main) !important;
    }

    div.block-container {
        padding: 2rem 3rem !important;
        max-width: 1400px;
    }

    /* Clean Header styling */
    .header-container {
        margin-bottom: 2.5rem;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 1.5rem;
    }
    .header-container h1 {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--text-main) !important;
        letter-spacing: -0.025em;
        margin: 0 0 0.5rem 0;
    }
    .header-container p {
        font-size: 1rem;
        color: var(--text-muted);
        margin: 0;
    }

    /* Modern Minimal Section Layouts */
    .section-divider {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-main);
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Label Styling Controls */
    label p {
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        color: var(--text-muted) !important;
    }

    /* Action Nodes */
    div.stButton > button {
        background: var(--brand-primary) !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 12px 0px !important;
        width: 100% !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2);
        transition: all 0.2s ease;
        margin-top: 1rem;
    }
    div.stButton > button:hover {
        background: var(--brand-hover) !important;
        transform: translateY(-1px);
        box-shadow: 0 8px 16px rgba(99, 102, 241, 0.35);
    }

    /* Scoring Card Feature */
    .score-hero-box {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .score-value {
        font-size: 3.5rem;
        font-weight: 800;
        color: var(--brand-primary);
        line-height: 1;
        margin: 0.5rem 0;
    }
    .score-status {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        margin: 0 auto;
    }
    .status-high { background: var(--success-bg); color: var(--success); border: 1px solid rgba(16, 185, 129, 0.2); }
    .status-med { background: var(--warning-bg); color: var(--warning); border: 1px solid rgba(245, 158, 11, 0.2); }

    /* Widescreen Professional Data Table CSS */
    .enterprise-table-container {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        overflow: hidden;
        margin-top: 1.5rem;
    }
    table.enterprise-table {
        width: 100%;
        border-collapse: collapse;
        text-align: left;
        font-size: 0.92rem;
    }
    table.enterprise-table th {
        background: #1C2336;
        color: var(--text-main);
        font-weight: 700;
        padding: 14px 20px;
        border-bottom: 1px solid var(--border-color);
        text-transform: uppercase;
        font-size: 0.78rem;
        letter-spacing: 0.05em;
    }
    table.enterprise-table td {
        padding: 16px 20px;
        border-bottom: 1px solid var(--border-color);
        color: #E5E7EB;
        vertical-align: middle;
    }
    table.enterprise-table tr:hover {
        background: #1A2235;
    }

    /* Table Pill badges */
    .table-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
    }
    .badge-eligible { background: rgba(16, 185, 129, 0.15); color: #34D399; }
    .badge-locked { background: rgba(239, 68, 68, 0.15); color: #F87171; }

    /* Progress Bars inside Table rows */
    .table-progress-outer {
        background: #242F49;
        height: 6px;
        border-radius: 10px;
        width: 120px;
        display: inline-block;
        overflow: hidden;
        vertical-align: middle;
        margin-right: 8px;
    }
    .table-progress-inner {
        background: var(--warning);
        height: 100%;
    }
    .gap-indicator {
        font-size: 0.75rem;
        color: var(--warning);
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Layout Header Area
st.markdown("""
<div class="header-container">
    <h1>🎓 Campus Placement Intelligence Console</h1>
    <p>Enterprise diagnostics tool mapping student predictive analytics against corporate recruitment benchmarks</p>
</div>
""", unsafe_allow_html=True)

# ==================================================================
# PIPELINE CONFIG & CORE BACKEND
# ==================================================================
FEATURE_ORDER = ['cgpa', 'skills', 'communication_skill_rating', 'total_projects', 'total_experience', 'education']
MODEL_PATH = 'placement_logistic_model.pkl'
SCALER_PATH = 'placement_scaler.pkl'
CUTOFFS_PATH = 'company_cutoffs.json'


@st.cache_resource
def load_ml_assets():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    return None, None


@st.cache_data
def load_company_data():
    if os.path.exists(CUTOFFS_PATH):
        with open(CUTOFFS_PATH, 'r') as f:
            return json.load(f)
    return {}


model, scaler = load_ml_assets()
company_data = load_company_data()

# ==================================================================
# LAYOUT STRUCTURE: LEFT SIDE PARAMETERS, RIGHT SIDE METRICS/TABLES
# ==================================================================
left_panel, right_panel = st.columns([1, 2], gap="large")

with left_panel:
    st.markdown('<div class="section-divider">📝 Academic Architecture</div>', unsafe_allow_html=True)
    branch = st.selectbox("Target Discipline / Branch",
                          options=sorted(company_data.keys()) if company_data else ["CSE", "BIOTECH", "CHEM"])

    cgpa = st.number_input("Current Cumulative CGPA", min_value=0.0, max_value=10.0, value=8.03, step=0.01)
    tenth = st.number_input("Secondary School (10th %)", min_value=0.0, max_value=100.0, value=85.0, step=0.1)
    twelfth = st.number_input("Higher Secondary (12th %)", min_value=0.0, max_value=100.0, value=85.0, step=0.1)

    st.markdown('<div class="section-divider">💡 Portfolio & Profile Factors</div>', unsafe_allow_html=True)
    major_p = st.number_input("Major Development Projects", min_value=0, max_value=10, value=2)
    mini_p = st.number_input("Mini Repos / Open-Source Work", min_value=0, max_value=10, value=1)
    skills_score = st.slider("Technical Skill Tier", 0.0, 10.0, 8.10, 0.1)
    comm_score = st.slider("Communication Core Assessment", 1.0, 5.0, 4.00, 0.1)

    st.markdown('<div class="section-divider">🚀 Professional Infrastructure</div>', unsafe_allow_html=True)
    intern_exp = st.selectbox("Vetted Internship Completed", options=["No", "Yes"])
    hackathon_exp = st.selectbox("Hackathon Submissions Tracked", options=["No", "Yes"])
    workshops_cnt = st.number_input("Industry Workshops Attended", min_value=0, max_value=10, value=1)

    calculate_analytics = st.button("Compute Profile Metrics")

# ==================================================================
# ENGINE PROCESSING NODE & WIDESCREEN REPORTS
# ==================================================================
with right_panel:
    total_projects = major_p + mini_p
    total_exp_vectors = workshops_cnt + (1 if intern_exp == "Yes" else 0) + (1 if hackathon_exp == "Yes" else 0)
    aggregate_edu = (tenth + twelfth) / 2.0

    if model is None or scaler is None:
        raw_prob = min(0.98, max(0.12, (cgpa / 10.0) * 0.5 + (skills_score / 10.0) * 0.35 + (total_exp_vectors * 0.05)))
    else:
        raw_data = {
            'cgpa': float(cgpa), 'skills': float(skills_score), 'communication_skill_rating': float(comm_score),
            'total_projects': int(total_projects), 'total_experience': int(total_exp_vectors),
            'education': float(aggregate_edu)
        }
        input_matrix = pd.DataFrame([raw_data]).reindex(columns=FEATURE_ORDER)
        scaled_matrix = scaler.transform(input_matrix)
        raw_prob = model.predict_proba(scaled_matrix)[0][1]

    badge_style = "status-high" if raw_prob >= 0.70 else "status-med"
    badge_label = "Highly Viable Candidate" if raw_prob >= 0.70 else "Competitive Market Profile"

    # Profile Header Summary Widget
    st.markdown(f"""
    <div class="score-hero-box">
        <div style="font-size: 0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; color: var(--text-muted);">Calculated Matrix Selection Confidence</div>
        <div class="score-value">{raw_prob * 100:.1f}%</div>
        <div class="score-status {badge_style}">{badge_label}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider">🏢 Global Matrix Alignment Ledger</div>', unsafe_allow_html=True)

    companies_list = company_data.get(branch, [])
    if not companies_list:
        companies_list = [
            {"company": "Deloitte India", "min_cgpa": 7.0, "avg_ctc": 8.5, "offers": 14},
            {"company": "Fractal Analytics", "min_cgpa": 7.5, "avg_ctc": 12.0, "offers": 8},
            {"company": "ZS Associates", "min_cgpa": 8.2, "avg_ctc": 13.5, "offers": 6},
            {"company": "McKenzie & Co.", "min_cgpa": 8.5, "avg_ctc": 18.0, "offers": 3}
        ]

    # Generate Professional Table Rows Programmatically
    table_rows_html = ""
    for c in sorted(companies_list, key=lambda x: x['min_cgpa']):
        is_eligible = cgpa >= c['min_cgpa']

        if is_eligible:
            status_tag = '<span class="table-badge badge-eligible">Eligible</span>'
            requirement_block = '<span>All Clear</span>'
        else:
            status_tag = '<span class="table-badge badge-locked">Locked</span>'
            gap = c['min_cgpa'] - cgpa
            progress_pct = max(10, min(95, 100 - (gap * 40)))
            requirement_block = f"""
            <div class="table-progress-outer"><div class="table-progress-inner" style="width:{progress_pct}%;"></div></div>
            <span class="gap-indicator">+{gap:.2f} CGPA</span>
            """

        table_rows_html += f"""
        <tr>
            <td style="font-weight:700; color:var(--text-main);">{c['company']}</td>
            <td>{c['min_cgpa']:.2f}</td>
            <td style="color:#34D399; font-weight:600;">{c['avg_ctc']:.1f} LPA</td>
            <td>{c['offers']} Hires</td>
            <td>{status_tag}</td>
            <td>{requirement_block}</td>
        </tr>
        """

    # CRITICAL FIX for image_fe5533.png: Render whole string directly using st.html()
    st.html(f"""
    <div class="enterprise-table-container">
        <table class="enterprise-table">
            <thead>
                <tr>
                    <th>Target Company</th>
                    <th>Minimum Cutoff</th>
                    <th>Average Compensation</th>
                    <th>Historical Volume</th>
                    <th>Access Status</th>
                    <th>Development Milestone</th>
                </tr>
            </thead>
            <tbody>
                {table_rows_html}
            </tbody>
        </table>
    </div>
    """)

st.markdown(
    '<p class="footer-note">Analytics Ledger Platform • In-Memory Processing Verification Layer • High Performance Layout Framework.</p>',
    unsafe_allow_html=True)
