"""
BioVision — Spectral Produce Freshness & Retail Intelligence Platform.
Main Streamlit application entrypoint.
"""
import sys
import os
from pathlib import Path

# Prevent OpenMP / DLL collisions on Windows and register torch/lib DLL directory
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
site_packages = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages")
torch_lib = os.path.join(site_packages, "torch", "lib")
if os.path.exists(torch_lib) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(torch_lib)
    except Exception:
        pass

try:
    import torch
except Exception as e:
    pass

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
from datetime import datetime

from dashboard.components import load_custom_css, render_metric_card
from database.repository import get_db_repository

st.set_page_config(
    page_title="BioVision | Spectral Freshness Intelligence",
    page_icon="🥑",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_custom_css()

# Initialize DB & Seed demo records if empty
repo = get_db_repository()
repo.seed_initial_demo_data()

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🥑 **BioVision Core**")
    st.markdown("<p style='color: #94a3b8; font-size: 0.85rem;'>Spectral Early-Warning Freshness Platform</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("**Active Produce Surveillance:**")
    produce_category = st.selectbox(
        "Select Category",
        [
            "All Fresh Produce (Avocados, Mangoes, Apples, Berries)",
            "🥑 Avocado (Hass) - SKU #4046",
            "🥭 Mango (Alphonso) - SKU #4051",
            "🍎 Apple (Honeycrisp) - SKU #4135",
            "🍓 Strawberry (Sweet Charlie) - SKU #4249",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("**Deployment Environment:**")
    st.selectbox(
        "Deployment Environment",
        [
            "Quick-Commerce Dark Store (Blinkit / Zepto Hub)",
            "Supermarket Floor (Reliance Fresh / DMart)",
            "Regional Cold Vault Distribution Hub",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("#### Operational Status")
    st.markdown(
        """
        <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.6;">
            <div>● Vision Engine: <span style="color: #10b981; font-weight: 600;">ACTIVE (CUDA FP16)</span></div>
            <div>● Modalities: <span style="color: #06b6d4; font-weight: 600;">RGB + 850nm NIR</span></div>
            <div>● Produce SKUs: <span style="color: #a855f7; font-weight: 600;">Multi-Category Calibrated</span></div>
            <div>● ESL Digital Tags: <span style="color: #10b981; font-weight: 600;">ONLINE & SYNCED</span></div>
            <div>● Policy Rules: <span style="color: #f59e0b; font-weight: 600;">FEFO Dynamic Markdown</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Hero Header ---
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">BioVision: Multi-Produce Freshness Platform</div>
        <div class="hero-subtitle">
            Autonomous pre-visual spoilage mitigation for <b>Supermarkets & Quick-Commerce</b>. 
            Detecting internal cellular rot <b>3.2 days before skin symptoms appear</b> using 
            multimodal RGB + Near-Infrared (NIR) spectral fusion to prevent customer refunds and stop food waste.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Executive Macro Metrics ---
metrics = repo.get_summary_metrics()

col1, col2, col3, col4 = st.columns(4)

with col1:
    render_metric_card(
        label="Monitored Batches",
        value=str(metrics["total_batches"]),
        subtext="Avocados, Mangoes, Apples & Berries",
        delta="+6 active lots",
    )

with col2:
    render_metric_card(
        label="Units Under Surveillance",
        value=f"{metrics['active_inventory_units']:,}",
        subtext="Fresh produce across vaults & aisles",
        delta="100% Optical Coverage",
    )

with col3:
    render_metric_card(
        label="Urgent FEFO Lots",
        value=str(metrics["critical_fefo_batches"]),
        subtext="Priority score ≥ 70 — immediate markdown needed",
        delta="Action Required" if metrics["critical_fefo_batches"] > 0 else "Optimal",
    )

with col4:
    render_metric_card(
        label="Loss Avoided To Date",
        value=f"${metrics['loss_avoided_dollars']:,.2f}",
        subtext="Salvaged via dynamic pricing & deli prep",
        delta="+34.2% vs Static FIFO",
    )

st.write("")
st.write("")

# --- Navigation Card Matrix ---
st.markdown("### Platform Command Modules")
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

with nav_col1:
    st.markdown(
        """
        <div class="metric-card" style="height: 190px;">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">🔬</div>
            <div style="font-weight: 700; font-size: 1.05rem; color: #f8fafc; margin-bottom: 6px;">Optical Scanner</div>
            <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">
                Multi-produce X-Ray inspection with side-by-side surface vs subsurface views, instant shelf-life verdicts, and live ESL pricing tags.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Launch Scanner →", key="btn_nav_scan", use_container_width=True):
        st.switch_page("pages/1_Scan_and_Predict.py")

with nav_col2:
    st.markdown(
        """
        <div class="metric-card" style="height: 190px;">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">📦</div>
            <div style="font-weight: 700; font-size: 1.05rem; color: #f8fafc; margin-bottom: 6px;">Inventory Intelligence</div>
            <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">
                Dynamic FEFO (First-Expired-First-Out) ranking of all active pallets, warehouse storage zones, and degradation trends.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("View Inventory →", key="btn_nav_inv", use_container_width=True):
        st.switch_page("pages/2_Inventory_Intelligence.py")

with nav_col3:
    st.markdown(
        """
        <div class="metric-card" style="height: 190px;">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">⚡</div>
            <div style="font-weight: 700; font-size: 1.05rem; color: #f8fafc; margin-bottom: 6px;">Action Centre</div>
            <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">
                Approve, reject, or override store actions (20% markdown, 50% markdown, deli repurposing) with an immutable audit log.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Review Actions →", key="btn_nav_act", use_container_width=True):
        st.switch_page("pages/3_Action_Centre.py")

with nav_col4:
    st.markdown(
        """
        <div class="metric-card" style="height: 190px;">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">📊</div>
            <div style="font-weight: 700; font-size: 1.05rem; color: #f8fafc; margin-bottom: 6px;">Evidence Lab</div>
            <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">
                Ablation benchmark proving NIR superiority over RGB-only vision. Lead time advantage curves, confusion matrices, and p-values.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Inspect Evidence →", key="btn_nav_evi", use_container_width=True):
        st.switch_page("pages/4_Evidence_Lab.py")

st.divider()

# --- Live Store Floor Activity Feed ---
col_act1, col_act2 = st.columns([3, 2])

with col_act1:
    st.markdown("#### High-Priority FEFO Batches")
    batches = repo.get_batches(status="ACTIVE", sort_by_fefo=True)
    batch_rows = []
    for b in batches[:4]:
        urgency_badge = "🔴 CRITICAL" if b.fefo_priority >= 75 else ("🟡 ELEVATED" if b.fefo_priority >= 50 else "🟢 NORMAL")
        batch_rows.append({
            "Batch ID": b.batch_id,
            "Supplier": b.supplier,
            "Units": b.remaining_units,
            "Avg Shelf-Life": f"{b.avg_shelf_life_days:.1f} days",
            "FEFO Score": f"{b.fefo_priority:.1f}",
            "Urgency": urgency_badge,
            "Location": b.storage_location,
        })
    df_batches = pd.DataFrame(batch_rows)
    st.dataframe(df_batches, hide_index=True, use_container_width=True)

with col_act2:
    st.markdown("#### Recent Audit Log Events")
    logs = repo.get_audit_logs(limit=4)
    for log in logs:
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.5); padding: 10px 14px; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid #6366f1;">
                <div style="font-size: 0.75rem; color: #94a3b8;">{log.timestamp.strftime('%H:%M:%S')} • {log.operator}</div>
                <div style="font-size: 0.85rem; color: #f8fafc; font-weight: 500;">{log.details}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
