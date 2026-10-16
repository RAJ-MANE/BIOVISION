"""
BioVision Inventory Intelligence Module.
Dynamic First-Expired-First-Out (FEFO) prioritization across warehouse pallets and retail shelf displays.
"""
import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import plotly.express as px

from dashboard.components import load_custom_css, render_metric_card
from database.repository import get_db_repository

st.set_page_config(
    page_title="Inventory Intelligence | BioVision",
    page_icon="📦",
    layout="wide",
)

load_custom_css()

st.markdown("## 📦 Inventory Intelligence & FEFO Prioritization")
st.markdown(
    "<p style='color: #94a3b8; font-size: 0.95rem;'>"
    "Real-time perishable produce inventory dynamically ranked by neural shelf-life prediction and cellular degradation index."
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

repo = get_db_repository()
batches = repo.get_batches(sort_by_fefo=True)

# Top KPI row
critical_count = sum(1 for b in batches if b.fefo_priority >= 75)
elevated_count = sum(1 for b in batches if 50 <= b.fefo_priority < 75)
normal_count = sum(1 for b in batches if b.fefo_priority < 50)
total_units = sum(b.remaining_units for b in batches)

col1, col2, col3, col4 = st.columns(4)
with col1:
    render_metric_card("Monitored Lots", str(len(batches)), "Active supplier pallets")
with col2:
    render_metric_card("Critical FEFO Lots", str(critical_count), "Urgent routing required (<48h)", delta="Action Needed" if critical_count > 0 else "")
with col3:
    render_metric_card("Elevated Velocity Lots", str(elevated_count), "Candidate for 20% markdown")
with col4:
    render_metric_card("Stable Prime Lots", str(normal_count), "Full shelf-life remaining (>6 days)")

st.write("")
st.write("")

# Filtering and search
f_col1, f_col2, f_col3, f_col4 = st.columns([2, 1, 1, 1])
with f_col1:
    search_query = st.text_input("🔍 Search by Batch ID, Supplier, or Location", "")
with f_col2:
    produce_filter = st.selectbox("Produce Type", ["All Produce", "Avocado", "Mango", "Apple", "Strawberry"])
with f_col3:
    status_filter = st.selectbox("Status Filter", ["All Statuses", "ACTIVE", "HOLD", "MARKDOWN", "CLOSED"])
with f_col4:
    urgency_filter = st.selectbox("Urgency Filter", ["All Urgencies", "Critical Only (FEFO ≥ 75)", "Stable Only (FEFO < 50)"])

# Filter logic
filtered_batches = batches
if search_query:
    filtered_batches = [
        b for b in filtered_batches
        if search_query.lower() in b.batch_id.lower()
        or search_query.lower() in b.supplier.lower()
        or search_query.lower() in b.storage_location.lower()
    ]
if produce_filter != "All Produce":
    filtered_batches = [b for b in filtered_batches if produce_filter.lower() in b.produce_type.lower()]
if status_filter != "All Statuses":
    filtered_batches = [b for b in filtered_batches if b.current_status == status_filter]
if urgency_filter == "Critical Only (FEFO ≥ 75)":
    filtered_batches = [b for b in filtered_batches if b.fefo_priority >= 75]
elif urgency_filter == "Stable Only (FEFO < 50)":
    filtered_batches = [b for b in filtered_batches if b.fefo_priority < 50]

# Inventory Table
st.markdown("### Active Produce Lots (FEFO Sorted)")

table_data = []
for b in filtered_batches:
    urgency_color = "#ef4444" if b.fefo_priority >= 75 else ("#f59e0b" if b.fefo_priority >= 50 else "#10b981")
    urgency_label = "CRITICAL (Move Now)" if b.fefo_priority >= 75 else ("ELEVATED" if b.fefo_priority >= 50 else "STABLE")

    table_data.append({
        "Batch ID": b.batch_id,
        "Produce": b.produce_type,
        "Supplier": b.supplier,
        "Received": b.received_date.strftime("%Y-%m-%d"),
        "Units Remaining": b.remaining_units,
        "Location": b.storage_location,
        "Est. Shelf-Life": f"{b.avg_shelf_life_days:.1f} days",
        "Degradation": f"{b.avg_degradation_score:.2f}",
        "FEFO Priority": round(b.fefo_priority, 1),
        "Urgency": urgency_label,
    })

if table_data:
    df_table = pd.DataFrame(table_data)
    st.dataframe(
        df_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "FEFO Priority": st.column_config.ProgressColumn(
                "FEFO Priority",
                help="Urgency score (0-100) combining neural shelf life, degradation score, and lot age.",
                format="%.1f",
                min_value=0,
                max_value=100,
            ),
        },
    )
else:
    st.info("No batches match the selected criteria.")

st.write("")
st.divider()

# Batch Drilldown Inspector
st.markdown("### 🔍 Lot Freshness & Inspection Details")
selected_b_id = st.selectbox("Select Batch to Inspect:", [b.batch_id for b in filtered_batches] if filtered_batches else ["None"])

if selected_b_id != "None":
    sel_batch = next((b for b in batches if b.batch_id == selected_b_id), None)
    if sel_batch:
        b_col1, b_col2, b_col3 = st.columns([1, 1, 2])
        with b_col1:
            st.markdown(f"**Supplier:** {sel_batch.supplier}")
            st.markdown(f"**Storage Zone:** {sel_batch.storage_location}")
            st.markdown(f"**Units:** {sel_batch.remaining_units} / {sel_batch.total_units}")
        with b_col2:
            st.markdown(f"**Avg Shelf-Life:** `{sel_batch.avg_shelf_life_days:.1f} days`")
            st.markdown(f"**Degradation Index:** `{sel_batch.avg_degradation_score:.2f}`")
            st.markdown(f"**Last Scanned:** {sel_batch.last_scanned_at.strftime('%Y-%m-%d %H:%M') if sel_batch.last_scanned_at else 'Pending'}")
        with b_col3:
            # Action recommendation card for this batch
            if sel_batch.fefo_priority >= 75:
                rec_title = "Route to Kitchen Deli (Guacamole) / Apply 50% Markdown"
                rec_bg = "rgba(239, 68, 68, 0.15)"
                rec_border = "#ef4444"
                rec_body = "Accelerated internal degradation detected. Move within 24 hours to prevent total inventory loss."
            elif sel_batch.fefo_priority >= 50:
                rec_title = "Front-of-Shelf Display / 20% Quick-Sale Markdown"
                rec_bg = "rgba(245, 158, 11, 0.15)"
                rec_border = "#f59e0b"
                rec_body = "Promote to forward customer facing bin to drive sell-through ahead of day 5 softness."
            else:
                rec_title = "Maintain Premium Cold Storage"
                rec_bg = "rgba(16, 185, 129, 0.15)"
                rec_border = "#10b981"
                rec_body = "Cellular tissue is dense and intact. Safe for reserve inventory or regular retail display."

            st.markdown(
                f"""
                <div style="background: {rec_bg}; border: 1px solid {rec_border}; border-radius: 10px; padding: 14px 18px;">
                    <div style="font-weight: 700; color: #f8fafc; font-size: 0.95rem; margin-bottom: 4px;">{rec_title}</div>
                    <div style="color: #cbd5e1; font-size: 0.85rem;">{rec_body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Scans history for this batch
        scans = repo.get_scans_by_batch(sel_batch.batch_id)
        if scans:
            st.markdown("#### Optical Scan History for this Lot")
            scan_rows = [
                {
                    "Scan ID": s.scan_id,
                    "Timestamp": s.timestamp.strftime("%Y-%m-%d %H:%M"),
                    "Stage": f"Stage {s.predicted_stage} ({s.predicted_stage_name})",
                    "Shelf Life": f"{s.predicted_shelf_life_days:.1f}d [{s.ci_lower_days:.1f}d - {s.ci_upper_days:.1f}d]",
                    "Degradation": f"{s.degradation_score:.2f}",
                    "Spectral Index": f"{s.spectral_contrast_index:.3f}",
                    "Reliability": s.reliability,
                }
                for s in scans
            ]
            st.dataframe(pd.DataFrame(scan_rows), hide_index=True, use_container_width=True)
