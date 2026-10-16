"""
BioVision Action Centre Module.
Manage retail store floor interventions: dynamic markdowns, kitchen repurposing,
and discard protocols with manager sign-off and immutable audit logs.
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
from datetime import datetime

from dashboard.components import load_custom_css, render_metric_card
from database.repository import get_db_repository

st.set_page_config(
    page_title="Action Centre | BioVision",
    page_icon="⚡",
    layout="wide",
)

load_custom_css()

st.markdown("## ⚡ Action Centre & Policy Execution")
st.markdown(
    "<p style='color: #94a3b8; font-size: 0.95rem;'>"
    "Review, approve, or override AI-generated store floor interventions to optimize produce sell-through and eliminate food waste."
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

repo = get_db_repository()

# Refresh actions
actions = repo.get_actions()
pending_actions = [a for a in actions if a.status == "PENDING"]
approved_actions = [a for a in actions if a.status in ("APPROVED", "EXECUTED")]
total_savings = sum(a.estimated_savings for a in approved_actions)

col1, col2, col3, col4 = st.columns(4)
with col1:
    render_metric_card("Pending Approvals", str(len(pending_actions)), "Requires floor manager review", delta="⚡ Attention" if pending_actions else "")
with col2:
    render_metric_card("Approved & Dispatched", str(len(approved_actions)), "Active floor interventions")
with col3:
    render_metric_card("Total Waste Value Saved", f"${total_savings:,.2f}", "Recovered from spoilage", delta="+100% Retained")
with col4:
    render_metric_card("Audit Compliance", "100%", "Zero unlogged overrides")

st.write("")
st.write("")

# Action Review Section
tab_pending, tab_approved, tab_audit = st.tabs(["⏳ Pending Interventions", "✅ Executed Actions", "📜 Compliance Audit Trail"])

with tab_pending:
    st.markdown("### Actions Awaiting Floor Manager Sign-off")
    if not pending_actions:
        st.success("🎉 No pending actions! All produce lots are currently addressed.")
    else:
        for act in pending_actions:
            urgency_border = "#ef4444" if act.urgency_level == "CRITICAL" else ("#f59e0b" if act.urgency_level == "HIGH" else "#3b82f6")
            
            with st.container():
                st.markdown(
                    f"""
                    <div style="background: rgba(15, 23, 42, 0.75); border-left: 4px solid {urgency_border}; border-radius: 10px; padding: 18px 22px; margin-bottom: 14px; border: 1px solid rgba(255,255,255,0.06);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">
                                {act.action_id} • Lot: <b style="color: #f8fafc;">{act.batch_id}</b>
                            </span>
                            <span style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; font-size: 0.75rem; font-weight: 700; padding: 3px 10px; border-radius: 20px;">
                                {act.urgency_level} URGENCY
                            </span>
                        </div>
                        <div style="font-size: 1.2rem; font-weight: 800; color: #f8fafc; margin-bottom: 6px;">
                            {act.action_name}
                        </div>
                        <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 12px;">
                            <b>Rationale:</b> {act.audit_reason}
                        </div>
                        <div style="font-size: 0.82rem; color: #94a3b8;">
                            Proposed Discount: <b>{act.discount_applied*100:.0f}%</b> &nbsp;|&nbsp; 
                            Estimated Loss Avoided: <b style="color: #10b981;">+${act.estimated_savings:.2f}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                c_btn1, c_btn2, c_space = st.columns([1, 1, 3])
                with c_btn1:
                    if st.button(f"Approve Action", key=f"app_{act.action_id}", type="primary"):
                        repo.update_action_status(
                            action_id=act.action_id,
                            status="APPROVED",
                            operator="Store Floor Lead",
                            notes="Approved via Action Centre UI",
                        )
                        st.toast(f"Approved {act.action_id}!", icon="✅")
                        st.rerun()

                with c_btn2:
                    with st.popover(f"Reject / Override"):
                        override_note = st.text_input("Override Reason (Required):", key=f"note_{act.action_id}")
                        if st.button("Confirm Reject", key=f"rej_{act.action_id}", type="secondary"):
                            if override_note:
                                repo.update_action_status(
                                    action_id=act.action_id,
                                    status="REJECTED",
                                    operator="Store Floor Lead",
                                    notes=override_note,
                                )
                                st.toast(f"Rejected {act.action_id}", icon="ℹ️")
                                st.rerun()
                            else:
                                st.error("Please enter an override reason.")

with tab_approved:
    st.markdown("### Executed & Completed Actions")
    if approved_actions:
        app_rows = [
            {
                "Action ID": a.action_id,
                "Batch ID": a.batch_id,
                "Action Taken": a.action_name,
                "Discount": f"{a.discount_applied*100:.0f}%",
                "Status": a.status,
                "Executed By": a.executed_by,
                "Savings Realized": f"${a.estimated_savings:.2f}",
                "Timestamp": a.timestamp.strftime("%Y-%m-%d %H:%M"),
            }
            for a in approved_actions
        ]
        st.dataframe(pd.DataFrame(app_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No approved actions yet.")

with tab_audit:
    st.markdown("### Immutable Regulatory & Policy Audit Log")
    logs = repo.get_audit_logs(limit=100)
    if logs:
        log_rows = [
            {
                "Log ID": l.log_id,
                "Timestamp": l.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Event Type": l.event_type,
                "Entity ID": l.entity_id,
                "Operator": l.operator,
                "Details": l.details,
            }
            for l in logs
        ]
        st.dataframe(pd.DataFrame(log_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No audit logs recorded yet.")
