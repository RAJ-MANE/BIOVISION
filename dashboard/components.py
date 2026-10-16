"""
Reusable UI widgets, Plotly visualization charts, and custom styled HTML components for BioVision.
"""
from typing import Dict, List, Any, Optional
import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def load_custom_css():
    """Inject glassmorphic enterprise CSS into the Streamlit session."""
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_metric_card(label: str, value: str, subtext: str = "", delta: Optional[str] = None):
    """Render a premium glassmorphic metric card."""
    delta_html = f"<span style='color: #10b981; font-weight: 600;'>{delta}</span> " if delta else ""
    html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-subtext">{delta_html}{subtext}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_status_badge(text: str, badge_type: str = "info"):
    """Render a colored status badge: success, warning, danger, info."""
    return f'<span class="badge badge-{badge_type}">{text}</span>'


def plot_ripeness_gauge(stage_idx: int, probabilities: List[float]) -> go.Figure:
    """Plot probability distribution across 5 ripeness stages."""
    stages = [
        "Stage 1: Hard",
        "Stage 2: Breaking",
        "Stage 3: Firm Ripe",
        "Stage 4: Ripe/Soft",
        "Stage 5: Overripe",
    ]
    colors = ["#10b981", "#3b82f6", "#6366f1", "#f59e0b", "#ef4444"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=[p * 100 for p in probabilities],
                y=stages,
                orientation="h",
                marker=dict(
                    color=colors,
                    line=dict(color="rgba(255, 255, 255, 0.2)", width=1),
                ),
                text=[f"{p*100:.1f}%" for p in probabilities],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=20, t=10, b=10),
        height=220,
        xaxis=dict(title="Model Confidence (%)", range=[0, 100], showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_lead_time_comparison(days: List[int], nir_curve: List[float], rgb_curve: List[float]) -> go.Figure:
    """Plot early degradation detection curves: NIR subsurface vs RGB surface browning."""
    fig = go.Figure()

    # NIR subsurface tissue breakdown
    fig.add_trace(
        go.Scatter(
            x=days,
            y=[v * 100 for v in nir_curve],
            mode="lines+markers",
            name="NIR Subsurface Spectral Degradation",
            line=dict(color="#06b6d4", width=3),
            marker=dict(size=7),
        )
    )

    # RGB visible surface browning
    fig.add_trace(
        go.Scatter(
            x=days,
            y=[v * 100 for v in rgb_curve],
            mode="lines+markers",
            name="RGB Surface Blemish / Browning",
            line=dict(color="#f43f5e", width=3, dash="dash"),
            marker=dict(size=7),
        )
    )

    # Add threshold line for decision trigger (40% degradation)
    fig.add_hline(
        y=40,
        line_dash="dot",
        line_color="#eab308",
        annotation_text="Early Retail Action Threshold (40%)",
        annotation_position="bottom right",
    )

    # Shaded intervention window
    fig.add_vrect(
        x0=3.8,
        x1=7.0,
        fillcolor="rgba(99, 102, 241, 0.15)",
        layer="below",
        line_width=0,
        annotation_text="3.2-Day Early Warning Window",
        annotation_position="top left",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20),
        height=320,
        xaxis=dict(title="Days Post-Harvest", dtick=1, showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Degradation Index (%)", range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_ablation_bar_chart(ablation_dict: Dict[str, Any]) -> go.Figure:
    """Plot accuracy and MAE comparison across the 3 ablation architectures."""
    models = list(ablation_dict.keys())
    accuracies = [ablation_dict[m]["accuracy"] for m in models]
    maes = [ablation_dict[m]["shelf_life_mae_days"] for m in models]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Classification Accuracy (%)",
            x=models,
            y=accuracies,
            marker_color=["#64748b", "#3b82f6", "#10b981"],
            text=[f"{a:.1f}%" for a in accuracies],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Shelf-Life Error MAE (Days - Lower is Better)",
            x=models,
            y=maes,
            marker_color=["#ef4444", "#f59e0b", "#06b6d4"],
            text=[f"{m:.2f}d" for m in maes],
            textposition="auto",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20),
        height=340,
        yaxis=dict(title="Metric Score", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_confusion_matrix(conf_matrix: List[List[int]], class_names: List[str]) -> go.Figure:
    """Plot confusion matrix heatmap."""
    z = np.array(conf_matrix)
    fig = px.imshow(
        z,
        text_auto=True,
        labels=dict(x="Predicted Ripeness Stage", y="Actual Ground Truth Stage", color="Specimen Count"),
        x=class_names,
        y=class_names,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=25, b=10),
        height=320,
    )
    return fig


def render_esl_tag(
    product_name: str,
    sku: str,
    old_price: float,
    new_price: float,
    discount_pct: float,
    shelf_life_days: float,
):
    """Render a simulated Electronic Shelf Label (ESL) with real-time price updates."""
    discount_html = f'<span class="esl-discount-tag">-{discount_pct*100:.0f}% OFF</span>' if discount_pct > 0 else ""
    old_price_html = f'<span class="esl-old-price">${old_price:.2f}/kg</span>' if discount_pct > 0 else ""
    
    html = f"""
    <div class="esl-container">
        <div class="esl-badge-sync">● ESL SYNCED</div>
        <div class="esl-product-name">{product_name}</div>
        <div class="esl-sku">SKU: {sku} • RETAIL SHELF TAG</div>
        <div class="esl-price-row">
            {old_price_html}
            <span class="esl-new-price">${new_price:.2f}<span style="font-size: 0.9rem; font-weight: 500; color: #94a3b8;">/kg</span></span>
            {discount_html}
        </div>
        <div class="esl-footer">
            <span>RSL: <b>{shelf_life_days:.1f}d Remaining</b></span>
            <span>FEFO DYNAMIC PRICING</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_verdict_hero(
    action: Dict[str, Any],
    economics: Dict[str, Any],
    shelf_life_days: float,
    latent_defect_prob: float,
):
    """Render a high-impact, associate-friendly operational verdict card."""
    urgency = action.get("urgency_level", "NORMAL")
    badge_type = "danger" if urgency == "CRITICAL" else ("warning" if urgency == "HIGH" else "success")
    badge_color = action.get("badge_color", "#10b981")

    html = f"""
    <div class="verdict-hero" style="border-left-color: {badge_color};">
        <div class="verdict-header">
            <span class="badge badge-{badge_type}">{urgency} ACTION REQUIRED</span>
            <span style="font-size: 0.8rem; color: #94a3b8; font-family: monospace;">CODE: {action.get('action_code', 'MAINTAIN')}</span>
        </div>
        <div class="verdict-title">{action.get('action_name', 'Maintain Full Price')}</div>
        <div class="verdict-body">{action.get('reasoning', '')}</div>
        <div class="verdict-stats">
            <div>
                <div class="vstat-label">Remaining Shelf-Life</div>
                <div class="vstat-value" style="color: #38bdf8;">{shelf_life_days:.1f} Days</div>
            </div>
            <div>
                <div class="vstat-label">Loss Avoided</div>
                <div class="vstat-value" style="color: #34d399;">+${economics.get('loss_avoided_vs_spoilage', 0.0):.2f}</div>
            </div>
            <div>
                <div class="vstat-label">Internal Decay Risk</div>
                <div class="vstat-value" style="color: {'#fb7185' if latent_defect_prob > 0.4 else '#94a3b8'};">
                    {latent_defect_prob*100:.0f}%
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_xray_divergence_alert(
    produce_name: str,
    rgb_stage_name: str,
    nir_degradation: float,
    lead_time_days: float = 3.2,
):
    """Render the high-impact pre-symptomatic divergence warning alert."""
    html = f"""
    <div class="divergence-alert">
        <div style="font-size: 1.8rem; line-height: 1;">⚡</div>
        <div>
            <div class="divergence-title">
                THE BIOVISION ADVANTAGE: Invisible Subsurface Spoilage Detected!
            </div>
            <div class="divergence-desc">
                Standard RGB cameras and human eyes only see <b>{rgb_stage_name}</b> (looks like 6–8 days fresh). 
                However, <b>850nm NIR penetration</b> reveals internal cellular breakdown (<b>{nir_degradation*100:.0f}% degradation</b>).
                BioVision flagged this <b>{lead_time_days:.1f} days before visible surface rotting</b>, preventing customer complaints and salvaging retail value.
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

