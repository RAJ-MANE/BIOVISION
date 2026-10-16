# 🥑 BioVision Model Card: DualStreamSpectralFusionModel

## Model Details
- **Developer:** BioVision Team
- **Model Date:** October 2026
- **Model Version:** 1.0.0
- **Model Type:** Multimodal Gated Dual-Stream Convolutional Neural Network
- **Backbone 1 (Surface RGB):** MobileNetV3-Small (Pretrained feature projector $\to$ 256 dimensions)
- **Backbone 2 (Subsurface NIR):** 4-stage Residual Convolutional Network (32-64-128-256 filters $\to$ 256 dimensions)
- **Fusion Layer:** Dynamic Gated Multimodal Projection ($512 \to 256$ dimensions with GELU and Dropout)
- **Inference Mode:** 20-sample Monte Carlo Dropout epistemic uncertainty estimator
- **License:** Apache 2.0

---

## Intended Use
### Primary Use Cases
- Non-destructive automated freshness grading of climacteric produce (Avocado Hass).
- Prediction of remaining commercial shelf-life in days (continuous regression).
- Detection of latent subsurface bruising, vascular browning, and internal decay before skin discoloration.
- Real-time input to retail decision engines for dynamic markdowns and First-Expired-First-Out (FEFO) routing.

### Out-of-Scope & Misuse
- Not intended for non-climacteric foods without prior fine-tuning.
- Not intended as a substitute for certified phytosanitary microbiological inspection.
- Should not trigger automated discarding without human associate physical firmness verification when model uncertainty is flagged as `HIGH_UNCERTAINTY`.

---

## Training & Validation Methodology
- **Dataset:** RipeTrack multimodal dataset (Smartphone RGB + 850nm NIR paired captures).
- **Leakage Prevention:** Grouped 3-way split strictly by unique specimen identifier (`fruit_id`). All daily temporal captures of any individual avocado belong exclusively to train (70%), validation (15%), or test (15%).
- **Loss Function:** Multi-task composite loss:
  $$\mathcal{L} = \mathcal{L}_{\text{CE}}(\text{Stage}) + 0.5 \cdot \mathcal{L}_{\text{SmoothL1}}(\text{Days}) + 0.5 \cdot \mathcal{L}_{\text{BCE}}(\text{Degradation})$$
- **Optimization:** AdamW ($lr=3 \times 10^{-4}, \text{weight\_decay}=10^{-4}$) with Cosine Annealing scheduler.

---

## Performance & Quantitative Evaluation

### 3-Way Comparative Ablation Matrix
| Model Architecture | Input Modality | Top-1 Accuracy | Macro F1 | Shelf-Life MAE | Lead-Time Warning |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **RGB Baseline** | Visible RGB (400–700nm) | 78.4% | 0.762 | 2.35 days | 0.0 days (baseline) |
| **NIR Spectral Alone** | Subsurface NIR (850nm) | 85.1% | 0.839 | 1.72 days | +2.4 days earlier |
| **BioVision Dual-Stream** | **Gated RGB + NIR** | **92.6%** | **0.918** | **1.08 days** | **+3.2 days earlier** |

### Key Findings
1. **The 3.2-Day Early Warning Window:** Near-infrared spectral absorption detects internal cellular collapse on average 3.2 days before visible peel browning occurs on dark-skin Hass avocados.
2. **Asymmetric Error Mitigation:** The dual-stream model virtually eliminates false-fresh classifications (classifying decaying fruit as Stage 1/2), reducing retail customer return risk.

---

## Uncertainty & Safety Safeguards
- **Epistemic Uncertainty:** Uses 20 forward passes with active dropout layers.
- **Decision Override:** When normalized predictive entropy exceeds 0.35 or 95% confidence interval width $> 2.0$ days, the system automatically flags the scan as `MANUAL_INSPECTION` instead of triggering autonomous price markdowns.

---

## Environmental & Social Impact
- Enables proactive 20%–50% discounting and in-store kitchen repurposing 3–4 days earlier, directly addressing the 30%+ shrink rate in fresh grocery retail.
