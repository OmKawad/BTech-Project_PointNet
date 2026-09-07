"""
Assembly Fit Risk Prediction – Interactive Streamlit Dashboard
================================================================
Premium 3D point cloud inspection & real-time risk prediction UI.

Features:
  • Generate or upload assembly point clouds
  • Interactive 3D Plotly visualisation (by Part ID, Normals, Risk Heatmap)
  • Real-time model inference with risk gauges
  • Risk level badge (Low / Medium / High)
  • Engineering recommendations
"""

import os
import sys
import json
import numpy as np
import torch
import streamlit as st

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.dataset_generator import generate_dataset, GENERATOR_MAP
from visualizer.plot_utils import (
    plot_point_cloud_by_part,
    plot_point_cloud_with_normals,
    plot_risk_heatmap,
    create_risk_gauge,
    compute_per_point_risk,
    RISK_COLORS,
)

# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Assembly Fit Risk Predictor",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS for premium dark theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }

    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Header */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d2ff 0%, #7b2ff7 50%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #8892b0;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* Risk badges */
    .risk-badge {
        display: inline-block;
        padding: 0.6rem 1.8rem;
        border-radius: 50px;
        font-size: 1.3rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        text-align: center;
        margin: 0.5rem auto;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .risk-low {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        box-shadow: 0 4px 20px rgba(46, 204, 113, 0.3);
    }
    .risk-medium {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        color: white;
        box-shadow: 0 4px 20px rgba(243, 156, 18, 0.3);
    }
    .risk-high {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        box-shadow: 0 4px 20px rgba(231, 76, 60, 0.3);
    }

    /* Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.2rem;
        backdrop-filter: blur(10px);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ccd6f6;
    }

    /* Recommendation box */
    .recommendation {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #7b2ff7;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        color: #ccd6f6;
        font-size: 0.95rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #ccd6f6;
    }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")

    input_mode = st.radio("Input Mode", ["Generate Sample", "Upload File"],
                          index=0, key="input_mode")

    if input_mode == "Generate Sample":
        assembly_type = st.selectbox(
            "Assembly Type",
            list(GENERATOR_MAP.keys()),
            format_func=lambda x: x.replace("_", " ").title(),
        )
        n_points = st.slider("Number of Points", 512, 4096, 2048, 256)

        st.markdown("---")
        st.markdown("### 🎚️ Defect Parameters")
        radial_dev = st.slider("Radial Deviation", -0.15, 0.15, 0.0, 0.005,
                                help="Positive → interference, Negative → excess clearance")
        tilt_angle = st.slider("Tilt Angle (°)", 0.0, 10.0, 0.0, 0.1,
                                help="Angular misalignment of mating axis")
        lateral_offset = st.slider("Lateral Offset", 0.0, 0.1, 0.0, 0.002,
                                    help="Radial eccentricity offset")
        noise_std = st.slider("Scanner Noise σ", 0.0, 0.01, 0.002, 0.001)

        generate_btn = st.button("🚀 Generate Assembly", use_container_width=True,
                                  type="primary")
    else:
        uploaded_file = st.file_uploader(
            "Upload Point Cloud (.npy)",
            type=["npy"],
            help="NumPy array of shape (N, 7): [x,y,z,nx,ny,nz,part_id]",
        )

    st.markdown("---")
    st.markdown("### 🎨 Visualisation")
    view_mode = st.selectbox("View Mode", [
        "Part ID Coloring",
        "Surface Normals",
        "Risk Heatmap",
    ])
    point_size = st.slider("Point Size", 1, 8, 3)

    st.markdown("---")
    st.markdown("### 🧠 Model")
    model_path = st.text_input("Checkpoint Path", "checkpoints/best_model.pth")
    use_model = st.checkbox("Run Model Inference", value=False,
                             help="Load trained model for risk prediction")


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown('<div class="hero-title">🔧 Assembly Fit Risk Predictor</div>',
            unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">PointNet Deep Learning with Spatial Transformer Networks (T-Net) for Assembly Quality Assessment</div>',
            unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Generate or load point cloud
# ──────────────────────────────────────────────
point_cloud = None
labels = None

if input_mode == "Generate Sample":
    if "point_cloud" not in st.session_state:
        st.session_state.point_cloud = None
        st.session_state.labels = None

    if generate_btn:
        from data.dataset_generator import AssemblyConfig, _compute_risk_labels

        cfg = AssemblyConfig(
            assembly_type=assembly_type,
            nominal_radius=1.0,
            height=2.0,
            radial_deviation=radial_dev,
            tilt_angle=tilt_angle,
            lateral_offset=lateral_offset,
            point_noise_std=noise_std,
        )

        generator_fn = GENERATOR_MAP[assembly_type]
        result = generator_fn(cfg, n_points)

        pts = result["points"].astype(np.float32)
        nrm = result["normals"].astype(np.float32)
        pids = result["part_ids"].astype(np.float32)

        # Add noise
        pts += np.random.normal(0, noise_std, pts.shape).astype(np.float32)
        nrm += np.random.normal(0, 0.01, nrm.shape).astype(np.float32)
        norms = np.linalg.norm(nrm, axis=-1, keepdims=True)
        nrm = nrm / np.clip(norms, 1e-8, None)

        pc = np.concatenate([pts, nrm, pids[:, None]], axis=-1)
        gt_labels = _compute_risk_labels(cfg)

        st.session_state.point_cloud = pc
        st.session_state.labels = gt_labels

    point_cloud = st.session_state.point_cloud
    labels = st.session_state.labels

else:
    if uploaded_file is not None:
        point_cloud = np.load(uploaded_file).astype(np.float32)
        if point_cloud.ndim != 2 or point_cloud.shape[1] != 7:
            st.error(f"Expected shape (N, 7), got {point_cloud.shape}")
            point_cloud = None


# ──────────────────────────────────────────────
# Main content
# ──────────────────────────────────────────────
if point_cloud is not None:
    # ── Model Inference ──
    model_preds = None
    if use_model and os.path.exists(os.path.join(ROOT, model_path)):
        try:
            from models.risk_predictor import AssemblyFitRiskModel
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(os.path.join(ROOT, model_path),
                                     map_location=device, weights_only=False)
            config = checkpoint["config"]
            model_cfg = config["model"]
            pn_cfg = model_cfg.get("pointnet", {})
            model = AssemblyFitRiskModel(
                backbone_type=model_cfg.get("backbone_type", "pointnet"),
                part_embed_dim=model_cfg.get("part_embed_dim", 16),
                max_parts=model_cfg.get("max_parts", 8),
                point_feat_dim=pn_cfg.get("point_feat_dim", 256),
                global_feat_dim=pn_cfg.get("global_feat_dim", 1024),
                use_tnet_input=pn_cfg.get("use_tnet_input", True),
                use_tnet_feature=pn_cfg.get("use_tnet_feature", True),
                num_heads=model_cfg["relation"]["num_heads"],
                relation_hidden_dim=model_cfg["relation"]["hidden_dim"],
                risk_hidden_dims=model_cfg["risk_head"]["hidden_dims"],
                dropout=0.0,
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            # Prepare input
            pc_tensor = torch.from_numpy(point_cloud).unsqueeze(0).to(device)
            # Normalize
            centroid = pc_tensor[:, :, :3].mean(dim=1, keepdim=True)
            pc_tensor[:, :, :3] -= centroid
            max_dist = pc_tensor[:, :, :3].norm(dim=-1).max()
            if max_dist > 1e-8:
                pc_tensor[:, :, :3] /= max_dist

            with torch.no_grad():
                model_preds = model(pc_tensor)
                model_preds = {k: v.cpu().numpy().squeeze()
                               for k, v in model_preds.items()}
        except Exception as e:
            st.warning(f"Model inference failed: {e}")

    # Determine which predictions to display
    display_preds = {}
    if model_preds is not None:
        display_preds = {
            "interference_risk": float(model_preds["interference_risk"]),
            "clearance_risk": float(model_preds["clearance_risk"]),
            "alignment_risk": float(model_preds["alignment_risk"]),
            "failure_prob": float(model_preds["failure_prob"]),
            "risk_level": int(model_preds["risk_level"]),
        }
        pred_source = "🧠 Model Prediction"
    elif labels is not None:
        display_preds = labels
        pred_source = "📐 Ground Truth (from generator)"
    else:
        # Estimate from geometry
        display_preds = {
            "interference_risk": 0.0,
            "clearance_risk": 0.0,
            "alignment_risk": 0.0,
            "failure_prob": 0.0,
            "risk_level": 0,
        }
        pred_source = "⚠️ No labels or model available"

    # ── Layout ──
    col_viz, col_risk = st.columns([3, 1])

    with col_viz:
        # 3D Visualisation
        if view_mode == "Part ID Coloring":
            fig = plot_point_cloud_by_part(point_cloud, point_size)
        elif view_mode == "Surface Normals":
            fig = plot_point_cloud_with_normals(point_cloud, point_size=point_size)
        else:
            risk_values = compute_per_point_risk(point_cloud)
            fig = plot_risk_heatmap(point_cloud, risk_values, point_size)

        st.plotly_chart(fig, use_container_width=True)

        # Point cloud info
        n_pts = len(point_cloud)
        n_parts = len(np.unique(point_cloud[:, 6].astype(int)))
        info_cols = st.columns(4)
        info_cols[0].metric("Points", f"{n_pts:,}")
        info_cols[1].metric("Parts", n_parts)
        info_cols[2].metric("Source", pred_source.split(" ")[0])
        info_cols[3].metric("View", view_mode.split(" ")[0])

    with col_risk:
        st.markdown(f"#### {pred_source}")

        # Risk Level Badge
        risk_level = display_preds.get("risk_level", 0)
        risk_names = {0: "Low", 1: "Medium", 2: "High"}
        risk_css = {0: "risk-low", 1: "risk-medium", 2: "risk-high"}
        risk_name = risk_names.get(risk_level, "Unknown")
        st.markdown(
            f'<div style="text-align:center;">'
            f'<div class="risk-badge {risk_css.get(risk_level, "")}">'
            f'{risk_name} Risk</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        # Risk gauges
        st.plotly_chart(
            create_risk_gauge(display_preds.get("interference_risk", 0), "Interference"),
            use_container_width=True, key="gauge_interf")
        st.plotly_chart(
            create_risk_gauge(display_preds.get("clearance_risk", 0), "Clearance"),
            use_container_width=True, key="gauge_clear")
        st.plotly_chart(
            create_risk_gauge(display_preds.get("alignment_risk", 0), "Alignment"),
            use_container_width=True, key="gauge_align")

        # Overall Failure Probability
        fail_prob = display_preds.get("failure_prob", 0)
        st.markdown(f"""
        <div class="metric-card" style="text-align:center; margin-top:1rem;">
            <div class="metric-label">Overall Failure Probability</div>
            <div class="metric-value">{fail_prob:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Engineering Recommendations ──
    st.markdown("---")
    st.markdown("### 📋 Engineering Recommendations")

    rec_cols = st.columns(3)
    interf = display_preds.get("interference_risk", 0)
    clear = display_preds.get("clearance_risk", 0)
    align = display_preds.get("alignment_risk", 0)

    with rec_cols[0]:
        if interf > 0.66:
            st.markdown(
                '<div class="recommendation">⚠️ <b>Critical Interference:</b> '
                'Material overlap detected at mating bore. Machining adjustment or '
                'dimensional rework required before assembly.</div>',
                unsafe_allow_html=True)
        elif interf > 0.33:
            st.markdown(
                '<div class="recommendation">🔶 <b>Moderate Interference:</b> '
                'Slight dimensional tightness. Verify tolerance stack-up; '
                'consider selective assembly or thermal fitting.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="recommendation">✅ <b>Interference OK:</b> '
                'Mating dimensions within acceptable clearance fit range.</div>',
                unsafe_allow_html=True)

    with rec_cols[1]:
        if clear > 0.66:
            st.markdown(
                '<div class="recommendation">⚠️ <b>Excessive Clearance:</b> '
                'Gap exceeds allowable limits. Risk of vibration and premature wear. '
                'Consider shimming or re-machining.</div>',
                unsafe_allow_html=True)
        elif clear > 0.33:
            st.markdown(
                '<div class="recommendation">🔶 <b>Moderate Clearance:</b> '
                'Slightly loose fit. Monitor for thermal expansion effects '
                'and verify under operational loads.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="recommendation">✅ <b>Clearance OK:</b> '
                'Gap is within nominal tolerance band.</div>',
                unsafe_allow_html=True)

    with rec_cols[2]:
        if align > 0.66:
            st.markdown(
                '<div class="recommendation">⚠️ <b>Severe Misalignment:</b> '
                'Axis tilt and eccentricity beyond limits. Re-fixture and '
                'realign before proceeding with assembly.</div>',
                unsafe_allow_html=True)
        elif align > 0.33:
            st.markdown(
                '<div class="recommendation">🔶 <b>Moderate Misalignment:</b> '
                'Minor angular deviation detected. Verify datum references '
                'and assembly fixtures.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="recommendation">✅ <b>Alignment OK:</b> '
                'Mating axis alignment within acceptable angular tolerance.</div>',
                unsafe_allow_html=True)

    # ── PointNet Architecture Properties ──
    st.markdown("---")
    with st.expander("🔬 PointNet Mathematical Properties & T-Net Spatial Alignment", expanded=False):
        pn_col1, pn_col2 = st.columns(2)
        with pn_col1:
            st.markdown("#### 🔄 Permutation Invariance")
            st.write("PointNet guarantees symmetric processing across all point permutations through symmetric channel-wise max pooling: $$g(\{x_1, \dots, x_n\}) = \max_{i=1 \dots n} h(x_i)$$")
            st.markdown("#### 📐 Coordinate Alignment (T-Net 3×3)")
            if model_preds is not None and "trans_input" in model_preds and model_preds["trans_input"] is not None:
                t3 = model_preds["trans_input"]
                st.write("Learned 3D Spatial Transformer Matrix $T_3$:")
                st.dataframe(t3.round(4))
                det = float(np.linalg.det(t3))
                st.caption(f"Matrix Determinant: {det:.4f} (Ideal Rigid: ~1.0)")
            else:
                st.info("Run Model Inference with a trained checkpoint to inspect learned T-Net transformation matrices.")

        with pn_col2:
            st.markdown("#### 🧬 Feature Space Alignment (T-Net 64×64)")
            st.write("Aligns 64-dimensional intermediate geometric features before deep embedding.")
            st.markdown("#### ⚖️ Orthogonal Regularization Constraint")
            st.write("Constrains high-dimensional feature transform $A$ to approximate rigid isometry: $$L_{\\text{reg}} = \\frac{1}{K^2} \\|I - A A^T\\|_F^2$$")
            if model_preds is not None and "trans_feat" in model_preds and model_preds["trans_feat"] is not None:
                t64 = model_preds["trans_feat"]
                diff = np.matmul(t64, t64.T) - np.eye(64)
                reg_val = float(np.linalg.norm(diff) / (64 * 64))
                st.metric("Feature Transform Orthogonality Penalty", f"{reg_val:.6f}")


else:
    # Landing state
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🔩</div>
        <h3 style="color: #ccd6f6;">Generate or Upload an Assembly Point Cloud</h3>
        <p style="color: #8892b0; max-width: 500px; margin: 0 auto;">
            Use the sidebar controls to generate a synthetic mechanical assembly
            with custom tolerance parameters, or upload an existing point cloud file.
        </p>
    </div>
    """, unsafe_allow_html=True)
