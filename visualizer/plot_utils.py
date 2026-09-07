"""
3D Point Cloud Plotting Utilities
===================================
Plotly-based interactive 3D visualization for assembly point clouds.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple


# Color palettes
PART_COLORS = [
    "#FF6B6B",  # Part 0 – Coral Red
    "#4ECDC4",  # Part 1 – Teal
    "#45B7D1",  # Part 2 – Sky Blue
    "#96CEB4",  # Part 3 – Sage Green
    "#FFEAA7",  # Part 4 – Light Yellow
    "#DDA0DD",  # Part 5 – Plum
    "#98D8C8",  # Part 6 – Mint
    "#F7DC6F",  # Part 7 – Gold
]

RISK_COLORS = {
    "Low": "#2ECC71",     # Green
    "Medium": "#F39C12",  # Orange
    "High": "#E74C3C",    # Red
}

RISK_HEATMAP_COLORSCALE = [
    [0.0, "#2ECC71"],   # Green (safe)
    [0.33, "#F1C40F"],  # Yellow
    [0.66, "#E67E22"],  # Orange
    [1.0, "#E74C3C"],   # Red (danger)
]


def plot_point_cloud_by_part(point_cloud: np.ndarray,
                              point_size: int = 3,
                              title: str = "Assembly Point Cloud (by Part ID)"
                              ) -> go.Figure:
    """
    Plot 3D point cloud colored by part_id.

    Args:
        point_cloud: (N, 7) – [x, y, z, nx, ny, nz, part_id]
        point_size: marker size
        title: plot title
    """
    fig = go.Figure()
    xyz = point_cloud[:, :3]
    part_ids = point_cloud[:, 6].astype(int)
    unique_parts = np.unique(part_ids)

    for pid in unique_parts:
        mask = part_ids == pid
        color = PART_COLORS[pid % len(PART_COLORS)]
        fig.add_trace(go.Scatter3d(
            x=xyz[mask, 0], y=xyz[mask, 1], z=xyz[mask, 2],
            mode="markers",
            marker=dict(size=point_size, color=color, opacity=0.8),
            name=f"Part {pid}",
            hovertemplate=(
                "x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<br>"
                f"Part: {pid}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
            bgcolor="rgb(20, 20, 30)",
        ),
        paper_bgcolor="rgb(15, 15, 25)",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(30, 30, 50, 0.8)"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=550,
    )
    return fig


def plot_point_cloud_with_normals(point_cloud: np.ndarray,
                                   normal_scale: float = 0.02,
                                   point_size: int = 2,
                                   title: str = "Point Cloud with Surface Normals"
                                   ) -> go.Figure:
    """
    Plot point cloud with surface normal vectors as quiver arrows.
    """
    fig = go.Figure()
    xyz = point_cloud[:, :3]
    normals = point_cloud[:, 3:6]
    part_ids = point_cloud[:, 6].astype(int)

    # Subsample for performance (normals are dense)
    n = len(xyz)
    if n > 500:
        idx = np.random.choice(n, 500, replace=False)
    else:
        idx = np.arange(n)

    # Points
    for pid in np.unique(part_ids):
        mask = part_ids == pid
        color = PART_COLORS[pid % len(PART_COLORS)]
        fig.add_trace(go.Scatter3d(
            x=xyz[mask, 0], y=xyz[mask, 1], z=xyz[mask, 2],
            mode="markers",
            marker=dict(size=point_size, color=color, opacity=0.6),
            name=f"Part {pid}",
        ))

    # Normal vectors (as lines)
    for i in idx:
        start = xyz[i]
        end = start + normals[i] * normal_scale
        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
            mode="lines",
            line=dict(color="yellow", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
            bgcolor="rgb(20, 20, 30)",
        ),
        paper_bgcolor="rgb(15, 15, 25)",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=550,
    )
    return fig


def plot_risk_heatmap(point_cloud: np.ndarray,
                       risk_values: np.ndarray,
                       point_size: int = 4,
                       title: str = "Mating Surface Risk Heatmap"
                       ) -> go.Figure:
    """
    Plot point cloud colored by per-point risk values (heatmap).

    Args:
        point_cloud: (N, 7)
        risk_values: (N,) risk score per point ∈ [0, 1]
    """
    xyz = point_cloud[:, :3]

    fig = go.Figure(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode="markers",
        marker=dict(
            size=point_size,
            color=risk_values,
            colorscale=RISK_HEATMAP_COLORSCALE,
            cmin=0, cmax=1,
            colorbar=dict(
                title="Risk",
                tickvals=[0, 0.33, 0.66, 1.0],
                ticktext=["Safe", "Low", "Med", "High"],
            ),
            opacity=0.85,
        ),
        hovertemplate=(
            "x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<br>"
            "Risk: %{marker.color:.3f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
            bgcolor="rgb(20, 20, 30)",
        ),
        paper_bgcolor="rgb(15, 15, 25)",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=550,
    )
    return fig


def create_risk_gauge(value: float, title: str,
                       max_val: float = 1.0) -> go.Figure:
    """Create a gauge/indicator chart for a single risk value."""
    if value < 0.33:
        color = RISK_COLORS["Low"]
    elif value < 0.66:
        color = RISK_COLORS["Medium"]
    else:
        color = RISK_COLORS["High"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title=dict(text=title, font=dict(size=14, color="white")),
        number=dict(suffix="%", font=dict(size=24, color="white")),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color="gray")),
            bar=dict(color=color),
            bgcolor="rgb(30, 30, 50)",
            bordercolor="gray",
            steps=[
                dict(range=[0, 33], color="rgba(46, 204, 113, 0.2)"),
                dict(range=[33, 66], color="rgba(243, 156, 18, 0.2)"),
                dict(range=[66, 100], color="rgba(231, 76, 60, 0.2)"),
            ],
            threshold=dict(
                line=dict(color="white", width=2),
                thickness=0.75,
                value=value * 100,
            ),
        ),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=200,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


def compute_per_point_risk(point_cloud: np.ndarray) -> np.ndarray:
    """
    Estimate per-point risk based on proximity to the mating interface.

    Points closer to the interface between Part A and Part B
    receive higher risk scores.
    """
    xyz = point_cloud[:, :3]
    part_ids = point_cloud[:, 6].astype(int)

    mask_a = part_ids == 0
    mask_b = part_ids != 0

    if not mask_a.any() or not mask_b.any():
        return np.zeros(len(xyz))

    xyz_a = xyz[mask_a]
    xyz_b = xyz[mask_b]

    # Distance from each point to nearest point of the OTHER part
    risk_scores = np.zeros(len(xyz))

    # For Part A points: distance to nearest Part B point
    from scipy.spatial import cKDTree
    tree_b = cKDTree(xyz_b)
    dist_a, _ = tree_b.query(xyz_a)
    tree_a = cKDTree(xyz_a)
    dist_b, _ = tree_a.query(xyz_b)

    # Normalize: closer = higher risk
    max_dist = max(dist_a.max(), dist_b.max(), 1e-8)
    risk_a = 1.0 - np.clip(dist_a / (max_dist * 0.3), 0, 1)
    risk_b = 1.0 - np.clip(dist_b / (max_dist * 0.3), 0, 1)

    risk_scores[mask_a] = risk_a
    risk_scores[mask_b] = risk_b

    return risk_scores
