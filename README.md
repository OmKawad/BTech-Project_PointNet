# Assembly Fit Risk Prediction using PointNet

> **BTech Final Year Project**
> Deep Learning-based Assembly Fit Risk Assessment from 3D Point Clouds using Canonical PointNet Architecture and Its Invariance Properties

---

## 🎯 Problem Statement

Mechanical assemblies require precise dimensional fit between mating components. Traditional inspection methods (manual CMM measurement, rigid ICP registration) cannot predict assembly fit risk **before physical assembly**. Existing deep learning approaches focus on either:
- Point cloud registration only (ICP, DCP, PointNetLK)
- Generic object classification/segmentation (ModelNet40 single objects)
- Joint prediction without tolerance analysis (JoinABLe)
- Disassembly sequencing only (AssemblyNet)

**None** of them predict **assembly fit quality** including interference, clearance, alignment, and failure risk directly from raw scan points.

---

## 🚀 Our PointNet Approach & Canonical Properties

We build upon canonical **PointNet** (Qi et al., CVPR 2017) adapted for 7D assembly point clouds, incorporating its defining mathematical properties:

```
Input: Point Cloud (x, y, z, nx, ny, nz, part_id)
    │
    ▼
[Part ID Embedding (16D) + Surface Normals]
    │
    ▼
[T-Net 3×3 Spatial Alignment]  ──► Canonical Coordinate Alignment
    │
    ▼
[Shared MLP: 64 → 64]
    │
    ▼
[T-Net 64×64 Feature Alignment] ──► Canonical Feature Space Alignment (with ||I - A Aᵀ||² Orthogonal Loss)
    │
    ▼
[Shared MLP: 64 → 128 → 1024]
    │
    ▼
[Symmetric Max-Pooling] ────────► Permutation-Invariant Global Shape Vector (1024D)
    │
    ▼
[Local + Global Concatenation: 1088 → 256] ──► Rich Point-Wise Descriptors
    │
    ▼
[Assembly Relationship Analysis] ──► Cross-Part Attention (A ↔ B) + Interface Extraction
    │
    ▼
[Multi-Task Risk Prediction Heads]
    │
    ├─► Interference Risk ∈ [0, 1]
    ├─► Clearance Risk ∈ [0, 1]
    ├─► Alignment Risk ∈ [0, 1]
    ├─► Fit Failure Probability ∈ [0, 1]
    └─► Categorical Risk Level ∈ {Low, Medium, High}
```

### PointNet Mathematical Foundations
1. **Permutation Invariance via Symmetric Function**:
   Since a point cloud is an unordered set, the global feature extractor is invariant to any of the $N!$ point orderings:
   $$g(\{x_1, \dots, x_N\}) = \gamma\left(\max_{i=1 \dots N} h(x_i)\right)$$
2. **Input Coordinate Alignment (T-Net 3×3)**:
   A mini-PointNet network estimates a $3 \times 3$ affine transformation matrix $T_3$ directly from input point coordinates, aligning assemblies into a canonical orientation.
3. **Feature Space Alignment (T-Net 64×64)**:
   A second mini-PointNet network estimates a $64 \times 64$ transformation matrix $T_{64}$ on intermediate point features.
4. **Orthogonal Regularization Loss**:
   To prevent arbitrary stretching/distortion in 64-dimensional feature space, an orthogonality constraint is enforced:
   $$L_{\text{reg}} = \frac{1}{K^2} \|I - A A^T\|_F^2$$
5. **Local + Global Feature Fusion**:
   Local point features ($64$-dim) are concatenated with the duplicated global shape signature ($1024$-dim) to produce $1088$-dimensional local-global descriptors refined to $256$-dim per point.

---

## 📁 Project Structure

```
btechproject/
├── data/
│   ├── dataset_generator.py      # Synthetic CAD assembly generator (shaft-hole, flange, peg, dovetail)
│   ├── dataset.py                # PyTorch Dataset & DataLoader
│   └── samples/                  # Pre-generated point cloud data
├── models/
│   ├── pointnet_utils.py         # PointNet primitives (T-Net 3x3, T-Net 64x64, regularizer, encoder)
│   ├── pointnet2_utils.py        # PointNet++ building blocks (for comparison)
│   ├── backbone.py               # PointNetBackbone (primary) & PointNet2Backbone
│   ├── assembly_relation.py      # Cross-Part Attention & Interface Proximity Analysis
│   └── risk_predictor.py         # Multi-task risk prediction heads + regularized loss
├── training/
│   ├── train.py                  # Training pipeline with multi-task + regularizer loss
│   ├── evaluate.py               # Evaluation & metrics (RMSE, F1, accuracy)
│   └── config.yaml               # Hyperparameters & PointNet settings
├── visualizer/
│   ├── app.py                    # Streamlit interactive dashboard with T-Net inspection
│   └── plot_utils.py             # 3D Plotly visualization utilities
├── experiments/
│   └── benchmark_comparison.py   # Baseline comparison (Random, Handcrafted MLP, Vanilla PointNet)
├── test_pipeline.py              # Verification & invariance smoke test
├── requirements.txt
└── README.md
```

---

## 🔧 Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify pipeline & mathematical invariance
python test_pipeline.py

# 3. Generate synthetic assembly dataset
python data/dataset_generator.py --num_samples 4000 --n_points 2048

# 4. Train the PointNet model
python training/train.py --config training/config.yaml

# 5. Evaluate on test set
python training/evaluate.py --checkpoint checkpoints/best_model.pth

# 6. Run benchmark comparisons
python experiments/benchmark_comparison.py

# 7. Launch interactive Streamlit dashboard
streamlit run visualizer/app.py
```

---

## 📊 Multi-Task Risk Prediction Heads

| Output Target | Type | Activation | Objective Loss |
|---------------|------|------------|----------------|
| **Interference Risk** | Continuous $[0, 1]$ | Sigmoid | Mean Squared Error (MSE) |
| **Clearance Risk** | Continuous $[0, 1]$ | Sigmoid | Mean Squared Error (MSE) |
| **Alignment Risk** | Continuous $[0, 1]$ | Sigmoid | Mean Squared Error (MSE) |
| **Failure Probability** | Continuous $[0, 1]$ | Sigmoid | Binary Cross-Entropy (BCE) |
| **Risk Level** | Categorical $\{0, 1, 2\}$ | Softmax | Cross-Entropy Loss |
| **Feature Transformation Regularizer** | Matrix constraint | Orthogonal penalty | $\lambda_{\text{reg}} \|I - A A^T\|_F^2$ |

---

## 🔬 Interactive 3D Streamlit Visualizer

The visualizer provides:
- **3D Point Cloud Rendering**: Color-coded by Part ID, Surface Normals, or Risk Heatmap.
- **Risk Gauges**: Real-time dials for Interference, Clearance, and Alignment risks.
- **Engineering Recommendations**: Actionable guidance (re-machining, shimming, thermal fitting, re-fixturing).
- **PointNet T-Net Diagnostic Card**: Visual inspection of the learned 3×3 spatial transformation matrix and 64×64 feature orthogonality penalty.
