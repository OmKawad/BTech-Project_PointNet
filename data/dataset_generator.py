"""
Synthetic CAD Assembly Point Cloud Generator
=============================================
Generates parametric mechanical assemblies with controlled tolerance defects
for training the Assembly Fit Risk Prediction model.

Assembly Archetypes:
  1. Shaft & Hole (cylindrical fit)
  2. Flange Joint (bolt-circle mating)
  3. Peg-in-Hole (prismatic insertion)
  4. Dovetail Joint (angular sliding fit)

Each assembly is labelled with:
  - interference_risk  ∈ [0, 1]
  - clearance_risk     ∈ [0, 1]
  - alignment_risk     ∈ [0, 1]
  - failure_prob        ∈ [0, 1]   (weighted combination)
  - risk_level          ∈ {0: Low, 1: Medium, 2: High}
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# ──────────────────────────────────────────────
# Helper geometry primitives
# ──────────────────────────────────────────────

def _sample_cylinder_surface(radius: float, height: float,
                             n_points: int, inner: bool = False) -> np.ndarray:
    """Sample points on the surface of a cylinder (inner or outer wall)."""
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    z = np.random.uniform(-height / 2, height / 2, n_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    points = np.stack([x, y, z], axis=-1)
    # Normals: radial direction (outward for outer, inward for inner)
    nx = np.cos(theta)
    ny = np.sin(theta)
    nz = np.zeros(n_points)
    normals = np.stack([nx, ny, nz], axis=-1)
    if inner:
        normals = -normals
    return points, normals


def _sample_disk(radius_inner: float, radius_outer: float,
                 z_pos: float, n_points: int, normal_dir: float = 1.0) -> np.ndarray:
    """Sample points on an annular disk at a given z position."""
    r = np.sqrt(np.random.uniform(radius_inner**2, radius_outer**2, n_points))
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.full(n_points, z_pos)
    points = np.stack([x, y, z], axis=-1)
    normals = np.tile(np.array([0.0, 0.0, normal_dir]), (n_points, 1))
    return points, normals


def _sample_box_surface(lx: float, ly: float, lz: float,
                        n_points: int, offset: np.ndarray = None) -> np.ndarray:
    """Sample points uniformly on the surface of a box."""
    if offset is None:
        offset = np.zeros(3)
    # Surface areas for each face pair
    areas = np.array([ly * lz, ly * lz, lx * lz, lx * lz, lx * ly, lx * ly])
    probs = areas / areas.sum()
    face_indices = np.random.choice(6, size=n_points, p=probs)

    points = np.zeros((n_points, 3))
    normals = np.zeros((n_points, 3))
    for face_id in range(6):
        mask = face_indices == face_id
        count = mask.sum()
        if count == 0:
            continue
        if face_id == 0:  # +x face
            points[mask] = np.stack([np.full(count, lx / 2),
                                     np.random.uniform(-ly / 2, ly / 2, count),
                                     np.random.uniform(-lz / 2, lz / 2, count)], axis=-1)
            normals[mask] = [1, 0, 0]
        elif face_id == 1:  # -x face
            points[mask] = np.stack([np.full(count, -lx / 2),
                                     np.random.uniform(-ly / 2, ly / 2, count),
                                     np.random.uniform(-lz / 2, lz / 2, count)], axis=-1)
            normals[mask] = [-1, 0, 0]
        elif face_id == 2:  # +y face
            points[mask] = np.stack([np.random.uniform(-lx / 2, lx / 2, count),
                                     np.full(count, ly / 2),
                                     np.random.uniform(-lz / 2, lz / 2, count)], axis=-1)
            normals[mask] = [0, 1, 0]
        elif face_id == 3:  # -y face
            points[mask] = np.stack([np.random.uniform(-lx / 2, lx / 2, count),
                                     np.full(count, -ly / 2),
                                     np.random.uniform(-lz / 2, lz / 2, count)], axis=-1)
            normals[mask] = [0, -1, 0]
        elif face_id == 4:  # +z face
            points[mask] = np.stack([np.random.uniform(-lx / 2, lx / 2, count),
                                     np.random.uniform(-ly / 2, ly / 2, count),
                                     np.full(count, lz / 2)], axis=-1)
            normals[mask] = [0, 0, 1]
        else:  # -z face
            points[mask] = np.stack([np.random.uniform(-lx / 2, lx / 2, count),
                                     np.random.uniform(-ly / 2, ly / 2, count),
                                     np.full(count, -lz / 2)], axis=-1)
            normals[mask] = [0, 0, -1]

    points += offset
    return points, normals


def _apply_misalignment(points: np.ndarray, normals: np.ndarray,
                         tilt_deg: float = 0.0, offset_xy: np.ndarray = None) -> Tuple:
    """Apply angular tilt and radial offset to simulate alignment defects."""
    if offset_xy is not None:
        points[:, :2] += offset_xy

    if abs(tilt_deg) > 1e-6:
        angle = np.radians(tilt_deg)
        # Tilt around x-axis
        rot = np.array([[1, 0, 0],
                        [0, np.cos(angle), -np.sin(angle)],
                        [0, np.sin(angle), np.cos(angle)]])
        points = points @ rot.T
        normals = normals @ rot.T
        # Re-normalize
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / np.clip(norms, 1e-8, None)

    return points, normals


# ──────────────────────────────────────────────
# Assembly generators
# ──────────────────────────────────────────────

@dataclass
class AssemblyConfig:
    """Parametric configuration for an assembly sample."""
    assembly_type: str = "shaft_hole"
    # Geometric parameters (randomised per sample)
    nominal_radius: float = 1.0
    height: float = 2.0
    # Tolerance defect magnitudes
    radial_deviation: float = 0.0      # +ve = interference, -ve = excess clearance
    tilt_angle: float = 0.0            # degrees
    lateral_offset: float = 0.0        # radial eccentricity
    # Noise
    point_noise_std: float = 0.002
    normal_noise_std: float = 0.01


def _compute_risk_labels(cfg: AssemblyConfig) -> Dict[str, float]:
    """
    Compute ground-truth risk labels from assembly configuration parameters.

    Interference risk: How much the shaft exceeds the hole bore.
    Clearance risk: How much excess gap exists beyond nominal tolerance.
    Alignment risk: How misaligned the axis is (tilt + eccentricity).
    """
    nominal = cfg.nominal_radius

    # Interference risk: positive radial deviation means shaft is larger than hole
    interference = max(0.0, cfg.radial_deviation / (0.1 * nominal))
    interference = min(1.0, interference)

    # Clearance risk: negative deviation means excessive gap
    clearance = max(0.0, -cfg.radial_deviation / (0.1 * nominal))
    clearance = min(1.0, clearance)

    # Alignment risk: combination of tilt and lateral offset
    tilt_component = min(1.0, abs(cfg.tilt_angle) / 5.0)      # 5° is max concern
    offset_component = min(1.0, abs(cfg.lateral_offset) / (0.05 * nominal))
    alignment = min(1.0, 0.6 * tilt_component + 0.4 * offset_component)

    # Overall failure probability (weighted combination)
    failure_prob = min(1.0, 0.35 * interference + 0.25 * clearance + 0.40 * alignment)

    # Categorical risk level
    if failure_prob < 0.33:
        risk_level = 0  # Low
    elif failure_prob < 0.66:
        risk_level = 1  # Medium
    else:
        risk_level = 2  # High

    return {
        "interference_risk": interference,
        "clearance_risk": clearance,
        "alignment_risk": alignment,
        "failure_prob": failure_prob,
        "risk_level": risk_level,
    }


def generate_shaft_hole(cfg: AssemblyConfig, n_points: int) -> Dict:
    """Generate a cylindrical shaft-in-hole assembly point cloud."""
    n_half = n_points // 2
    shaft_radius = cfg.nominal_radius + cfg.radial_deviation
    hole_radius = cfg.nominal_radius

    # Part 0: Shaft (outer cylinder)
    shaft_pts, shaft_nrm = _sample_cylinder_surface(
        shaft_radius, cfg.height, n_half, inner=False)
    # Add top/bottom caps
    cap_pts = int(n_half * 0.1)
    body_pts = n_half - 2 * cap_pts
    shaft_pts, shaft_nrm = _sample_cylinder_surface(
        shaft_radius, cfg.height, body_pts, inner=False)
    top_pts, top_nrm = _sample_disk(0, shaft_radius, cfg.height / 2, cap_pts, 1.0)
    bot_pts, bot_nrm = _sample_disk(0, shaft_radius, -cfg.height / 2, cap_pts, -1.0)
    shaft_pts = np.concatenate([shaft_pts, top_pts, bot_pts], axis=0)
    shaft_nrm = np.concatenate([shaft_nrm, top_nrm, bot_nrm], axis=0)

    # Apply misalignment to shaft
    offset_xy = np.array([cfg.lateral_offset, 0.0]) if cfg.lateral_offset else None
    shaft_pts, shaft_nrm = _apply_misalignment(
        shaft_pts, shaft_nrm, cfg.tilt_angle, offset_xy)

    # Part 1: Hole (inner cylinder wall of housing block)
    housing_outer = hole_radius * 2.0
    n_hole_wall = int(n_half * 0.6)
    n_housing = n_half - n_hole_wall
    hole_pts, hole_nrm = _sample_cylinder_surface(
        hole_radius, cfg.height, n_hole_wall, inner=True)
    housing_pts, housing_nrm = _sample_box_surface(
        housing_outer * 2, housing_outer * 2, cfg.height, n_housing)

    hole_pts = np.concatenate([hole_pts, housing_pts], axis=0)
    hole_nrm = np.concatenate([hole_nrm, housing_nrm], axis=0)

    # Combine with part_id
    all_pts = np.concatenate([shaft_pts, hole_pts], axis=0)
    all_nrm = np.concatenate([shaft_nrm, hole_nrm], axis=0)
    part_ids = np.concatenate([np.zeros(len(shaft_pts)),
                                np.ones(len(hole_pts))]).astype(np.float32)

    return {"points": all_pts, "normals": all_nrm, "part_ids": part_ids}


def generate_flange_joint(cfg: AssemblyConfig, n_points: int) -> Dict:
    """Generate a flange-to-flange bolted joint assembly."""
    n_half = n_points // 2
    flange_radius = cfg.nominal_radius * 1.5
    bolt_circle_radius = cfg.nominal_radius * 1.1
    flange_thickness = cfg.height * 0.15

    # Part 0: Top flange (disk with bolt holes implied by surface)
    top_pts, top_nrm = _sample_disk(0, flange_radius, flange_thickness / 2,
                                     int(n_half * 0.5), 1.0)
    bot_pts, bot_nrm = _sample_disk(0, flange_radius, -flange_thickness / 2,
                                     int(n_half * 0.3), -1.0)
    rim_pts, rim_nrm = _sample_cylinder_surface(
        flange_radius, flange_thickness, n_half - int(n_half * 0.5) - int(n_half * 0.3))
    flange_a = np.concatenate([top_pts, bot_pts, rim_pts], axis=0)
    nrm_a = np.concatenate([top_nrm, bot_nrm, rim_nrm], axis=0)

    # Shift flange A up
    flange_a[:, 2] += flange_thickness / 2 + cfg.radial_deviation * 0.5

    # Part 1: Bottom flange (mirror)
    top_pts2, top_nrm2 = _sample_disk(0, flange_radius, flange_thickness / 2,
                                       int(n_half * 0.5), 1.0)
    bot_pts2, bot_nrm2 = _sample_disk(0, flange_radius, -flange_thickness / 2,
                                       int(n_half * 0.3), -1.0)
    rim_pts2, rim_nrm2 = _sample_cylinder_surface(
        flange_radius, flange_thickness, n_half - int(n_half * 0.5) - int(n_half * 0.3))
    flange_b = np.concatenate([top_pts2, bot_pts2, rim_pts2], axis=0)
    nrm_b = np.concatenate([top_nrm2, bot_nrm2, rim_nrm2], axis=0)

    # Shift flange B down
    flange_b[:, 2] -= flange_thickness / 2

    # Apply misalignment to top flange
    offset_xy = np.array([cfg.lateral_offset, 0.0]) if cfg.lateral_offset else None
    flange_a, nrm_a = _apply_misalignment(flange_a, nrm_a, cfg.tilt_angle, offset_xy)

    all_pts = np.concatenate([flange_a, flange_b], axis=0)
    all_nrm = np.concatenate([nrm_a, nrm_b], axis=0)
    part_ids = np.concatenate([np.zeros(len(flange_a)),
                                np.ones(len(flange_b))]).astype(np.float32)

    return {"points": all_pts, "normals": all_nrm, "part_ids": part_ids}


def generate_peg_in_hole(cfg: AssemblyConfig, n_points: int) -> Dict:
    """Generate a rectangular peg-in-hole (prismatic) assembly."""
    n_half = n_points // 2
    peg_w = cfg.nominal_radius * 0.8
    peg_h = cfg.nominal_radius * 0.8
    peg_l = cfg.height
    slot_w = peg_w + 0.02 - cfg.radial_deviation
    slot_h = peg_h + 0.02 - cfg.radial_deviation

    # Part 0: Peg (rectangular prism)
    peg_pts, peg_nrm = _sample_box_surface(peg_w, peg_h, peg_l, n_half)
    offset_xy = np.array([cfg.lateral_offset, 0.0]) if cfg.lateral_offset else None
    peg_pts, peg_nrm = _apply_misalignment(peg_pts, peg_nrm, cfg.tilt_angle, offset_xy)

    # Part 1: Slot (hollow rectangular block – outer box minus inner cavity)
    outer_pts, outer_nrm = _sample_box_surface(
        slot_w * 3, slot_h * 3, peg_l, int(n_half * 0.5))
    # Inner walls of the slot (4 planes)
    inner_count = n_half - int(n_half * 0.5)
    per_wall = inner_count // 4
    walls = []
    wall_normals = []
    # +x wall
    w_pts = np.stack([np.full(per_wall, slot_w / 2),
                      np.random.uniform(-slot_h / 2, slot_h / 2, per_wall),
                      np.random.uniform(-peg_l / 2, peg_l / 2, per_wall)], axis=-1)
    walls.append(w_pts)
    wall_normals.append(np.tile([-1, 0, 0], (per_wall, 1)))
    # -x wall
    w_pts = np.stack([np.full(per_wall, -slot_w / 2),
                      np.random.uniform(-slot_h / 2, slot_h / 2, per_wall),
                      np.random.uniform(-peg_l / 2, peg_l / 2, per_wall)], axis=-1)
    walls.append(w_pts)
    wall_normals.append(np.tile([1, 0, 0], (per_wall, 1)))
    # +y wall
    w_pts = np.stack([np.random.uniform(-slot_w / 2, slot_w / 2, per_wall),
                      np.full(per_wall, slot_h / 2),
                      np.random.uniform(-peg_l / 2, peg_l / 2, per_wall)], axis=-1)
    walls.append(w_pts)
    wall_normals.append(np.tile([0, -1, 0], (per_wall, 1)))
    # -y wall
    remainder = inner_count - 3 * per_wall
    w_pts = np.stack([np.random.uniform(-slot_w / 2, slot_w / 2, remainder),
                      np.full(remainder, -slot_h / 2),
                      np.random.uniform(-peg_l / 2, peg_l / 2, remainder)], axis=-1)
    walls.append(w_pts)
    wall_normals.append(np.tile([0, 1, 0], (remainder, 1)))

    slot_pts = np.concatenate([outer_pts] + walls, axis=0)
    slot_nrm = np.concatenate([outer_nrm] + wall_normals, axis=0)

    all_pts = np.concatenate([peg_pts, slot_pts], axis=0)
    all_nrm = np.concatenate([peg_nrm, slot_nrm], axis=0)
    part_ids = np.concatenate([np.zeros(len(peg_pts)),
                                np.ones(len(slot_pts))]).astype(np.float32)

    return {"points": all_pts, "normals": all_nrm, "part_ids": part_ids}


def generate_dovetail(cfg: AssemblyConfig, n_points: int) -> Dict:
    """Generate a dovetail (trapezoidal tongue-and-groove) assembly."""
    n_half = n_points // 2
    base_w = cfg.nominal_radius
    top_w = base_w * 0.7  # Narrower top for dovetail angle
    depth = cfg.height * 0.4
    length = cfg.height

    # Part 0: Dovetail tongue (trapezoidal prism – approximated with box)
    tongue_pts, tongue_nrm = _sample_box_surface(
        (base_w + top_w) / 2 + cfg.radial_deviation,
        depth, length, n_half)
    offset_xy = np.array([cfg.lateral_offset, 0.0]) if cfg.lateral_offset else None
    tongue_pts, tongue_nrm = _apply_misalignment(
        tongue_pts, tongue_nrm, cfg.tilt_angle, offset_xy)

    # Part 1: Groove (cavity in a larger block)
    groove_w = (base_w + top_w) / 2 + 0.02
    block_pts, block_nrm = _sample_box_surface(
        base_w * 3, depth * 3, length, int(n_half * 0.5))
    # Inner groove walls
    inner_count = n_half - int(n_half * 0.5)
    per_wall = inner_count // 3
    walls = []
    wall_normals = []
    # Bottom of groove
    w_pts = np.stack([np.random.uniform(-groove_w / 2, groove_w / 2, per_wall),
                      np.full(per_wall, -depth / 2),
                      np.random.uniform(-length / 2, length / 2, per_wall)], axis=-1)
    walls.append(w_pts)
    wall_normals.append(np.tile([0, 1, 0], (per_wall, 1)))
    # Left wall
    w_pts = np.stack([np.full(per_wall, -groove_w / 2),
                      np.random.uniform(-depth / 2, depth / 2, per_wall),
                      np.random.uniform(-length / 2, length / 2, per_wall)], axis=-1)
    walls.append(w_pts)
    wall_normals.append(np.tile([1, 0, 0], (per_wall, 1)))
    # Right wall
    remainder = inner_count - 2 * per_wall
    w_pts = np.stack([np.full(remainder, groove_w / 2),
                      np.random.uniform(-depth / 2, depth / 2, remainder),
                      np.random.uniform(-length / 2, length / 2, remainder)], axis=-1)
    walls.append(w_pts)
    wall_normals.append(np.tile([-1, 0, 0], (remainder, 1)))

    groove_pts = np.concatenate([block_pts] + walls, axis=0)
    groove_nrm = np.concatenate([block_nrm] + wall_normals, axis=0)

    all_pts = np.concatenate([tongue_pts, groove_pts], axis=0)
    all_nrm = np.concatenate([tongue_nrm, groove_nrm], axis=0)
    part_ids = np.concatenate([np.zeros(len(tongue_pts)),
                                np.ones(len(groove_pts))]).astype(np.float32)

    return {"points": all_pts, "normals": all_nrm, "part_ids": part_ids}


# ──────────────────────────────────────────────
# Main dataset generation driver
# ──────────────────────────────────────────────

GENERATOR_MAP = {
    "shaft_hole": generate_shaft_hole,
    "flange_joint": generate_flange_joint,
    "peg_in_hole": generate_peg_in_hole,
    "dovetail": generate_dovetail,
}


def _random_assembly_config(assembly_type: str, difficulty: str = "mixed") -> AssemblyConfig:
    """Create a random parametric configuration for a given assembly type."""
    rng = np.random.default_rng()
    nominal_radius = rng.uniform(0.5, 2.0)
    height = rng.uniform(1.0, 4.0)

    if difficulty == "low":
        radial_dev = rng.uniform(-0.005, 0.005) * nominal_radius
        tilt = rng.uniform(0, 0.5)
        offset = rng.uniform(0, 0.005) * nominal_radius
    elif difficulty == "high":
        radial_dev = rng.choice([-1, 1]) * rng.uniform(0.05, 0.12) * nominal_radius
        tilt = rng.uniform(3.0, 8.0)
        offset = rng.uniform(0.03, 0.08) * nominal_radius
    else:  # mixed
        radial_dev = rng.choice([-1, 1]) * rng.uniform(0.0, 0.12) * nominal_radius
        tilt = rng.uniform(0, 8.0)
        offset = rng.uniform(0, 0.08) * nominal_radius

    return AssemblyConfig(
        assembly_type=assembly_type,
        nominal_radius=nominal_radius,
        height=height,
        radial_deviation=radial_dev,
        tilt_angle=tilt,
        lateral_offset=offset,
    )


def generate_dataset(num_samples: int, n_points: int = 2048,
                     assembly_types: Optional[List[str]] = None,
                     save_dir: Optional[str] = None,
                     noise_std: float = 0.002,
                     normal_noise_std: float = 0.01,
                     seed: int = 42) -> List[Dict]:
    """
    Generate a full dataset of synthetic assembly point clouds.

    Returns a list of dicts, each containing:
        - 'point_cloud': np.ndarray of shape (N, 7)  [x, y, z, nx, ny, nz, part_id]
        - 'labels': dict with risk values and risk_level
        - 'assembly_type': str
    """
    np.random.seed(seed)
    if assembly_types is None:
        assembly_types = list(GENERATOR_MAP.keys())

    dataset = []
    for i in range(num_samples):
        atype = assembly_types[i % len(assembly_types)]
        cfg = _random_assembly_config(atype)
        cfg.point_noise_std = noise_std
        cfg.normal_noise_std = normal_noise_std

        generator_fn = GENERATOR_MAP[atype]
        result = generator_fn(cfg, n_points)

        pts = result["points"].astype(np.float32)
        nrm = result["normals"].astype(np.float32)
        pids = result["part_ids"].astype(np.float32)

        # Add scanner noise
        pts += np.random.normal(0, noise_std, pts.shape).astype(np.float32)
        nrm += np.random.normal(0, normal_noise_std, nrm.shape).astype(np.float32)
        # Re-normalize normals
        norms = np.linalg.norm(nrm, axis=-1, keepdims=True)
        nrm = nrm / np.clip(norms, 1e-8, None)

        # Resample to exactly n_points if needed
        current_n = len(pts)
        if current_n > n_points:
            indices = np.random.choice(current_n, n_points, replace=False)
        elif current_n < n_points:
            indices = np.random.choice(current_n, n_points, replace=True)
        else:
            indices = np.arange(n_points)

        pts = pts[indices]
        nrm = nrm[indices]
        pids = pids[indices]

        # Assemble 7D point cloud
        point_cloud = np.concatenate([pts, nrm, pids[:, None]], axis=-1)  # (N, 7)

        labels = _compute_risk_labels(cfg)

        sample = {
            "point_cloud": point_cloud,
            "labels": labels,
            "assembly_type": atype,
            "config": {
                "nominal_radius": cfg.nominal_radius,
                "height": cfg.height,
                "radial_deviation": cfg.radial_deviation,
                "tilt_angle": cfg.tilt_angle,
                "lateral_offset": cfg.lateral_offset,
            },
        }
        dataset.append(sample)

    # Save to disk if requested
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for i, sample in enumerate(dataset):
            np.save(os.path.join(save_dir, f"assembly_{i:05d}.npy"), sample["point_cloud"])

        # Save labels as a separate file
        import json
        labels_list = []
        for i, sample in enumerate(dataset):
            entry = {"index": i, "assembly_type": sample["assembly_type"]}
            entry.update(sample["labels"])
            entry.update(sample["config"])
            labels_list.append(entry)
        with open(os.path.join(save_dir, "labels.json"), "w") as f:
            json.dump(labels_list, f, indent=2)

        print(f"[DatasetGenerator] Saved {len(dataset)} samples to {save_dir}")

    return dataset


# ──────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic assembly dataset")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--n_points", type=int, default=2048)
    parser.add_argument("--save_dir", type=str, default="data/samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(
        num_samples=args.num_samples,
        n_points=args.n_points,
        save_dir=args.save_dir,
        seed=args.seed,
    )
