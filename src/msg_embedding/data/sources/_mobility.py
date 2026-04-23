"""UE mobility models for trajectory-based channel simulation.

Generates spatially continuous UE trajectories with realistic motion
patterns.  Each model takes an initial position, speed, and number of
time steps, and returns a ``[num_steps, 3]`` array of XYZ coordinates.

Supported motion modes
----------------------
* **static** — UE stays in place (v=0).
* **linear** — constant speed, constant direction.
* **random_walk** — each step picks a random turning angle (Gaussian).
* **random_waypoint** — pick random destination → move at constant
  speed → optional pause → repeat.  Classic RWP from Camp et al. 2002.
* **hexagonal_boundary** — like random_walk, but reflects off hex cell
  boundary so the UE stays within the serving cell.

All models enforce an optional circular boundary constraint so UEs do
not leave the network coverage area.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MOBILITY_MODES = ("static", "linear", "random_walk", "random_waypoint")


def generate_trajectory(
    rng: np.random.Generator,
    start_pos: np.ndarray,
    speed_kmh: float,
    num_steps: int,
    dt_s: float = 0.5e-3,
    mode: str = "linear",
    boundary_radius_m: float | None = None,
    boundary_center: np.ndarray | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Generate a UE trajectory.

    Parameters
    ----------
    rng : numpy random Generator
    start_pos : [3] array — initial XYZ in metres
    speed_kmh : UE speed in km/h
    num_steps : number of time samples (= num_samples)
    dt_s : time interval between consecutive samples (seconds).
        Default 0.5 ms (one slot @ 30 kHz SCS).
    mode : one of MOBILITY_MODES
    boundary_radius_m : if set, UE is reflected back when it exceeds
        this distance from *boundary_center*.
    boundary_center : [2] or [3] array, default origin.

    Returns
    -------
    positions : ndarray, shape ``[num_steps, 3]``
    """
    if mode not in MOBILITY_MODES:
        raise ValueError(
            f"Unknown mobility mode {mode!r}; expected one of {MOBILITY_MODES}"
        )

    speed_ms = speed_kmh / 3.6
    start = np.asarray(start_pos, dtype=np.float64).copy()

    if boundary_center is not None:
        bc = np.asarray(boundary_center, dtype=np.float64)[:2]
    else:
        bc = np.zeros(2, dtype=np.float64)

    if mode == "static":
        positions = _static(start, num_steps)
    elif mode == "linear":
        positions = _linear(rng, start, speed_ms, num_steps, dt_s)
    elif mode == "random_walk":
        positions = _random_walk(
            rng, start, speed_ms, num_steps, dt_s,
            turn_std_deg=float(kwargs.get("turn_std_deg", 25.0)),
        )
    elif mode == "random_waypoint":
        positions = _random_waypoint(
            rng, start, speed_ms, num_steps, dt_s,
            area_radius_m=boundary_radius_m or 500.0,
            area_center=bc,
            pause_prob=float(kwargs.get("pause_prob", 0.0)),
            v_min_kmh=float(kwargs.get("v_min_kmh", speed_kmh * 0.3)),
            v_max_kmh=float(kwargs.get("v_max_kmh", speed_kmh * 1.5)),
        )
    else:
        positions = _static(start, num_steps)

    if boundary_radius_m is not None and boundary_radius_m > 0:
        _enforce_boundary(positions, bc, boundary_radius_m)

    return positions


def compute_instantaneous_speed(
    positions: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Compute instantaneous speed (m/s) from position trajectory.

    Returns shape ``[num_steps]``.  First element duplicates the second.
    """
    diffs = np.diff(positions[:, :2], axis=0)  # [N-1, 2]
    speeds = np.linalg.norm(diffs, axis=1) / dt_s  # [N-1]
    return np.concatenate([[speeds[0]], speeds]) if len(speeds) > 0 else np.zeros(1)


def compute_doppler_from_trajectory(
    positions: np.ndarray,
    bs_pos: np.ndarray,
    carrier_freq_hz: float,
    dt_s: float,
) -> np.ndarray:
    """Compute per-sample Doppler shift from UE trajectory relative to BS.

    Uses radial velocity: v_r = d(distance)/dt.
    Returns shape ``[num_steps]`` in Hz.
    """
    c = 3e8
    wavelength = c / carrier_freq_hz

    bs_2d = np.asarray(bs_pos[:2], dtype=np.float64)
    rel = positions[:, :2] - bs_2d[None, :]  # [N, 2]
    dist = np.linalg.norm(rel, axis=1)  # [N]
    dist = np.maximum(dist, 1e-3)

    radial_vel = np.diff(dist) / dt_s  # [N-1]  positive = moving away
    radial_vel = np.concatenate([[radial_vel[0]], radial_vel])

    doppler = -radial_vel / wavelength  # negative radial vel = approaching = positive Doppler
    return doppler


# ---------------------------------------------------------------------------
# Motion models
# ---------------------------------------------------------------------------

def _static(start: np.ndarray, num_steps: int) -> np.ndarray:
    return np.tile(start, (num_steps, 1))


def _linear(
    rng: np.random.Generator,
    start: np.ndarray,
    speed_ms: float,
    num_steps: int,
    dt_s: float,
) -> np.ndarray:
    heading = rng.uniform(0, 2 * np.pi)
    dx = speed_ms * dt_s * np.cos(heading)
    dy = speed_ms * dt_s * np.sin(heading)
    positions = np.zeros((num_steps, 3), dtype=np.float64)
    positions[0] = start
    for i in range(1, num_steps):
        positions[i, 0] = positions[i - 1, 0] + dx
        positions[i, 1] = positions[i - 1, 1] + dy
        positions[i, 2] = start[2]
    return positions


def _random_walk(
    rng: np.random.Generator,
    start: np.ndarray,
    speed_ms: float,
    num_steps: int,
    dt_s: float,
    turn_std_deg: float = 25.0,
) -> np.ndarray:
    """Correlated random walk: heading changes by Gaussian increment each step."""
    heading = rng.uniform(0, 2 * np.pi)
    turn_std_rad = np.deg2rad(turn_std_deg)
    step_len = speed_ms * dt_s

    positions = np.zeros((num_steps, 3), dtype=np.float64)
    positions[0] = start
    for i in range(1, num_steps):
        heading += rng.normal(0, turn_std_rad)
        positions[i, 0] = positions[i - 1, 0] + step_len * np.cos(heading)
        positions[i, 1] = positions[i - 1, 1] + step_len * np.sin(heading)
        positions[i, 2] = start[2]
    return positions


def _random_waypoint(
    rng: np.random.Generator,
    start: np.ndarray,
    speed_ms: float,
    num_steps: int,
    dt_s: float,
    area_radius_m: float,
    area_center: np.ndarray,
    pause_prob: float = 0.0,
    v_min_kmh: float = 1.0,
    v_max_kmh: float = 120.0,
) -> np.ndarray:
    """Random Waypoint model (Camp et al., 2002).

    1. Pick a random destination uniformly in the circle.
    2. Pick a random speed in [v_min, v_max].
    3. Move in a straight line at that speed.
    4. On arrival, optionally pause (with probability pause_prob).
    5. Repeat.
    """
    positions = np.zeros((num_steps, 3), dtype=np.float64)
    positions[0] = start
    cur = start[:2].copy()
    z = start[2]

    v_min = v_min_kmh / 3.6
    v_max = max(v_max_kmh / 3.6, v_min + 0.1)

    # Pick first waypoint
    wp = _random_point_in_circle(rng, area_center, area_radius_m)
    seg_speed = rng.uniform(v_min, v_max)
    paused = False

    for i in range(1, num_steps):
        if paused:
            positions[i] = [cur[0], cur[1], z]
            paused = False
            wp = _random_point_in_circle(rng, area_center, area_radius_m)
            seg_speed = rng.uniform(v_min, v_max)
            continue

        direction = wp - cur
        dist_to_wp = np.linalg.norm(direction)
        step_len = seg_speed * dt_s

        if dist_to_wp <= step_len:
            cur = wp.copy()
            positions[i] = [cur[0], cur[1], z]
            if rng.random() < pause_prob:
                paused = True
            else:
                wp = _random_point_in_circle(rng, area_center, area_radius_m)
                seg_speed = rng.uniform(v_min, v_max)
        else:
            unit = direction / dist_to_wp
            cur = cur + unit * step_len
            positions[i] = [cur[0], cur[1], z]

    return positions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_point_in_circle(
    rng: np.random.Generator,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    r = radius * np.sqrt(rng.uniform())
    theta = rng.uniform(0, 2 * np.pi)
    return np.array([center[0] + r * np.cos(theta), center[1] + r * np.sin(theta)])


def _enforce_boundary(
    positions: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> None:
    """In-place reflection: if UE exits circle, reflect it back."""
    for i in range(1, len(positions)):
        rel = positions[i, :2] - center
        d = np.linalg.norm(rel)
        if d > radius:
            overshoot = d - radius
            unit = rel / d
            positions[i, :2] = center + unit * (radius - overshoot)
            d2 = np.linalg.norm(positions[i, :2] - center)
            if d2 > radius:
                positions[i, :2] = center + unit * radius * 0.99
