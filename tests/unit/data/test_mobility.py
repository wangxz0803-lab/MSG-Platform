"""Tests for the UE mobility trajectory generation module."""

from __future__ import annotations

import numpy as np
import pytest

from msg_embedding.data.sources._mobility import (
    MOBILITY_MODES,
    compute_doppler_from_trajectory,
    compute_instantaneous_speed,
    generate_trajectory,
)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture()
def start_pos() -> np.ndarray:
    return np.array([100.0, 200.0, 1.5])


class TestGenerateTrajectory:
    def test_static_returns_constant_positions(self, rng, start_pos):
        pos = generate_trajectory(rng, start_pos, speed_kmh=30, num_steps=50, mode="static")
        assert pos.shape == (50, 3)
        np.testing.assert_array_equal(pos[0], pos[-1])

    def test_linear_moves_in_straight_line(self, rng, start_pos):
        pos = generate_trajectory(rng, start_pos, speed_kmh=60, num_steps=100, dt_s=0.01, mode="linear")
        assert pos.shape == (100, 3)
        diffs = np.diff(pos[:, :2], axis=0)
        directions = np.arctan2(diffs[:, 1], diffs[:, 0])
        np.testing.assert_allclose(directions, directions[0], atol=1e-10)

    def test_random_walk_has_varying_direction(self, rng, start_pos):
        pos = generate_trajectory(rng, start_pos, speed_kmh=30, num_steps=200, dt_s=0.01, mode="random_walk")
        assert pos.shape == (200, 3)
        diffs = np.diff(pos[:, :2], axis=0)
        directions = np.arctan2(diffs[:, 1], diffs[:, 0])
        assert np.std(directions) > 0.01

    def test_random_waypoint_stays_in_area(self, rng, start_pos):
        pos = generate_trajectory(
            rng, np.array([0.0, 0.0, 1.5]), speed_kmh=60, num_steps=500,
            dt_s=0.01, mode="random_waypoint", boundary_radius_m=300.0,
        )
        dists = np.linalg.norm(pos[:, :2], axis=1)
        assert dists.max() <= 300.0 + 1.0

    def test_boundary_reflection(self, rng):
        pos = generate_trajectory(
            rng, np.array([490.0, 0.0, 1.5]), speed_kmh=100, num_steps=1000,
            dt_s=0.01, mode="linear", boundary_radius_m=500.0,
        )
        dists = np.linalg.norm(pos[:, :2], axis=1)
        assert dists.max() <= 500.0 + 1.0

    def test_height_preserved(self, rng, start_pos):
        for mode in MOBILITY_MODES:
            pos = generate_trajectory(rng, start_pos, speed_kmh=30, num_steps=50, mode=mode)
            np.testing.assert_array_equal(pos[:, 2], start_pos[2])

    def test_invalid_mode_raises(self, rng, start_pos):
        with pytest.raises(ValueError, match="Unknown mobility mode"):
            generate_trajectory(rng, start_pos, speed_kmh=30, num_steps=10, mode="teleport")

    def test_all_modes_produce_correct_shape(self, rng, start_pos):
        for mode in MOBILITY_MODES:
            pos = generate_trajectory(rng, start_pos, speed_kmh=30, num_steps=20, mode=mode)
            assert pos.shape == (20, 3), f"mode={mode}"


class TestComputeInstantaneousSpeed:
    def test_static_has_zero_speed(self, rng, start_pos):
        pos = generate_trajectory(rng, start_pos, speed_kmh=0, num_steps=50, mode="static")
        speeds = compute_instantaneous_speed(pos, dt_s=0.01)
        np.testing.assert_allclose(speeds, 0.0, atol=1e-10)

    def test_linear_has_constant_speed(self, rng, start_pos):
        pos = generate_trajectory(rng, start_pos, speed_kmh=36, num_steps=100, dt_s=0.01, mode="linear")
        speeds = compute_instantaneous_speed(pos, dt_s=0.01)
        np.testing.assert_allclose(speeds, 10.0, atol=0.01)


class TestComputeDoppler:
    def test_approaching_bs_gives_positive_doppler(self):
        pos = np.zeros((100, 3))
        pos[:, 0] = np.linspace(500, 400, 100)
        pos[:, 2] = 1.5
        bs_pos = np.array([0.0, 0.0, 25.0])
        doppler = compute_doppler_from_trajectory(pos, bs_pos, 3.5e9, dt_s=0.01)
        assert doppler.mean() > 0

    def test_receding_bs_gives_negative_doppler(self):
        pos = np.zeros((100, 3))
        pos[:, 0] = np.linspace(100, 200, 100)
        pos[:, 2] = 1.5
        bs_pos = np.array([0.0, 0.0, 25.0])
        doppler = compute_doppler_from_trajectory(pos, bs_pos, 3.5e9, dt_s=0.01)
        assert doppler.mean() < 0

    def test_static_has_near_zero_doppler(self, rng, start_pos):
        pos = generate_trajectory(rng, start_pos, speed_kmh=0, num_steps=50, mode="static")
        bs_pos = np.array([0.0, 0.0, 25.0])
        doppler = compute_doppler_from_trajectory(pos, bs_pos, 3.5e9, dt_s=0.01)
        np.testing.assert_allclose(doppler, 0.0, atol=1e-5)
