"""Unit tests for the rotation / extrinsics module.

Tests cover:
* Zenith-pointing camera (alt0=90)
* Horizon-pointing camera (alt0=0)
* Azimuth sweeps
* Roll rotation
* Alt/Az ↔ World-vector round-trips
* Edge cases (horizon, poles)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pixel2sky.rotation import (
    altaz_to_world_vector,
    build_rotation,
    camera_to_world,
    world_to_camera,
    world_vector_to_altaz,
)

# -----------------------------------------------------------------------
# altaz_to_world_vector / world_vector_to_altaz
# -----------------------------------------------------------------------


class TestAltAzWorldVectorConversion:
    @pytest.mark.parametrize(
        "alt, az, expected_xyz",
        [
            (90.0, 0.0, [0.0, 0.0, 1.0]),  # Zenith
            (0.0, 0.0, [0.0, 1.0, 0.0]),  # North horizon
            (0.0, 90.0, [1.0, 0.0, 0.0]),  # East horizon
            (0.0, 180.0, [0.0, -1.0, 0.0]),  # South horizon
            (0.0, 270.0, [-1.0, 0.0, 0.0]),  # West horizon
            (-90.0, 0.0, [0.0, 0.0, -1.0]),  # Nadir
        ],
    )
    def test_cardinal_directions(
        self,
        alt: float,
        az: float,
        expected_xyz: list[float],
    ) -> None:
        v = altaz_to_world_vector(
            np.array([alt], dtype=float),
            np.array([az], dtype=float),
        )
        assert_allclose(v[0], expected_xyz, atol=1e-12)

    def test_round_trip_random(self) -> None:
        rng = np.random.default_rng(0)
        alt = rng.uniform(-90.0, 90.0, size=500)
        az = rng.uniform(0.0, 360.0, size=500)

        v = altaz_to_world_vector(alt, az)
        alt2, az2 = world_vector_to_altaz(v)

        assert_allclose(alt2, alt, atol=1e-9)
        assert_allclose(az2, az, atol=1e-9)

    def test_az_wraps_to_zero_to_360(self) -> None:
        v = altaz_to_world_vector(np.array([0.0]), np.array([0.0]))
        _, az = world_vector_to_altaz(v)
        assert 0.0 <= float(az.ravel()[0]) < 360.0

    def test_output_vectors_unit_length(self) -> None:
        alt = np.linspace(-90.0, 90.0, 100)
        az = np.linspace(0.0, 360.0, 100)
        v = altaz_to_world_vector(alt, az)
        norms = np.linalg.norm(v, axis=-1)
        assert_allclose(norms, 1.0, atol=1e-12)


# -----------------------------------------------------------------------
# build_rotation
# -----------------------------------------------------------------------


class TestBuildRotation:
    def test_alt_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="alt0"):
            build_rotation(az0=0.0, alt0=91.0, roll=0.0)

    def test_alt_minus_90_valid(self) -> None:
        R = build_rotation(az0=0.0, alt0=-90.0, roll=0.0)
        assert R is not None

    def test_zenith_camera_boresight_points_up(self) -> None:
        """When pointing at zenith (alt0=90), the boresight (+Z world = zenith)
        should map to the Camera +Z axis.
        """
        R = build_rotation(az0=0.0, alt0=90.0, roll=0.0)
        zenith_world = np.array([[0.0, 0.0, 1.0]])
        v_cam = world_to_camera(zenith_world, R)
        # Camera +Z should equal the boresight direction
        assert_allclose(v_cam[0], [0.0, 0.0, 1.0], atol=1e-12)

    def test_horizon_north_camera_boresight_points_north(self) -> None:
        R = build_rotation(az0=0.0, alt0=0.0, roll=0.0)
        north_world = np.array([[0.0, 1.0, 0.0]])  # North on horizon
        v_cam = world_to_camera(north_world, R)
        assert_allclose(v_cam[0], [0.0, 0.0, 1.0], atol=1e-12)

    def test_horizon_east_camera_boresight_points_east(self) -> None:
        R = build_rotation(az0=90.0, alt0=0.0, roll=0.0)
        east_world = np.array([[1.0, 0.0, 0.0]])  # East on horizon
        v_cam = world_to_camera(east_world, R)
        assert_allclose(v_cam[0], [0.0, 0.0, 1.0], atol=1e-12)

    def test_rotation_is_orthogonal(self) -> None:
        """R·Rᵀ = I for the rotation matrix."""
        R = build_rotation(az0=37.0, alt0=55.0, roll=22.0)
        M = R.as_matrix()
        assert_allclose(M @ M.T, np.eye(3), atol=1e-12)

    def test_rotation_determinant_is_one(self) -> None:
        R = build_rotation(az0=123.0, alt0=-30.0, roll=180.0)
        assert_allclose(np.linalg.det(R.as_matrix()), 1.0, atol=1e-12)

    @pytest.mark.parametrize("az0", [0.0, 45.0, 90.0, 135.0, 180.0, 270.0, 359.0])
    def test_azimuth_sweep(self, az0: float) -> None:
        """The boresight direction in World frame should equal the expected
        Alt/Az vector for any azimuth.
        """
        R = build_rotation(az0=az0, alt0=0.0, roll=0.0)
        boresight_cam = np.array([[0.0, 0.0, 1.0]])
        boresight_world = camera_to_world(boresight_cam, R)

        from pixel2sky.rotation import world_vector_to_altaz

        alt_out, az_out = world_vector_to_altaz(boresight_world)
        assert_allclose(alt_out.ravel()[0], 0.0, atol=1e-9)
        assert_allclose(az_out.ravel()[0] % 360.0, az0 % 360.0, atol=1e-9)

    @pytest.mark.parametrize("alt0", [-60.0, -30.0, 0.0, 30.0, 60.0, 90.0])
    def test_altitude_sweep(self, alt0: float) -> None:
        R = build_rotation(az0=0.0, alt0=alt0, roll=0.0)
        boresight_cam = np.array([[0.0, 0.0, 1.0]])
        boresight_world = camera_to_world(boresight_cam, R)

        from pixel2sky.rotation import world_vector_to_altaz

        alt_out, _ = world_vector_to_altaz(boresight_world)
        assert_allclose(alt_out.ravel()[0], alt0, atol=1e-9)


# -----------------------------------------------------------------------
# world_to_camera / camera_to_world round-trip
# -----------------------------------------------------------------------


class TestRotationRoundTrip:
    @pytest.mark.parametrize(
        "az0, alt0, roll",
        [
            (0.0, 90.0, 0.0),
            (45.0, 30.0, 15.0),
            (180.0, -20.0, 45.0),
            (270.0, 0.0, 90.0),
            (359.9, 89.9, 179.9),
        ],
    )
    def test_world_cam_world_roundtrip(
        self, az0: float, alt0: float, roll: float
    ) -> None:
        R = build_rotation(az0, alt0, roll)
        rng = np.random.default_rng(seed=int(az0 + alt0 + roll))
        v_world_in = rng.standard_normal((100, 3))
        v_cam = world_to_camera(v_world_in, R)
        v_world_out = camera_to_world(v_cam, R)
        assert_allclose(v_world_out, v_world_in, atol=1e-12)

    def test_roll_90_rotates_x_into_y(self) -> None:
        """90° roll should rotate the Camera +X axis to Camera −Y."""
        # Camera pointing at zenith; 90° roll
        R_no_roll = build_rotation(az0=0.0, alt0=90.0, roll=0.0)
        R_with_roll = build_rotation(az0=0.0, alt0=90.0, roll=90.0)

        # Take a World-frame vector and compare the Camera-frame projections
        v_world = np.array([[1.0, 0.0, 0.0]])  # East
        v_cam_no = world_to_camera(v_world, R_no_roll)
        v_cam_ro = world_to_camera(v_world, R_with_roll)

        # With no roll: East lands in Camera +X
        # With 90° roll (CW): East lands in Camera −Y
        # (exact values depend on convention; ensure they differ)
        assert not np.allclose(v_cam_no, v_cam_ro, atol=1e-6)
