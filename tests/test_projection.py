"""Unit tests for the camera projection models.

Covers both :class:`~pixel2sky.projection.Rectilinear` and
:class:`~pixel2sky.projection.EquidistantFisheye`, including:

* Optical-axis identity
* Round-trip consistency (pixel → ray → pixel)
* Behind-camera masking
* Non-square pixel support
* Invalid-parameter rejection
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pixel2sky.projection import EquidistantFisheye, Rectilinear


# -----------------------------------------------------------------------
# Shared fixtures
# -----------------------------------------------------------------------

@pytest.fixture(params=[100.0, 500.0, 1200.0])
def focal_length(request: pytest.FixtureRequest) -> float:
    return request.param


# -----------------------------------------------------------------------
# ProjectionModel base validation
# -----------------------------------------------------------------------

class TestProjectionModelValidation:
    def test_negative_focal_length_raises(self) -> None:
        with pytest.raises(ValueError, match="focal_length"):
            Rectilinear(focal_length=-1.0)

    def test_zero_focal_length_raises(self) -> None:
        with pytest.raises(ValueError, match="focal_length"):
            Rectilinear(focal_length=0.0)

    def test_negative_fy_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="fy_scale"):
            Rectilinear(focal_length=500.0, fy_scale=-1.0)

    def test_repr_contains_class_name(self) -> None:
        model = Rectilinear(focal_length=400.0)
        assert "Rectilinear" in repr(model)

    def test_fisheye_repr_contains_class_name(self) -> None:
        model = EquidistantFisheye(focal_length=400.0)
        assert "EquidistantFisheye" in repr(model)


# -----------------------------------------------------------------------
# Rectilinear
# -----------------------------------------------------------------------

class TestRectilinear:
    def test_optical_axis_back_projects_to_z(self, focal_length: float) -> None:
        """(dx, dy) = (0, 0) must yield the +Z unit vector."""
        model = Rectilinear(focal_length=focal_length)
        ray = model.pixel_to_ray(np.array([0.0]), np.array([0.0]))
        assert_allclose(ray[0], [0.0, 0.0, 1.0], atol=1e-12)

    def test_pixel_to_ray_shape_preserved(self, focal_length: float) -> None:
        model = Rectilinear(focal_length=focal_length)
        dx = np.zeros((10, 20))
        dy = np.zeros((10, 20))
        rays = model.pixel_to_ray(dx, dy)
        assert rays.shape == (10, 20, 3)

    def test_ray_to_pixel_shape_preserved(self, focal_length: float) -> None:
        model = Rectilinear(focal_length=focal_length)
        rays = np.zeros((5, 3))
        rays[:, 2] = 1.0  # +Z axis
        dx, dy = model.ray_to_pixel(rays)
        assert dx.shape == (5,)
        assert dy.shape == (5,)

    def test_round_trip_arbitrary_offsets(self, focal_length: float) -> None:
        """pixel_to_ray followed by ray_to_pixel should be the identity."""
        model = Rectilinear(focal_length=focal_length)
        rng = np.random.default_rng(42)
        # Keep offsets small enough to stay in the front hemisphere
        dx_in = rng.uniform(-focal_length * 0.5, focal_length * 0.5, size=200)
        dy_in = rng.uniform(-focal_length * 0.5, focal_length * 0.5, size=200)

        rays = model.pixel_to_ray(dx_in, dy_in)
        dx_out, dy_out = model.ray_to_pixel(rays)

        assert_allclose(dx_out, dx_in, atol=1e-9)
        assert_allclose(dy_out, dy_in, atol=1e-9)

    def test_behind_camera_is_nan(self, focal_length: float) -> None:
        """Rays with z ≤ 0 must produce NaN pixel offsets."""
        model = Rectilinear(focal_length=focal_length)
        behind = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
        dx, dy = model.ray_to_pixel(behind)
        assert np.all(np.isnan(dx))
        assert np.all(np.isnan(dy))

    def test_non_square_pixels(self) -> None:
        """fy = 2·fx should double the dy offset for the same ray."""
        f = 500.0
        m_sq = Rectilinear(focal_length=f, fy_scale=1.0)
        m_ns = Rectilinear(focal_length=f, fy_scale=2.0)

        ray = np.array([[0.1, 0.2, 1.0]])  # arbitrary forward ray
        _, dy_sq = m_sq.ray_to_pixel(ray)
        _, dy_ns = m_ns.ray_to_pixel(ray)

        assert_allclose(dy_ns, 2.0 * dy_sq, rtol=1e-10)

    def test_output_rays_are_unit_vectors(self, focal_length: float) -> None:
        model = Rectilinear(focal_length=focal_length)
        dx = np.linspace(-focal_length, focal_length, 50)
        dy = np.zeros(50)
        rays = model.pixel_to_ray(dx, dy)
        norms = np.linalg.norm(rays, axis=-1)
        assert_allclose(norms, 1.0, atol=1e-12)

    def test_symmetry_left_right(self, focal_length: float) -> None:
        """Symmetric offset ±dx should give ±x ray component."""
        model = Rectilinear(focal_length=focal_length)
        dx = np.array([100.0, -100.0])
        dy = np.array([0.0, 0.0])
        rays = model.pixel_to_ray(dx, dy)
        assert_allclose(rays[0, 0], -rays[1, 0], atol=1e-12)
        assert_allclose(rays[0, 1], rays[1, 1], atol=1e-12)


# -----------------------------------------------------------------------
# EquidistantFisheye
# -----------------------------------------------------------------------

class TestEquidistantFisheye:
    def test_optical_axis_back_projects_to_z(self, focal_length: float) -> None:
        model = EquidistantFisheye(focal_length=focal_length)
        ray = model.pixel_to_ray(np.array([0.0]), np.array([0.0]))
        assert_allclose(ray[0], [0.0, 0.0, 1.0], atol=1e-12)

    def test_round_trip_forward_hemisphere(self, focal_length: float) -> None:
        """Full round-trip for rays in the front hemisphere."""
        model = EquidistantFisheye(focal_length=focal_length)
        rng = np.random.default_rng(7)
        # θ ∈ [0°, 89°]
        theta = rng.uniform(0.0, np.deg2rad(89.0), size=300)
        phi = rng.uniform(0.0, 2 * np.pi, size=300)
        dx_in = focal_length * theta * np.cos(phi)
        dy_in = focal_length * theta * np.sin(phi)

        rays = model.pixel_to_ray(dx_in, dy_in)
        dx_out, dy_out = model.ray_to_pixel(rays)

        assert_allclose(dx_out, dx_in, atol=1e-9)
        assert_allclose(dy_out, dy_in, atol=1e-9)

    def test_round_trip_rear_hemisphere(self, focal_length: float) -> None:
        """Fisheye model supports θ > 90° (behind optical axis)."""
        model = EquidistantFisheye(focal_length=focal_length)
        # θ = 135° → r = f·(3π/4)
        theta = np.deg2rad(135.0)
        r = focal_length * theta
        dx_in = np.array([r, 0.0])
        dy_in = np.array([0.0, r])

        rays = model.pixel_to_ray(dx_in, dy_in)
        dx_out, dy_out = model.ray_to_pixel(rays)

        assert_allclose(dx_out, dx_in, atol=1e-9)
        assert_allclose(dy_out, dy_in, atol=1e-9)

    def test_output_rays_are_unit_vectors(self, focal_length: float) -> None:
        model = EquidistantFisheye(focal_length=focal_length)
        theta = np.linspace(0.0, np.deg2rad(170.0), 50)
        dx = focal_length * theta
        dy = np.zeros(50)
        rays = model.pixel_to_ray(dx, dy)
        norms = np.linalg.norm(rays, axis=-1)
        assert_allclose(norms, 1.0, atol=1e-12)

    def test_radial_distance_matches_formula(self, focal_length: float) -> None:
        """Projected radius must equal f·θ for known angles."""
        model = EquidistantFisheye(focal_length=focal_length)
        thetas = np.deg2rad([0.0, 30.0, 60.0, 90.0, 120.0])
        # Rays pointing straight right in Camera frame at angle θ from +Z
        rays = np.stack(
            [np.sin(thetas), np.zeros_like(thetas), np.cos(thetas)], axis=-1
        )
        dx, dy = model.ray_to_pixel(rays)

        expected_r = focal_length * thetas
        actual_r = np.sqrt(dx**2 + dy**2)
        assert_allclose(actual_r, expected_r, atol=1e-9)

    def test_principal_point_no_division_by_zero(self) -> None:
        """r=0 (principal point) must not raise and must return (0, 0)."""
        model = EquidistantFisheye(focal_length=300.0)
        ray = np.array([[0.0, 0.0, 1.0]])
        dx, dy = model.ray_to_pixel(ray)
        assert_allclose(dx, [0.0], atol=1e-12)
        assert_allclose(dy, [0.0], atol=1e-12)

    def test_pixel_shape_2d_batch(self) -> None:
        model = EquidistantFisheye(focal_length=400.0)
        dx = np.zeros((8, 16))
        dy = np.zeros((8, 16))
        rays = model.pixel_to_ray(dx, dy)
        assert rays.shape == (8, 16, 3)
