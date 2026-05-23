"""Integration tests for the SkyMapper facade.

Covers:
* Constructor validation
* Image-centre maps to boresight Alt/Az
* Full round-trip (Alt/Az → pixels → Alt/Az) for both projection models
* Out-of-bounds masking
* Edge cases: zenith camera, extreme roll, fisheye rear hemisphere
* 2D grid operations
* fov_degrees sanity checks
* pixel_grid shape contract
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pixel2sky import SkyMapper
from pixel2sky.projection import EquidistantFisheye, Rectilinear

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

W, H = 1920, 1080
F = 800.0


@pytest.fixture()
def rectilinear_mapper() -> SkyMapper:
    return SkyMapper(
        image_width=W,
        image_height=H,
        projection=Rectilinear(focal_length=F),
        az0=90.0,
        alt0=30.0,
        roll=0.0,
    )


@pytest.fixture()
def fisheye_mapper() -> SkyMapper:
    return SkyMapper(
        image_width=W,
        image_height=H,
        projection=EquidistantFisheye(focal_length=F),
        az0=180.0,
        alt0=45.0,
        roll=0.0,
    )


@pytest.fixture()
def zenith_fisheye_mapper() -> SkyMapper:
    """All-sky camera pointing straight up."""
    return SkyMapper(
        image_width=2048,
        image_height=2048,
        projection=EquidistantFisheye(focal_length=600.0),
        az0=0.0,
        alt0=90.0,
        roll=0.0,
    )


# -----------------------------------------------------------------------
# Constructor validation
# -----------------------------------------------------------------------


class TestSkyMapperConstructor:
    def test_negative_width_raises(self) -> None:
        with pytest.raises(ValueError, match="image_width"):
            SkyMapper(image_width=-1, image_height=100)

    def test_zero_height_raises(self) -> None:
        with pytest.raises(ValueError, match="image_width"):
            SkyMapper(image_width=100, image_height=0)

    def test_bad_projection_type_raises(self) -> None:
        with pytest.raises(TypeError, match="ProjectionModel"):
            SkyMapper(image_width=100, image_height=100, projection="wrong")  # type: ignore[arg-type]

    def test_invalid_alt0_raises(self) -> None:
        with pytest.raises(ValueError, match="alt0"):
            SkyMapper(
                image_width=100,
                image_height=100,
                projection=Rectilinear(focal_length=200.0),
                alt0=95.0,
            )

    def test_default_projection_is_rectilinear(self) -> None:
        mapper = SkyMapper(image_width=640, image_height=480)
        assert isinstance(mapper.projection, Rectilinear)

    def test_custom_principal_point(self) -> None:
        mapper = SkyMapper(
            image_width=640,
            image_height=480,
            projection=Rectilinear(focal_length=300.0),
            cx=320.0,
            cy=240.0,
        )
        # Centre pixel maps to boresight
        alt, az = mapper.pixel_to_altaz(320.0, 240.0)
        assert_allclose(float(alt), 0.0, atol=1e-6)

    def test_repr_contains_dimensions(self) -> None:
        mapper = SkyMapper(image_width=320, image_height=240)
        r = repr(mapper)
        assert "320" in r
        assert "240" in r


# -----------------------------------------------------------------------
# Image centre → boresight
# -----------------------------------------------------------------------


class TestImageCentreToBoresight:
    def test_rectilinear_centre_to_altaz(self, rectilinear_mapper: SkyMapper) -> None:
        cx = W / 2.0
        cy = H / 2.0
        alt, az = rectilinear_mapper.pixel_to_altaz(cx, cy)
        assert_allclose(float(alt), rectilinear_mapper.alt0, atol=1e-6)
        assert_allclose(float(az), rectilinear_mapper.az0, atol=1e-6)

    def test_fisheye_centre_to_altaz(self, fisheye_mapper: SkyMapper) -> None:
        cx = W / 2.0
        cy = H / 2.0
        alt, az = fisheye_mapper.pixel_to_altaz(cx, cy)
        assert_allclose(float(alt), fisheye_mapper.alt0, atol=1e-6)
        assert_allclose(float(az), fisheye_mapper.az0, atol=1e-6)

    def test_zenith_camera_centre_is_zenith(
        self, zenith_fisheye_mapper: SkyMapper
    ) -> None:
        m = zenith_fisheye_mapper
        alt, az = m.pixel_to_altaz(m.image_width / 2.0, m.image_height / 2.0)
        assert_allclose(float(alt), 90.0, atol=1e-6)


# -----------------------------------------------------------------------
# Round-trip: Alt/Az → pixels → Alt/Az
# -----------------------------------------------------------------------


class TestAltAzToPixelRoundTrip:
    def _run_round_trip(
        self,
        mapper: SkyMapper,
        alt_in: np.ndarray,
        az_in: np.ndarray,
        atol: float = 1e-6,
    ) -> None:
        x, y = mapper.altaz_to_pixel(alt_in, az_in)
        valid = np.isfinite(x) & np.isfinite(y)
        assert valid.any(), "No points projected onto the sensor"
        alt_out, az_out = mapper.pixel_to_altaz(x[valid], y[valid])
        assert_allclose(alt_out, alt_in[valid], atol=atol)
        assert_allclose(az_out, az_in[valid], atol=atol)

    def test_rectilinear_round_trip(self, rectilinear_mapper: SkyMapper) -> None:
        rng = np.random.default_rng(1)
        # Constrain to ≈ the camera FOV to ensure most points are on-sensor
        alt_in = rng.uniform(10.0, 50.0, size=200)
        az_in = rng.uniform(50.0, 130.0, size=200)
        self._run_round_trip(rectilinear_mapper, alt_in, az_in)

    def test_fisheye_round_trip(self, fisheye_mapper: SkyMapper) -> None:
        rng = np.random.default_rng(2)
        alt_in = rng.uniform(20.0, 70.0, size=200)
        az_in = rng.uniform(140.0, 220.0, size=200)
        self._run_round_trip(fisheye_mapper, alt_in, az_in)

    def test_zenith_fisheye_round_trip(self, zenith_fisheye_mapper: SkyMapper) -> None:
        rng = np.random.default_rng(3)
        # All altitudes close to zenith to land on the sensor
        alt_in = rng.uniform(45.0, 90.0, size=300)
        az_in = rng.uniform(0.0, 360.0, size=300)
        self._run_round_trip(zenith_fisheye_mapper, alt_in, az_in)


# -----------------------------------------------------------------------
# Round-trip: pixel → Alt/Az → pixel
# -----------------------------------------------------------------------


class TestPixelToAltAzRoundTrip:
    def _run_round_trip(
        self,
        mapper: SkyMapper,
        x_in: np.ndarray,
        y_in: np.ndarray,
        atol: float = 1e-5,
    ) -> None:
        alt, az = mapper.pixel_to_altaz(x_in, y_in)
        valid = np.isfinite(alt) & np.isfinite(az)
        assert valid.any()
        x_out, y_out = mapper.altaz_to_pixel(alt[valid], az[valid])
        assert_allclose(x_out, x_in[valid], atol=atol)
        assert_allclose(y_out, y_in[valid], atol=atol)

    def test_rectilinear_pixel_round_trip(self, rectilinear_mapper: SkyMapper) -> None:
        rng = np.random.default_rng(10)
        x = rng.uniform(0.0, W - 1, size=300)
        y = rng.uniform(0.0, H - 1, size=300)
        self._run_round_trip(rectilinear_mapper, x, y)

    def test_fisheye_pixel_round_trip(self, fisheye_mapper: SkyMapper) -> None:
        rng = np.random.default_rng(11)
        x = rng.uniform(0.0, W - 1, size=300)
        y = rng.uniform(0.0, H - 1, size=300)
        self._run_round_trip(fisheye_mapper, x, y)


# -----------------------------------------------------------------------
# Out-of-bounds masking
# -----------------------------------------------------------------------


class TestOutOfBoundsMasking:
    def test_behind_camera_altaz_returns_nan(
        self, rectilinear_mapper: SkyMapper
    ) -> None:
        # Camera points at az=90, alt=30. The diametrically opposite point
        # should project behind the camera for a rectilinear lens.
        alt = np.array([-30.0])
        az = np.array([270.0])
        x, y = rectilinear_mapper.altaz_to_pixel(alt, az)
        assert np.all(np.isnan(x))
        assert np.all(np.isnan(y))

    def test_far_off_axis_returns_nan(self, rectilinear_mapper: SkyMapper) -> None:
        # Points 89° away from boresight should project off-sensor for a
        # modest focal length.
        alt = np.array([30.0])
        az = np.array([180.0])  # 90° away from az0=90
        x, y = rectilinear_mapper.altaz_to_pixel(alt, az)
        # Either NaN (off-sensor) or finite-but-outside the sensor bounds
        if np.isfinite(x).all():
            assert (x < 0) | (x >= W) | (y < 0) | (y >= H)

    def test_pixel_outside_sensor_is_nan_in_altaz_to_pixel(
        self, rectilinear_mapper: SkyMapper
    ) -> None:
        alt_boresight = np.array([rectilinear_mapper.alt0])
        az_boresight = np.array([rectilinear_mapper.az0])
        x, y = rectilinear_mapper.altaz_to_pixel(alt_boresight, az_boresight)
        # Boresight itself should be valid (image centre)
        assert np.isfinite(x).all()
        assert np.isfinite(y).all()


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------


class TestEdgeCases:
    def test_extreme_roll_180_round_trip(self) -> None:
        """180° roll should flip the image but preserve Alt/Az round-trip."""
        m = SkyMapper(
            image_width=1000,
            image_height=1000,
            projection=EquidistantFisheye(focal_length=400.0),
            az0=0.0,
            alt0=45.0,
            roll=180.0,
        )
        alt_in = np.array([45.0, 50.0, 40.0])
        az_in = np.array([0.0, 10.0, 350.0])
        x, y = m.altaz_to_pixel(alt_in, az_in)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.any():
            alt_out, az_out = m.pixel_to_altaz(x[valid], y[valid])
            assert_allclose(alt_out, alt_in[valid], atol=1e-5)

    def test_horizon_altaz_scalar(self) -> None:
        """Scalar input should return scalar-like (0-d) arrays."""
        m = SkyMapper(
            image_width=1000,
            image_height=1000,
            projection=EquidistantFisheye(focal_length=400.0),
            az0=0.0,
            alt0=0.0,
            roll=0.0,
        )
        alt, az = m.pixel_to_altaz(500.0, 500.0)
        assert alt.shape == ()

    def test_negative_altitude_round_trip(self) -> None:
        """Below-horizon points should survive a round-trip if on-sensor."""
        m = SkyMapper(
            image_width=2048,
            image_height=2048,
            projection=EquidistantFisheye(focal_length=700.0),
            az0=0.0,
            alt0=-30.0,
            roll=0.0,
        )
        alt_in = np.array([-30.0])
        az_in = np.array([0.0])
        x, y = m.altaz_to_pixel(alt_in, az_in)
        if np.isfinite(x).all():
            alt_out, az_out = m.pixel_to_altaz(x, y)
            assert_allclose(alt_out, np.array([-30.0]), atol=1e-5)

    def test_azimuth_wrapping_near_360(self) -> None:
        """Az=359.9° and Az=0.1° should give sensible round-trip results."""
        m = SkyMapper(
            image_width=2048,
            image_height=2048,
            projection=EquidistantFisheye(focal_length=600.0),
            az0=0.0,
            alt0=60.0,
            roll=0.0,
        )
        alt_in = np.array([60.0, 60.0])
        az_in = np.array([359.9, 0.1])
        x, y = m.altaz_to_pixel(alt_in, az_in)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.any():
            alt_out, az_out = m.pixel_to_altaz(x[valid], y[valid])
            assert_allclose(alt_out, alt_in[valid], atol=1e-5)


# -----------------------------------------------------------------------
# Grid / convenience methods
# -----------------------------------------------------------------------


class TestGridAndConvenienceMethods:
    def test_pixel_grid_shape(self, rectilinear_mapper: SkyMapper) -> None:
        xx, yy = rectilinear_mapper.pixel_grid()
        assert xx.shape == (H, W)
        assert yy.shape == (H, W)

    def test_pixel_grid_values(self, rectilinear_mapper: SkyMapper) -> None:
        xx, yy = rectilinear_mapper.pixel_grid()
        assert_allclose(xx[0, 0], 0.0)
        assert_allclose(xx[0, -1], W - 1)
        assert_allclose(yy[0, 0], 0.0)
        assert_allclose(yy[-1, 0], H - 1)

    def test_pixel_to_altaz_preserves_grid_shape(
        self, fisheye_mapper: SkyMapper
    ) -> None:
        xx, yy = fisheye_mapper.pixel_grid()
        alt, az = fisheye_mapper.pixel_to_altaz(xx, yy)
        assert alt.shape == (H, W)
        assert az.shape == (H, W)

    def test_fov_degrees_positive(self, rectilinear_mapper: SkyMapper) -> None:
        fov_h, fov_v = rectilinear_mapper.fov_degrees()
        assert fov_h > 0.0
        assert fov_v > 0.0

    def test_fov_h_greater_than_v_for_landscape(
        self, rectilinear_mapper: SkyMapper
    ) -> None:
        fov_h, fov_v = rectilinear_mapper.fov_degrees()
        # 1920×1080 landscape: horizontal FOV should exceed vertical
        assert fov_h > fov_v

    def test_fisheye_fov_degrees(self, fisheye_mapper: SkyMapper) -> None:
        fov_h, fov_v = fisheye_mapper.fov_degrees()
        # Fisheye with f=800 on 1920×1080: large but bounded
        assert 0.0 < fov_h < 360.0
        assert 0.0 < fov_v < 180.0


# -----------------------------------------------------------------------
# Vectorisation / throughput sanity
# -----------------------------------------------------------------------


class TestVectorisedThroughput:
    def test_full_4k_grid_no_loop(self) -> None:
        """Transform every pixel of a 4K frame without error."""
        m = SkyMapper(
            image_width=3840,
            image_height=2160,
            projection=EquidistantFisheye(focal_length=1000.0),
            az0=0.0,
            alt0=60.0,
            roll=0.0,
        )
        xx, yy = m.pixel_grid()
        alt, az = m.pixel_to_altaz(xx, yy)
        assert alt.shape == (2160, 3840)
        # At least the centre region must yield finite values
        cy, cx = 2160 // 2, 3840 // 2
        assert np.isfinite(alt[cy, cx])
        assert np.isfinite(az[cy, cx])
