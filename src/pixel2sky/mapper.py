"""High-level API facade: the SkyMapper class.

:class:`SkyMapper` is the primary entry point for ``pixel2sky``. It composes
a :class:`~pixel2sky.projection.ProjectionModel` (camera intrinsics) with a
rotation built from pointing parameters (camera extrinsics) to provide two
public transformation methods:

* :meth:`SkyMapper.pixel_to_altaz` — pixels → Alt/Az sky coordinates
* :meth:`SkyMapper.altaz_to_pixel` — Alt/Az sky coordinates → pixels

Both methods are **fully vectorised**: they accept and return :class:`numpy`
arrays of arbitrary shape and perform no Python-level loops, making them
suitable for transforming entire image grids in a single call.

Architecture
------------
The transformation pipeline is:

    Pixel (x, y)
        │  subtract principal point (cx, cy)
        ▼
    Offset pixel (dx, dy)
        │  ProjectionModel.pixel_to_ray()
        ▼
    Camera-frame unit ray  [Xc, Yc, Zc]
        │  Rotation.inv().apply()
        ▼
    World-frame unit vector  [East, North, Zenith]
        │  world_vector_to_altaz()
        ▼
    Sky coordinates (Altitude °, Azimuth °)

The inverse pipeline runs in the opposite direction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from pixel2sky.projection import ProjectionModel, Rectilinear
from pixel2sky.rotation import (
    altaz_to_world_vector,
    build_rotation,
    camera_to_world,
    world_to_camera,
    world_vector_to_altaz,
)


class SkyMapper:
    """Bidirectional pixel ↔ sky coordinate transformer.

    ``SkyMapper`` combines a lens projection model (intrinsics) with a 3D
    rotation derived from the camera's pointing direction (extrinsics) to map
    2D image pixels to 3D local-horizontal sky coordinates and back.

    Attributes:
        image_width: Sensor width in pixels.
        image_height: Sensor height in pixels.
        projection: The :class:`~pixel2sky.projection.ProjectionModel` in use.
        az0: Azimuth of the camera boresight, degrees (clockwise from North).
        alt0: Altitude of the camera boresight, degrees (0=horizon, 90=zenith).
        roll: Sensor roll about the optical axis, degrees (clockwise).

    Examples:
        >>> import numpy as np
        >>> from pixel2sky import SkyMapper
        >>> from pixel2sky.projection import EquidistantFisheye
        >>> mapper = SkyMapper(
        ...     image_width=1920,
        ...     image_height=1080,
        ...     projection=EquidistantFisheye(focal_length=500.0),
        ...     az0=90.0,
        ...     alt0=45.0,
        ...     roll=0.0,
        ... )
        >>> alt, az = mapper.pixel_to_altaz(960.0, 540.0)  # image centre
        >>> np.isclose(alt, 45.0, atol=1e-6)
        True
        >>> np.isclose(az, 90.0, atol=1e-6)
        True
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        projection: ProjectionModel | None = None,
        az0: float = 0.0,
        alt0: float = 0.0,
        roll: float = 0.0,
        cx: float | None = None,
        cy: float | None = None,
    ) -> None:
        """Initialise the SkyMapper with intrinsic and extrinsic parameters.

        Args:
            image_width: Width of the image sensor in pixels (> 0).
            image_height: Height of the image sensor in pixels (> 0).
            projection: A :class:`~pixel2sky.projection.ProjectionModel`
                instance describing the lens geometry.  When ``None`` a
                :class:`~pixel2sky.projection.Rectilinear` model with a focal
                length equal to ``image_width`` is used (≈ 53° horizontal
                FOV for a full-frame equivalent).
            az0: Azimuth of the camera boresight in degrees. Measured
                clockwise from North: 0 = North, 90 = East, 180 = South,
                270 = West.  Defaults to 0.
            alt0: Altitude (elevation) of the camera boresight in degrees.
                0 = horizon, 90 = zenith, negative = below horizon.
                Defaults to 0.
            roll: Clockwise rotation of the sensor about the optical axis in
                degrees.  Defaults to 0 (sensor top toward zenith / horizon).
            cx: Override for the image-plane principal point x-coordinate
                (column), in **absolute** pixel units.  When ``None`` the
                image centre ``image_width / 2.0`` is used.
            cy: Override for the image-plane principal point y-coordinate
                (row), in **absolute** pixel units.  When ``None`` the image
                centre ``image_height / 2.0`` is used.

        Raises:
            TypeError: If ``projection`` is not a
                :class:`~pixel2sky.projection.ProjectionModel` instance.
            ValueError: If ``image_width`` or ``image_height`` are not
                strictly positive integers, or if ``alt0`` is outside
                ``[−90, 90]``.
        """
        if image_width <= 0 or image_height <= 0:
            raise ValueError(
                "image_width and image_height must be positive integers, "
                f"got ({image_width}, {image_height})"
            )

        self.image_width: int = int(image_width)
        self.image_height: int = int(image_height)

        # Principal point defaults to image centre
        self._cx: float = float(cx) if cx is not None else image_width / 2.0
        self._cy: float = float(cy) if cy is not None else image_height / 2.0

        if projection is None:
            projection = Rectilinear(focal_length=float(image_width))
        if not isinstance(projection, ProjectionModel):
            raise TypeError(
                f"projection must be a ProjectionModel instance, "
                f"got {type(projection)!r}"
            )
        self.projection: ProjectionModel = projection

        self.az0: float = float(az0)
        self.alt0: float = float(alt0)
        self.roll: float = float(roll)

        # Pre-compute and cache the rotation object — it is immutable
        self._rotation: Rotation = build_rotation(az0, alt0, roll)

    # ------------------------------------------------------------------
    # Public transformation methods
    # ------------------------------------------------------------------

    def pixel_to_altaz(
        self,
        x: NDArray[np.float64] | float,
        y: NDArray[np.float64] | float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Transform 2D pixel coordinates to Alt/Az sky coordinates.

        This is a fully vectorised operation. Pass arrays of any shape;
        ``alt`` and ``az`` will have the same shape.

        Pipeline::

            (x, y) → (dx, dy) → Camera ray → World vector → (Alt, Az)

        Args:
            x: Pixel column coordinate(s), 0-indexed from the left edge.
                Accepts a scalar or any array shape.
            y: Pixel row coordinate(s), 0-indexed from the top edge.
                Same shape as ``x``.

        Returns:
            A tuple ``(alt, az)`` where:

            * ``alt`` – Altitude in degrees ``∈ [−90, 90]``.
            * ``az``  – Azimuth in degrees ``∈ [0, 360)``, clockwise from
              North.

            Both arrays have the same shape as ``x``.  Pixels that land
            outside the valid angular range of the projection model return
            ``NaN``.

        Examples:
            Transform a full 1080p image grid in one call:

            >>> x = np.arange(1920, dtype=float)
            >>> y = np.arange(1080, dtype=float)
            >>> xx, yy = np.meshgrid(x, y)
            >>> alt, az = mapper.pixel_to_altaz(xx, yy)
            >>> alt.shape
            (1080, 1920)
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # 1. Convert absolute pixel coords to offsets from principal point
        dx = x - self._cx
        dy = y - self._cy

        # 2. Back-project through the lens model → unit ray in Camera frame
        v_cam = self.projection.pixel_to_ray(dx, dy)

        # 3. Rotate into World frame
        original_shape = v_cam.shape[:-1]
        v_cam_flat = v_cam.reshape(-1, 3)
        v_world_flat = camera_to_world(v_cam_flat, self._rotation)
        v_world = v_world_flat.reshape(*original_shape, 3)

        # 4. Convert World-frame vector to spherical Alt/Az
        alt, az = world_vector_to_altaz(v_world)

        return alt, az

    def altaz_to_pixel(
        self,
        alt: NDArray[np.float64] | float,
        az: NDArray[np.float64] | float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Transform Alt/Az sky coordinates to 2D pixel coordinates.

        Points that are behind the camera, outside the sensor bounds, or
        otherwise unprojectable are returned as ``NaN``.

        Pipeline::

            (Alt, Az) → World vector → Camera ray → (dx, dy) → (x, y)

        Args:
            alt: Altitude angle(s) in degrees ``∈ [−90, 90]``.
                0 = horizon, 90 = zenith.  Accepts scalar or any array shape.
            az: Azimuth angle(s) in degrees.  0 = North, 90 = East (clockwise
                from North).  Same shape as ``alt``.

        Returns:
            A tuple ``(x, y)`` of pixel coordinates where:

            * ``x`` – Column coordinate, 0-indexed from the left edge.
            * ``y`` – Row coordinate, 0-indexed from the top edge.

            Both arrays have the same shape as ``alt``.  Invalid or
            out-of-bounds points are ``NaN``.

        Examples:
            >>> alt = np.array([45.0, 30.0, 60.0])
            >>> az  = np.array([90.0, 180.0, 270.0])
            >>> x, y = mapper.altaz_to_pixel(alt, az)
        """
        alt = np.asarray(alt, dtype=np.float64)
        az = np.asarray(az, dtype=np.float64)

        # 1. Convert spherical Alt/Az to World-frame unit vector
        v_world = altaz_to_world_vector(alt, az)

        # 2. Rotate into Camera frame
        original_shape = v_world.shape[:-1]
        v_world_flat = v_world.reshape(-1, 3)
        v_cam_flat = world_to_camera(v_world_flat, self._rotation)
        v_cam = v_cam_flat.reshape(*original_shape, 3)

        # 3. Project through lens model → offset pixel coordinates
        dx, dy = self.projection.ray_to_pixel(v_cam)

        # 4. Convert offsets to absolute pixel coordinates
        x = dx + self._cx
        y = dy + self._cy

        # 5. Mask points that fall outside the sensor boundaries
        x, y = self._mask_out_of_bounds(x, y)

        return x, y

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def pixel_grid(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return a meshgrid covering every pixel of the sensor.

        Returns:
            Tuple ``(xx, yy)`` of shape ``(image_height, image_width)``
            containing the centre coordinate of every pixel.

        Examples:
            >>> xx, yy = mapper.pixel_grid()
            >>> alt, az = mapper.pixel_to_altaz(xx, yy)
        """
        x = np.arange(self.image_width, dtype=np.float64)
        y = np.arange(self.image_height, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        return xx.astype(np.float64), yy.astype(np.float64)

    def fov_degrees(self) -> tuple[float, float]:
        """Approximate horizontal and vertical field of view.

        Computes the angular span by transforming the four image corners and
        measuring the great-circle separations between opposing pairs.

        Returns:
            Tuple ``(fov_h, fov_v)`` in degrees, where ``fov_h`` is the
            horizontal FOV and ``fov_v`` is the vertical FOV.

        Note:
            For very wide-angle or fisheye lenses the FOV is computed
            correctly because it uses the full sky-angle calculation rather
            than a paraxial approximation.
        """
        corners_x = np.array([0.0, self.image_width - 1, 0.0, self.image_width - 1])
        corners_y = np.array([0.0, 0.0, self.image_height - 1, self.image_height - 1])
        alt_c, az_c = self.pixel_to_altaz(corners_x, corners_y)

        def _angular_separation(
            alt1: float, az1: float, alt2: float, az2: float
        ) -> float:
            a1 = np.deg2rad(alt1)
            a2 = np.deg2rad(alt2)
            daz = np.deg2rad(az2 - az1)
            cos_sep = np.sin(a1) * np.sin(a2) + np.cos(a1) * np.cos(a2) * np.cos(daz)
            return float(np.rad2deg(np.arccos(np.clip(cos_sep, -1.0, 1.0))))

        fov_h = _angular_separation(alt_c[0], az_c[0], alt_c[1], az_c[1])
        fov_v = _angular_separation(alt_c[0], az_c[0], alt_c[2], az_c[2])
        return fov_h, fov_v

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _mask_out_of_bounds(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Replace out-of-bounds pixel coordinates with NaN.

        Pixels are considered valid when their centre lies within the half-open
        rectangle ``[0, image_width) × [0, image_height)``.

        Args:
            x: Column pixel coordinates, any shape.
            y: Row pixel coordinates, same shape as ``x``.

        Returns:
            ``(x_masked, y_masked)`` with the same shape; invalid entries are
            ``NaN``.
        """
        valid = (
            np.isfinite(x)
            & np.isfinite(y)
            & (x >= 0.0)
            & (x < self.image_width)
            & (y >= 0.0)
            & (y < self.image_height)
        )
        x_out = np.where(valid, x, np.nan)
        y_out = np.where(valid, y, np.nan)
        return x_out, y_out

    def __repr__(self) -> str:
        return (
            f"SkyMapper("
            f"image={self.image_width}×{self.image_height}, "
            f"projection={self.projection!r}, "
            f"az0={self.az0}°, alt0={self.alt0}°, roll={self.roll}°)"
        )
