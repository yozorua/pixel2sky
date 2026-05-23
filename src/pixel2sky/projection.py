"""Camera intrinsic projection models.

This module implements the lens geometry (intrinsics) layer of the pipeline.
Each model converts between 2D sensor coordinates (in pixels, relative to the
principal point) and the 3D unit vector pointing into the scene, expressed in
the **Camera frame** where:

  * +X  →  right  (increasing column)
  * +Y  →  down   (increasing row)
  * +Z  →  into the scene (optical axis)

Coordinate conventions
----------------------
All public methods work with *offset* pixel coordinates—that is, pixel
position measured relative to the principal point (cx, cy):

    dx = x - cx
    dy = y - cy

Subclasses must implement :meth:`ProjectionModel.pixel_to_ray` and
:meth:`ProjectionModel.ray_to_pixel`.  The :class:`SkyMapper` facade calls
only these two entry-points.

References
----------
Kannala & Brandt (2006), "A Generic Camera Model and Calibration Method for
Conventional, Wide-Angle, and Fish-Eye Lenses", IEEE TPAMI 28(8).
"""

from __future__ import annotations

import abc
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


class ProjectionModel(abc.ABC):
    """Abstract base class for camera projection models.

    A projection model encapsulates the mathematical relationship between a
    2D image sensor and the 3D ray directions it can observe. Concrete
    subclasses implement specific lens geometries (pinhole, fisheye, etc.).

    Attributes:
        fx: Focal length along the x-axis, in pixels.
        fy: Focal length along the y-axis, in pixels.
        cx: Principal point x-coordinate, in pixels (column).
        cy: Principal point y-coordinate, in pixels (row).
    """

    def __init__(
        self,
        focal_length: float,
        cx: float = 0.0,
        cy: float = 0.0,
        fy_scale: float = 1.0,
    ) -> None:
        """Initialise shared intrinsic parameters.

        Args:
            focal_length: Effective focal length in pixels (used as ``fx``).
                For most symmetric lenses a single value suffices.
            cx: Principal point x-offset from the image centre, in pixels.
                Positive values shift the principal point to the right.
                Defaults to 0 (image centre assumed by the caller).
            cy: Principal point y-offset from the image centre, in pixels.
                Positive values shift the principal point downward.
                Defaults to 0.
            fy_scale: Ratio ``fy / fx`` to model non-square pixels.
                Defaults to 1.0 (square pixels).

        Raises:
            ValueError: If ``focal_length`` is not strictly positive, or if
                ``fy_scale`` is not strictly positive.
        """
        if focal_length <= 0.0:
            raise ValueError(
                f"focal_length must be > 0, got {focal_length!r}"
            )
        if fy_scale <= 0.0:
            raise ValueError(f"fy_scale must be > 0, got {fy_scale!r}")

        self.fx: float = float(focal_length)
        self.fy: float = float(focal_length) * float(fy_scale)
        self.cx: float = float(cx)
        self.cy: float = float(cy)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def pixel_to_ray(
        self,
        dx: NDArray[np.float64],
        dy: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Map offset pixel coordinates to unit rays in the Camera frame.

        Args:
            dx: Horizontal pixel offsets from the principal point, shape
                ``(...,)``.  Positive values are to the right.
            dy: Vertical pixel offsets from the principal point, shape
                ``(...,)``.  Positive values are downward.

        Returns:
            Unit direction vectors of shape ``(..., 3)`` in the Camera frame
            ``[x_cam, y_cam, z_cam]``.  The vectors are guaranteed to be
            normalised (‖v‖ = 1).
        """

    @abc.abstractmethod
    def ray_to_pixel(
        self,
        rays: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Map unit rays in the Camera frame to offset pixel coordinates.

        Args:
            rays: Direction vectors in the Camera frame, shape ``(..., 3)``.
                Need not be unit vectors; they will be normalised internally.

        Returns:
            A tuple ``(dx, dy)`` of pixel offsets from the principal point,
            each of shape ``(...,)``.  Points that cannot be projected (e.g.,
            rays pointing behind the camera) are returned as ``NaN``.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_normalise(v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalise an array of vectors to unit length.

        Zero-length vectors produce ``NaN`` rather than raising an exception,
        which preserves the shape contract for fully vectorised operations.

        Args:
            v: Array of shape ``(..., 3)``.

        Returns:
            Array of the same shape with each row normalised to ‖v‖ = 1.
        """
        norms = np.linalg.norm(v, axis=-1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(norms > 0, v / norms, np.nan)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy})"
        )


class Rectilinear(ProjectionModel):
    """Standard pinhole (rectilinear) projection model.

    The rectilinear model describes an ideal pinhole camera where straight
    lines in 3D space project to straight lines in the image.  It is the
    default model used in most computer-vision pipelines.

    Mathematical description
    ------------------------
    **Projection** (ray → pixel):

    Given a Camera-frame ray ``(Xc, Yc, Zc)`` with ``Zc > 0``:

    .. math::

        dx = f_x \\cdot \\frac{X_c}{Z_c}, \\quad
        dy = f_y \\cdot \\frac{Y_c}{Z_c}

    **Back-projection** (pixel → ray):

    .. math::

        \\mathbf{v} = \\frac{(dx/f_x,\\; dy/f_y,\\; 1)}{
            \\|(dx/f_x,\\; dy/f_y,\\; 1)\\|}

    Field of view is limited to a half-angle :math:`< 90°`; rays with
    ``Zc ≤ 0`` are invalid and produce ``NaN``.
    """

    def pixel_to_ray(
        self,
        dx: NDArray[np.float64],
        dy: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Back-project offset pixel coordinates to Camera-frame unit rays.

        Args:
            dx: Horizontal pixel offsets, shape ``(...,)``.
            dy: Vertical pixel offsets, shape ``(...,)``.

        Returns:
            Unit rays of shape ``(..., 3)`` in the Camera frame.
        """
        dx = np.asarray(dx, dtype=np.float64)
        dy = np.asarray(dy, dtype=np.float64)

        xn = dx / self.fx  # normalised image coordinates
        yn = dy / self.fy
        ones = np.ones_like(xn)

        rays = np.stack([xn, yn, ones], axis=-1)
        return self._safe_normalise(rays)

    def ray_to_pixel(
        self,
        rays: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Project Camera-frame rays to offset pixel coordinates.

        Args:
            rays: Direction vectors of shape ``(..., 3)``.

        Returns:
            Tuple ``(dx, dy)`` of pixel offsets, each shape ``(...,)``.
            Rays with ``z ≤ 0`` (behind the camera) yield ``NaN``.
        """
        rays = np.asarray(rays, dtype=np.float64)
        xc, yc, zc = rays[..., 0], rays[..., 1], rays[..., 2]

        with np.errstate(invalid="ignore", divide="ignore"):
            dx = np.where(zc > 0, self.fx * xc / zc, np.nan)
            dy = np.where(zc > 0, self.fy * yc / zc, np.nan)

        return dx, dy


class EquidistantFisheye(ProjectionModel):
    """Equidistant fisheye projection model (``r = f · θ``).

    The equidistant model maps the angle θ between a scene ray and the
    optical axis linearly to the radial distance ``r`` on the sensor.  It is
    the standard model for wide-angle and all-sky fisheye lenses.

    Mathematical description
    ------------------------
    **Projection** (ray → pixel):

    Given a Camera-frame ray ``(Xc, Yc, Zc)``:

    .. math::

        \\theta = \\arctan2\\!\\left(
            \\sqrt{X_c^2 + Y_c^2},\\; Z_c
        \\right) \\in [0,\\; \\pi]

        r = f_x \\cdot \\theta

        \\phi = \\arctan2(Y_c, X_c)

        dx = r \\cos\\phi = f_x \\cdot \\theta \\cdot
            \\frac{X_c}{\\sqrt{X_c^2 + Y_c^2}}

        dy = r \\sin\\phi = f_x \\cdot \\theta \\cdot
            \\frac{Y_c}{\\sqrt{X_c^2 + Y_c^2}}

    **Back-projection** (pixel → ray):

    .. math::

        r   = \\sqrt{dx^2 + dy^2}, \\quad
        \\theta = r / f_x

        X_c = \\sin\\theta \\cdot dx / r, \\quad
        Y_c = \\sin\\theta \\cdot dy / r, \\quad
        Z_c = \\cos\\theta

    This model accepts rays pointing in any hemisphere (``θ ∈ [0°, 180°]``),
    making it suitable for cameras with a field of view up to 360°.

    Note:
        The equidistant model uses a single effective focal length (``fx``).
        Non-square pixel effects should be corrected before calling this class.
    """

    def pixel_to_ray(
        self,
        dx: NDArray[np.float64],
        dy: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Back-project offset pixel coordinates to Camera-frame unit rays.

        Pixels at the principal point (``dx=0, dy=0``) map to the optical
        axis ``(0, 0, 1)`` by l'Hôpital's rule.

        Args:
            dx: Horizontal pixel offsets, shape ``(...,)``.
            dy: Vertical pixel offsets, shape ``(...,)``.

        Returns:
            Unit rays of shape ``(..., 3)`` in the Camera frame.
        """
        dx = np.asarray(dx, dtype=np.float64)
        dy = np.asarray(dy, dtype=np.float64)

        r = np.sqrt(dx**2 + dy**2)
        theta = r / self.fx  # angle from optical axis [radians]

        sin_theta = np.sin(theta)

        # Avoid division by zero at the principal point (r=0 → optical axis)
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(r > 0, sin_theta / r, 0.0)

        xc = scale * dx
        yc = scale * dy
        zc = np.cos(theta)

        return np.stack([xc, yc, zc], axis=-1)

    def ray_to_pixel(
        self,
        rays: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Project Camera-frame rays to offset pixel coordinates.

        The equidistant model can project rays from any direction (full
        sphere), so this method does not mark any ray as invalid solely on
        the basis of ``z < 0``.  Callers should apply their own sensor-bounds
        mask afterward.

        Args:
            rays: Direction vectors of shape ``(..., 3)``.  Need not be
                normalised.

        Returns:
            Tuple ``(dx, dy)`` of pixel offsets, each shape ``(...,)``.
        """
        rays = np.asarray(rays, dtype=np.float64)
        rays = self._safe_normalise(rays)

        xc, yc, zc = rays[..., 0], rays[..., 1], rays[..., 2]

        # θ: angle from +Z axis (optical axis)
        rho = np.sqrt(xc**2 + yc**2)
        theta = np.arctan2(rho, zc)  # ∈ [0, π]

        r = self.fx * theta  # radial pixel distance

        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(rho > 0, r / rho, 0.0)

        dx = scale * xc
        dy = scale * yc

        return dx, dy
