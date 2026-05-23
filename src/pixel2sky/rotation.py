"""Camera extrinsic rotation: World ↔ Camera frame conversions.

This module implements the coordinate-frame rotation that bridges the
**World frame** (local horizontal / Alt-Az) and the **Camera frame**
(image sensor axes).

Coordinate-frame definitions
-----------------------------
**World frame** (local horizontal, right-handed, ENU convention)

  * +X  →  East
  * +Y  →  North
  * +Z  →  Zenith

**Camera frame** (right-handed, aligned with the sensor)

  * +X  →  right  (increasing pixel column)
  * +Y  →  down   (increasing pixel row)
  * +Z  →  into the scene (optical axis direction)

Rotation decomposition
----------------------
The extrinsic rotation ``R`` maps a World-frame unit vector to a
Camera-frame unit vector:

    v_cam = R · v_world

It is built by composing three elementary rotations applied in order:

1. **Azimuth rotation** ``R_az`` — rotate about the World +Z axis by
   ``−az0`` so that the camera boresight points north (→ intermediate
   frame whose +Z points toward the desired azimuth at the horizon).

2. **Altitude rotation** ``R_alt`` — tilt up from the horizon by ``alt0``.
   This rotates about the intermediate-frame +X axis (East direction after
   step 1).

3. **Roll rotation** ``R_roll`` — rotate the sensor about the optical axis
   by ``−roll`` to account for camera rotation relative to the horizon.

4. **Axis permutation** ``R_perm`` — re-label axes to match the Camera-frame
   convention (+Z forward, +Y down).

The combined rotation is:

    R = R_perm · R_roll · R_alt · R_az

All angle arguments are in **degrees** at the public API boundary;
internal computations use radians.

Note on sign/handedness
-----------------------
Azimuth is measured clockwise from North (as is standard in astronomy and
navigation).  The right-handed World-frame X=East, Y=North, Z=Zenith means
a clockwise rotation of ``az0`` corresponds to a negative rotation about +Z
in mathematical (counterclockwise-positive) convention.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


def build_rotation(
    az0: float,
    alt0: float,
    roll: float,
) -> Rotation:
    """Build the World → Camera rotation from pointing parameters.

    The returned :class:`scipy.spatial.transform.Rotation` object ``R``
    satisfies:

        v_cam = R.apply(v_world)

    The rotation matrix is constructed directly from the Camera axes expressed
    in the World (ENU) frame, avoiding gimbal-lock-prone Euler-angle sequences.

    Camera-axis definitions (roll = 0):

    * **Camera +X** (image right): tangent to the iso-altitude circle through
      (az0, alt0), pointing toward increasing azimuth.
    * **Camera +Y** (image down): tangent to the meridian through (az0, alt0),
      pointing toward decreasing altitude.
    * **Camera +Z** (boresight/optical axis): unit vector toward (az0, alt0).

    Roll rotates the sensor about the optical axis, carrying +X and +Y with it.

    Args:
        az0: Azimuth of the camera boresight, in degrees.  Measured
            clockwise from North (0 = North, 90 = East, 180 = South,
            270 = West).
        alt0: Altitude (elevation) of the camera boresight, in degrees.
            0 = horizon, 90 = zenith, negative values below the horizon.
        roll: Clockwise rotation of the sensor about the optical axis,
            in degrees (as viewed from behind the camera).  0 means the
            top of the image faces the zenith (or the horizon for a
            horizontally pointing camera).

    Returns:
        A :class:`~scipy.spatial.transform.Rotation` representing the
        composed extrinsic rotation.

    Raises:
        ValueError: If ``alt0`` is outside ``[−90, 90]`` degrees.

    Examples:
        >>> R = build_rotation(az0=0.0, alt0=90.0, roll=0.0)
        >>> v_world = np.array([0.0, 0.0, 1.0])   # zenith
        >>> v_cam = R.apply(v_world)
        >>> np.allclose(v_cam, [0.0, 0.0, 1.0])   # boresight → zenith
        True
    """
    if not -90.0 <= alt0 <= 90.0:
        raise ValueError(
            f"alt0 must be in [-90, 90] degrees, got {alt0!r}"
        )

    az_r = np.deg2rad(az0)
    alt_r = np.deg2rad(alt0)
    roll_r = np.deg2rad(roll)

    ca, sa = np.cos(az_r), np.sin(az_r)
    ce, se = np.cos(alt_r), np.sin(alt_r)
    cr, sr = np.cos(roll_r), np.sin(roll_r)

    # Camera axes in World (ENU) frame when roll = 0.
    # These form an orthonormal right-handed basis: x0 × y0 = z0.
    x0 = np.array([ca,      -sa,      0.0])  # image right  (∂boresight/∂az, normalised)
    y0 = np.array([se * sa,  se * ca, -ce])  # image down   (−∂boresight/∂alt)
    z0 = np.array([ce * sa,  ce * ca,  se])  # boresight    (toward az0, alt0)

    # Apply clockwise sensor roll about z0.
    # A CW roll carries x0 toward y0:  x' = cos(r)·x0 + sin(r)·y0
    #                                   y' = −sin(r)·x0 + cos(r)·y0
    x_cam = cr * x0 + sr * y0
    y_cam = -sr * x0 + cr * y0

    # Build rotation matrix with rows = Camera axes expressed in World frame.
    # R @ v_world = v_cam  (maps World vectors into Camera frame)
    R_mat = np.stack([x_cam, y_cam, z0], axis=0)

    return Rotation.from_matrix(R_mat)


def world_to_camera(
    v_world: NDArray[np.float64],
    R: Rotation,
) -> NDArray[np.float64]:
    """Rotate World-frame vectors into the Camera frame.

    Args:
        v_world: Array of shape ``(..., 3)`` with unit vectors in the World
            frame ``(East, North, Zenith)``.
        R: The World → Camera rotation returned by :func:`build_rotation`.

    Returns:
        Array of shape ``(..., 3)`` with the same vectors expressed in the
        Camera frame ``(right, down, forward)``.
    """
    return R.apply(np.asarray(v_world, dtype=np.float64))


def camera_to_world(
    v_cam: NDArray[np.float64],
    R: Rotation,
) -> NDArray[np.float64]:
    """Rotate Camera-frame vectors into the World frame.

    Args:
        v_cam: Array of shape ``(..., 3)`` with unit vectors in the Camera
            frame ``(right, down, forward)``.
        R: The World → Camera rotation returned by :func:`build_rotation`.
            Its inverse is used automatically.

    Returns:
        Array of shape ``(..., 3)`` with the same vectors expressed in the
        World frame ``(East, North, Zenith)``.
    """
    return R.inv().apply(np.asarray(v_cam, dtype=np.float64))


def altaz_to_world_vector(
    alt: NDArray[np.float64],
    az: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert Alt/Az sky coordinates to World-frame unit vectors.

    Uses the local horizontal (ENU) convention:

    .. math::

        X &= \\cos(alt)\\sin(az)  \\quad (\\text{East}) \\\\
        Y &= \\cos(alt)\\cos(az)  \\quad (\\text{North}) \\\\
        Z &= \\sin(alt)           \\quad (\\text{Zenith})

    Args:
        alt: Altitude angles in degrees, shape ``(...,)``.
            0 = horizon, 90 = zenith.
        az: Azimuth angles in degrees, shape ``(...,)``.
            0 = North, 90 = East (clockwise from North).

    Returns:
        Unit vectors of shape ``(..., 3)`` in the World frame.
    """
    alt_r = np.deg2rad(np.asarray(alt, dtype=np.float64))
    az_r = np.deg2rad(np.asarray(az, dtype=np.float64))

    cos_alt = np.cos(alt_r)
    x = cos_alt * np.sin(az_r)  # East
    y = cos_alt * np.cos(az_r)  # North
    z = np.sin(alt_r)           # Zenith

    return np.stack([x, y, z], axis=-1)


def world_vector_to_altaz(
    v: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert World-frame unit vectors to Alt/Az sky coordinates.

    Args:
        v: Array of shape ``(..., 3)`` in the World frame
            ``(East, North, Zenith)``.  Need not be normalised.

    Returns:
        Tuple ``(alt, az)`` where:

        * ``alt`` – Altitude in degrees ``∈ [−90, 90]``.
        * ``az``  – Azimuth in degrees ``∈ [0, 360)``, measured clockwise
          from North.
    """
    v = np.asarray(v, dtype=np.float64)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]

    norms = np.linalg.norm(v, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        x_n = np.where(norms > 0, x / norms, np.nan)
        y_n = np.where(norms > 0, y / norms, np.nan)
        z_n = np.where(norms > 0, z / norms, np.nan)

    alt = np.rad2deg(np.arcsin(np.clip(z_n, -1.0, 1.0)))
    az = np.rad2deg(np.arctan2(x_n, y_n)) % 360.0  # clockwise from North

    return alt, az
