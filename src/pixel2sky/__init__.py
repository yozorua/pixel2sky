"""pixel2sky: Bidirectional pixel ↔ sky coordinate transformation library.

This package provides production-grade transformations between 2D image pixel
coordinates and 3D local horizontal sky coordinates (Altitude, Azimuth). It
accounts for both camera intrinsic parameters (lens geometry / projection
model) and extrinsic parameters (pointing direction and sensor orientation).

Typical usage::

    import numpy as np
    from pixel2sky import SkyMapper
    from pixel2sky.projection import EquidistantFisheye

    mapper = SkyMapper(
        image_width=4096,
        image_height=3072,
        projection=EquidistantFisheye(focal_length=800.0),
        az0=180.0,
        alt0=45.0,
        roll=0.0,
    )

    # Transform a grid of pixels to Alt/Az
    x = np.arange(4096)
    y = np.arange(3072)
    xx, yy = np.meshgrid(x, y)
    alt, az = mapper.pixel_to_altaz(xx.ravel(), yy.ravel())

    # Round-trip back to pixels
    xp, yp = mapper.altaz_to_pixel(alt, az)
"""

from pixel2sky._version import __version__
from pixel2sky.mapper import SkyMapper
from pixel2sky.projection import (
    EquidistantFisheye,
    ProjectionModel,
    Rectilinear,
    StereographicFisheye,
)

__all__ = [
    "__version__",
    "SkyMapper",
    "ProjectionModel",
    "Rectilinear",
    "EquidistantFisheye",
    "StereographicFisheye",
]
