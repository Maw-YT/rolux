"""CPU fallback for depth→normals (GPU path in shader_worker is preferred)."""

from __future__ import annotations

import numpy as np


def depth_to_normals(depth: np.ndarray, *, inv_focal: float = 1.35) -> np.ndarray:
    """View-space normals from depth; BGR uint8 with RGB = 0.5+0.5*n after upload."""
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim > 2:
        d = d[..., 0]
    dmin, dmax = float(d.min()), float(d.max())
    d = (d - dmin) / max(dmax - dmin, 1e-6)
    z = 0.35 + d * (1.75 - 0.35)
    h, w = d.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    ndc_x = xs / max(w - 1, 1) * 2.0 - 1.0
    ndc_y = ys / max(h - 1, 1) * 2.0 - 1.0
    px = ndc_x * z * inv_focal
    py = ndc_y * z * inv_focal
    # Central differences on view positions.
    dpdx = np.stack(
        [np.gradient(px, axis=1), np.gradient(py, axis=1), np.gradient(z, axis=1)],
        axis=-1,
    )
    dpdy = np.stack(
        [np.gradient(px, axis=0), np.gradient(py, axis=0), np.gradient(z, axis=0)],
        axis=-1,
    )
    n = np.cross(dpdx, dpdy)
    # Face camera (-Z).
    flip = n[..., 2] > 0
    n[flip] *= -1
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(norm, 1e-6)
    rgb = ((n * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1])
