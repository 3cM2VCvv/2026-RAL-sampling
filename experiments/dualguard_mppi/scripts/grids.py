"""O(1) nearest-neighbour lookup on a uniform, optionally periodic grid.

Both BRT exports sit on uniform axes, so the nearest cell is
``round((x - origin) / dx)``: one gather instead of an ``argmin`` scan over every
axis, evaluated 1000 x 100 times per control step.  On a uniform grid the two return
the identical cell, so this is a speed change only.

Out-of-bounds handling follows the paper repo's ``src/dubins_environment.py``, which
builds its interpolators with ``bounds_error=False, fill_value=-1.0``: pass
``fill=OOB_FILL`` and a state outside the grid reads as unsafe.  Control tables are
gathered with ``fill=None`` instead and clamp to the edge cell, because there is no
meaningful fill value for a control -- the ``oob`` flag is what marks those reads as
uncertified.
"""

import jax.numpy as jnp
import numpy as np

OOB_FILL = -1.0


class NearestGrid:
    def __init__(self, axes, periodic_dim=None):
        axes = [np.asarray(a, dtype=np.float64).squeeze() for a in axes]
        for a in axes:
            d = np.diff(a)
            assert a.ndim == 1 and a.size >= 2 and np.allclose(d, d[0], rtol=1e-9), (
                "axis is not 1-D and uniformly spaced; the index arithmetic assumes it is"
            )
        self.ndim = len(axes)
        self.shape = tuple(a.size for a in axes)
        self.axes = axes

        n = np.array(self.shape, dtype=np.int32)
        strides = np.ones(self.ndim, dtype=np.int32)
        for i in range(self.ndim - 2, -1, -1):
            strides[i] = strides[i + 1] * n[i + 1]
        check = np.ones(self.ndim, dtype=bool)
        if periodic_dim is not None:
            check[periodic_dim] = False

        self.periodic_dim = periodic_dim
        dx = np.array([a[1] - a[0] for a in axes], dtype=np.float32)
        self.cell = np.asarray(dx, dtype=np.float64)
        self.span = (
            None
            if periodic_dim is None
            else float(dx[periodic_dim]) * (int(n[periodic_dim]) - 1)
        )

        self._n = jnp.asarray(n)
        self._origin = jnp.asarray([a[0] for a in axes], dtype=jnp.float32)
        self._dx = jnp.asarray(dx)
        self._hi = jnp.asarray([a[-1] for a in axes], dtype=jnp.float32)
        self._strides = jnp.asarray(strides)
        self._check = jnp.asarray(check)

    def locate(self, x):
        """x: (ndim,) -> (flat C-order index, out-of-bounds flag)."""
        p = self.periodic_dim
        if p is not None:
            x = x.at[p].set(
                self._origin[p] + jnp.mod(x[p] - self._origin[p], self.span)
            )
        oob = jnp.any(
            jnp.where(
                self._check, (x < self._origin - 1e-6) | (x > self._hi + 1e-6), False
            )
        )
        idx = jnp.clip(
            jnp.round((x - self._origin) / self._dx).astype(jnp.int32), 0, self._n - 1
        )
        return jnp.sum(idx * self._strides), oob

    def gather(self, flat_table, x, fill=None):
        flat, oob = self.locate(x)
        v = flat_table[flat]
        return v if fill is None else jnp.where(oob, fill, v)

    @staticmethod
    def flatten(array):
        """Table laid out to match :meth:`locate`'s strides, on device."""
        return jnp.asarray(np.ascontiguousarray(array).reshape(-1))
