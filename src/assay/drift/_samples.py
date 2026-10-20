"""Input coercion shared by the drift statistics: turn array-likes into checked float vectors."""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def as_sample(values: npt.ArrayLike, *, name: str, allow_empty: bool = False) -> FloatArray:
    """Return ``values`` as a one-dimensional float64 array of finite numbers.

    Raises ``ValueError`` if the input is not one-dimensional, contains NaN or infinity,
    or is empty while ``allow_empty`` is false. ``name`` is used in error messages.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.ndim} dimensions")
    if array.size == 0 and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array
