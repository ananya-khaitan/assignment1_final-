"""Train-window-only VMD/CEEMDAN helpers and ApEn routing diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

try:
    from vmdpy import VMD  # type: ignore
except ImportError:  # pragma: no cover - verified by runtime preflight
    VMD = None

try:
    from PyEMD import CEEMDAN  # type: ignore
except ImportError:  # pragma: no cover - verified by runtime preflight
    CEEMDAN = None


class CausalComponentHistory:
    """Maintain decomposed component histories between scheduled re-fits.

    VMD and CEEMDAN are re-estimated at a documented scheduled fit origin,
    rather than re-estimated with every one-step forecast.  When the next
    realized close becomes known, it is an admissible input at the following
    origin.  This class appends that observed innovation to the smoothest
    component and carries the other component end-states forward.  The result
    exactly reconstructs each newly observed price, contains no future value,
    and gives fixed-parameter component models an updated causal history.

    It is intentionally not presented as a new VMD/CEEMDAN decomposition.  A
    full decomposition is performed again only at the next scheduled fit.
    """

    def __init__(self, modes: np.ndarray, observed: np.ndarray) -> None:
        array = np.asarray(modes, dtype=float)
        values = np.asarray(observed, dtype=float)
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError("Component history requires a non-empty two-dimensional mode array")
        if values.ndim != 1 or len(values) != array.shape[1]:
            raise ValueError("Component history must align exactly with its observed price history")
        if not np.isfinite(array).all() or not np.isfinite(values).all():
            raise ValueError("Component history requires finite values")
        self._modes = array.copy()
        self._observed = values.copy()
        # The component with the smallest total variation is the appropriate
        # level/residual carrier.  This accommodates VMD's low-frequency-first
        # convention and CEEMDAN implementations that place the residue last.
        variation = np.mean(np.abs(np.diff(self._modes, axis=1)), axis=1)
        self.level_component = int(np.argmin(variation))

    @property
    def modes(self) -> np.ndarray:
        """Current components in the same order as the fit decomposition."""
        return self._modes

    @property
    def observed_size(self) -> int:
        return len(self._observed)

    def advance(self, observed: np.ndarray) -> np.ndarray:
        """Ingest only newly observed historical closes and return components.

        ``observed`` must be the current forecast origin's history.  Its
        existing prefix is verified so a caller cannot silently mutate a past
        fit window.  No target beyond this history can influence the state.
        """
        values = np.asarray(observed, dtype=float)
        if values.ndim != 1 or len(values) < len(self._observed):
            raise ValueError("Updated history must retain the complete fitted history")
        if not np.isfinite(values).all():
            raise ValueError("Updated history contains non-finite values")
        if not np.allclose(values[: len(self._observed)], self._observed, rtol=1e-10, atol=1e-10):
            raise ValueError("Updated history changes values that were already observed at the fit origin")

        for next_value in values[len(self._observed) :]:
            innovation = float(next_value - self._observed[-1])
            # A continuation starts each component at its known last state;
            # allocating the observed innovation to the level carrier keeps
            # the component sum exactly equal to the newly observed close.
            self._modes = np.concatenate([self._modes, self._modes[:, -1:]], axis=1)
            self._modes[self.level_component, -1] += innovation
            self._observed = np.append(self._observed, next_value)

        return self._modes


def _sort_modes(modes: np.ndarray, omega: np.ndarray) -> np.ndarray:
    if omega.ndim == 2:
        omega = omega[-1]
    return modes[np.argsort(omega), :]


def approximate_entropy(signal: np.ndarray, m: int = 2, r_coef: float = 0.2) -> float:
    """Compute exact ApEn without materializing an O(n²) distance tensor.

    ApEn counts Chebyshev-neighbouring delay vectors. The former dense
    implementation allocated every pairwise vector distance at each scheduled
    refit, which made the full eight-company causal run impractically slow.
    ``cKDTree.query_ball_point(..., p=inf)`` returns the same inclusive counts
    (including a vector's self-match) using a spatial index.
    """
    x = np.asarray(signal, dtype=float)
    if len(x) <= m + 2:
        return float("nan")
    tolerance = float(r_coef * np.std(x))
    if tolerance <= 0:
        return 0.0

    def phi(order: int) -> float:
        patterns = np.array([x[i : i + order] for i in range(len(x) - order + 1)])
        tree = cKDTree(patterns)
        counts = np.asarray(tree.query_ball_point(patterns, r=tolerance, p=np.inf, return_length=True), dtype=float)
        probabilities = counts / len(patterns)
        return float(np.mean(np.log(np.maximum(probabilities, 1e-12))))

    return float(phi(m) - phi(m + 1))


def vmd_decompose_train_only(
    signal: np.ndarray,
    alpha: float,
    tau: float,
    k: int,
    dc: int,
    init: int,
    tol: float,
    *,
    allow_approximation: bool = False,
) -> np.ndarray:
    """Fit VMD using only the supplied historical training slice.

    A reported VMD experiment must have ``vmdpy`` installed.  The former
    moving-average substitute is available only when explicitly requested for
    a smoke test and is never returned under a VMD label by the runner.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or len(signal) < max(20, k * 3):
        raise ValueError("VMD requires a one-dimensional training series with sufficient observations")
    if VMD is None:
        if not allow_approximation:
            raise ImportError("vmdpy is required for VMD experiments; no substitute is used in reported runs")
        windows = np.linspace(5, max(7, len(signal) // 6), num=max(2, k), dtype=int)
        residual = signal.copy()
        modes: list[np.ndarray] = []
        for window in windows[:-1]:
            # This explicit smoke-test approximation is not a causal VMD result.
            trend = np.convolve(residual, np.ones(max(3, int(window))) / max(3, int(window)), mode="same")
            modes.append(residual - trend)
            residual = trend
        modes.append(residual)
        return np.asarray(modes[:k], dtype=float)

    mean = float(np.mean(signal))
    std = float(np.std(signal) + 1e-9)
    standardized = (signal - mean) / std
    modes, _, omega = VMD(standardized, alpha, tau, k, dc, init, tol)
    if modes.ndim == 3:
        modes = modes[-1]
    restored = _sort_modes(np.asarray(modes, dtype=float), np.asarray(omega)) * std
    # Some vmdpy releases return an even-length reconstruction for an odd
    # input.  Keep the component time axis exactly aligned with the supplied
    # historical slice; edge padding uses no future observation.
    if restored.shape[1] < len(signal):
        restored = np.pad(restored, ((0, 0), (0, len(signal) - restored.shape[1])), mode="edge")
    elif restored.shape[1] > len(signal):
        restored = restored[:, : len(signal)]
    # VMD is fit to demeaned standardized data. Put the level back into the
    # lowest-frequency component so summed component forecasts remain prices.
    restored[0] = restored[0] + mean
    # Enforce exact reconstruction after numerical/edge adjustments while
    # retaining the correction in the low-frequency price-level component.
    restored[0] = restored[0] + (signal - restored.sum(axis=0))
    return restored


def ceemdan_decompose_train_only(signal: np.ndarray, trials: int = 20, seed: int = 42) -> np.ndarray:
    """Fit CEEMDAN using only the supplied historical training slice."""
    if CEEMDAN is None:
        raise ImportError("EMD-signal/PyEMD is required for CEEMDAN experiments")
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or len(signal) < 20:
        raise ValueError("CEEMDAN requires a one-dimensional training series with sufficient observations")
    decomposer = CEEMDAN(trials=trials, parallel=False)
    # PyEMD exposes noise_seed; preserve compatibility with older releases.
    if hasattr(decomposer, "noise_seed"):
        decomposer.noise_seed(seed)
    modes = decomposer.ceemdan(signal)
    if modes is None or len(modes) == 0:
        raise RuntimeError("CEEMDAN returned no intrinsic mode functions")
    return np.asarray(modes, dtype=float)
