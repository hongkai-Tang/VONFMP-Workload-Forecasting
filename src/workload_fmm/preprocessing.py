from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import default_mask, ensure_3d_resource_tensor


@dataclass
class ResourcePreprocessor:
    """Apply optional log compression and train-only z-score normalization."""

    log_indices: tuple[int, ...] = ()
    eps: float = 1e-8
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def _log_transform(self, values: np.ndarray) -> np.ndarray:
        out = np.array(values, dtype=float, copy=True)
        for idx in self.log_indices:
            if 0 <= idx < out.shape[-1]:
                out[..., idx] = np.log1p(np.maximum(out[..., idx], 0.0))
        return out

    def fit(self, values: np.ndarray, mask: np.ndarray | None = None) -> "ResourcePreprocessor":
        arr = ensure_3d_resource_tensor(values)
        active = default_mask(arr, mask)
        transformed = self._log_transform(arr)
        flat = transformed[active]
        if flat.size == 0:
            raise ValueError("no active samples available to fit preprocessing statistics")
        self.mean_ = np.nanmean(flat, axis=0)
        self.std_ = np.nanstd(flat, axis=0)
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("ResourcePreprocessor.fit must be called before transform")
        arr = ensure_3d_resource_tensor(values, len(self.mean_))
        transformed = self._log_transform(arr)
        return (transformed - self.mean_) / (self.std_ + self.eps)

    def fit_transform(self, values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        return self.fit(values, mask).transform(values)
