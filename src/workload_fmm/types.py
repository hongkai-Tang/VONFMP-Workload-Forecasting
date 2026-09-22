from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

DEFAULT_RESOURCE_NAMES = ("cpu", "gpu", "mem", "read", "write")
RESOURCE_NAMES = DEFAULT_RESOURCE_NAMES
RESOURCE_DIM = len(DEFAULT_RESOURCE_NAMES)


@dataclass
class ShapePrototype:
    """A DTW-medoid resource pattern prototype."""

    id: int
    values: np.ndarray
    sigma: float
    medoid_source_index: int | None = None
    coverage_radius: float | None = None

    @property
    def length(self) -> int:
        return int(self.values.shape[0])

    @property
    def resource_dim(self) -> int:
        return int(self.values.shape[1])


@dataclass
class ShapeSet:
    """A reproducible fuzzy-state shape set S."""

    prototypes: list[ShapePrototype]
    coverage: float
    epsilon: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_states(self) -> int:
        return len(self.prototypes)

    def lengths(self) -> list[int]:
        return [p.length for p in self.prototypes]

    def sigmas(self) -> np.ndarray:
        return np.asarray([p.sigma for p in self.prototypes], dtype=float)
