"""Prepared sensing geometry and four-frequency measurement inputs."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data"


@dataclass(frozen=True)
class Measurement:
    case: str
    geometry: str
    sensitivity: np.ndarray
    eit2rect: np.ndarray
    rect2eit: np.ndarray
    positions: np.ndarray
    voltages: np.ndarray
    frequency_labels: tuple

    def to_images(self, coefficients):
        """Use the authors' one-based position table for display coordinates."""
        coefficients = np.asarray(coefficients, dtype=np.float32)
        if coefficients.shape[-1] != self.sensitivity.shape[1]:
            raise ValueError("Coefficient count must match the sensing mesh")
        images = np.zeros(coefficients.shape[:-1] + (64, 64), dtype=np.float32)
        row, column = self.positions[:, 2:4].T - 1
        images[..., row, column] = coefficients
        return images

    @property
    def display_mask(self):
        return self.to_images(np.ones(self.sensitivity.shape[1], np.float32)).astype(bool)


def available_cases(data_dir=DEFAULT_DATA):
    return json.loads((Path(data_dir) / "manifest.json").read_text(encoding="utf-8"))["cases"]


def load_case(case="case2_sim", data_dir=DEFAULT_DATA):
    directory = Path(data_dir)
    cases = available_cases(directory)
    if case not in cases:
        raise ValueError(f"Unknown case {case!r}; choose from {', '.join(cases)}")
    entry = cases[case]
    geometry = entry["geometry"]
    with np.load(directory / f"{geometry}.npz", allow_pickle=False) as arrays:
        sensitivity = arrays["sensitivity"].copy()
        eit2rect = arrays["eit2rect"].astype(np.float32, copy=True)
        rect2eit = arrays["rect2eit"].astype(np.float32, copy=True)
        positions = arrays["positions"].astype(np.int32, copy=True)
    with np.load(directory / "voltages.npz", allow_pickle=False) as arrays:
        voltages = arrays[case].astype(np.float32, copy=True)
    n = sensitivity.shape[1]
    if sensitivity.ndim != 2 or voltages.shape != (sensitivity.shape[0], 4):
        raise ValueError("Expected aligned measurements with four frequency columns")
    if eit2rect.shape != (4096, n) or rect2eit.shape != (n, 4096):
        raise ValueError("Projection dimensions do not match the sensing mesh")
    if positions.shape != (n, 4) or np.any(positions[:, 2:4] < 1) or np.any(positions[:, 2:4] > 64):
        raise ValueError("Position table must map to a 64 x 64 image")
    if not all(np.isfinite(a).all() for a in (sensitivity, eit2rect, rect2eit, voltages)):
        raise ValueError("Nonfinite values in the reconstruction inputs")
    return Measurement(case, geometry, sensitivity, eit2rect, rect2eit, positions, voltages, tuple(entry["frequency_labels"]))


def normalize_sensitivity(sensitivity):
    """Retain both normalization steps and dtypes of the archived inference."""
    matrix = np.array(sensitivity, dtype=np.float64, copy=True)
    for row in range(matrix.shape[0]):
        denominator = np.sum(matrix[row])
        if not np.isfinite(denominator) or denominator == 0:
            raise ValueError("Sensitivity row sum must be nonzero and finite")
        matrix[row] = matrix[row] / denominator
    matrix = matrix.astype(np.float32)
    row_sums = np.sum(matrix, axis=1)
    if np.any(row_sums == 0):
        raise ValueError("Sensitivity normalization produced a zero row sum")
    return (matrix.T / row_sums).T
