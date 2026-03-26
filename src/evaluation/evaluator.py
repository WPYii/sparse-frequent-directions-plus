import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds


@dataclass
class EvaluationResult:
    sketch_size: int
    k: int
    runtime_sec: float
    projection_error: float
    covariance_error: float
    sketch: np.ndarray
    metadata: Optional[Dict[str, Any]] = None


class SketchEvaluator:
    def __init__(self, A, k: int = 10):
        self.A = A
        self.k = k

    def evaluate(self, model):
        start = time.perf_counter()
        B = model.fit(self.A)
        runtime_sec = time.perf_counter() - start

        proj_err = self.projection_error(B)
        cov_err = self.covariance_error(B)

        return EvaluationResult(
            sketch_size=model.l,
            k=self.k,
            runtime_sec=runtime_sec,
            projection_error=proj_err,
            covariance_error=cov_err,
            sketch=B,
            metadata={"model_name": model.__class__.__name__},
        )

    def projection_error(self, B: np.ndarray):
        n, d = self.A.shape
        k_eff = min(self.k, B.shape[0], n, d)

        _, _, vt_B = np.linalg.svd(B, full_matrices=False)
        V_Bk = vt_B[:k_eff].T  

        total_energy = self.frobenius_norm_squared(self.A)
        AV = self.A @ V_Bk
        projected_energy = np.sum(AV**2)
        numerator = max(0, total_energy - projected_energy)

        denominator = total_energy - self.top_k_energy(self.A, k_eff)
        return float(numerator / denominator) if denominator > 0 else np.nan

    def covariance_error(self, B: np.ndarray):
        if sp.issparse(self.A):
            AtA = (self.A.T @ self.A).toarray()
        else:
            AtA = self.A.T @ self.A

        BtB = B.T @ B
        diff = AtA - BtB

        eigvals = np.linalg.eigvalsh(diff)
        spectral_norm = np.max(np.abs(eigvals))

        denominator = self.frobenius_norm_squared(self.A)
        if denominator <= 0:
            return np.nan

        return float(spectral_norm / denominator)

    @staticmethod
    def frobenius_norm_squared(A):
        if sp.issparse(A):
            return float(A.multiply(A).sum())
        return float(np.sum(A * A))

    @staticmethod
    def top_k_energy(A, k: int):
        if sp.issparse(A):
            _, s, _ = svds(A, k=k)
            s = np.sort(s)[::-1]
        else:
            _, s, _ = np.linalg.svd(A, full_matrices=False)
            s = s[:k]

        return float(np.sum(s ** 2))