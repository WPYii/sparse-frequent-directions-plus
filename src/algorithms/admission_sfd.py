import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


class AdmissionSparseFrequentDirections:
    """
    Admission-controlled Sparse Frequent Directions.

    This class keeps the same external API as the existing FD/SFD classes:
        - __init__(l, ...)
        - fit(A)
        - process_row(row)
        - get_sketch()

    Design:
        1. Use the SAME SFD-style buffer trigger:
              buffer full if nnz(buffer) >= l * d  OR  rows(buffer) >= d
        2. When the buffer is full, first compute cheap block-level proxy scores.
        3. Randomly admit a subset of rows from the block.
        4. Run the original SFD sparse shrink only on admitted rows.

    Admission score modes:
        - "uniform": every row gets the same base score 1
        - "nnz":     score proportional to row.nnz
        - "l2":      score proportional to ||row||_2^2

    Notes:
        - This is intentionally lightweight. It does NOT implement true ridge leverage scores.
        - It is meant as a practical plug-in baseline for experiments.
    """

    def __init__(
        self,
        l: int,
        n_iter: int = 2,
        random_state: int | None = 0,
        admission_mode: str = "nnz",
        admission_scale: float = 1.0,
        min_prob: float = 0.0,
        max_prob: float = 1.0,
        rescale_admitted_rows: bool = True,
        min_admitted_rows: int = 1,
    ):
        if l <= 0:
            raise ValueError("Sketch size l must be positive.")
        if n_iter <= 0:
            raise ValueError("n_iter must be positive.")
        if admission_mode not in {"uniform", "nnz", "l2"}:
            raise ValueError("admission_mode must be one of {'uniform', 'nnz', 'l2'}.")
        if admission_scale <= 0:
            raise ValueError("admission_scale must be positive.")
        if not (0.0 <= min_prob <= 1.0):
            raise ValueError("min_prob must be in [0, 1].")
        if not (0.0 <= max_prob <= 1.0):
            raise ValueError("max_prob must be in [0, 1].")
        if min_prob > max_prob:
            raise ValueError("min_prob cannot exceed max_prob.")
        if min_admitted_rows < 0:
            raise ValueError("min_admitted_rows must be nonnegative.")

        self.l = l
        self.n_iter = n_iter
        self.rng = np.random.default_rng(random_state)

        self.admission_mode = admission_mode
        self.admission_scale = admission_scale
        self.min_prob = min_prob
        self.max_prob = max_prob
        self.rescale_admitted_rows = rescale_admitted_rows
        self.min_admitted_rows = min_admitted_rows

        self.B = None
        self.d = None
        self.buffer_rows: list[sp.csr_matrix] = []
        self.buffer_nnz = 0

        # Optional metadata for later inspection/debugging
        self.total_rows_seen = 0
        self.total_rows_admitted = 0
        self.total_blocks = 0
        self.total_rows_dropped = 0

    def fit(self, A):
        n, d = A.shape
        if self.B is None:
            self.B = np.zeros((self.l, d), dtype=float)
            self.d = d

        iterator = tqdm(range(n), desc="Running Admission SFD")
        for i in iterator:
            row = A.getrow(i) if sp.issparse(A) else sp.csr_matrix(A[i].reshape(1, -1))
            self.process_row(row)

        return self.get_sketch()

    def process_row(self, row):
        if self.B is None:
            d = row.shape[1]
            self.B = np.zeros((self.l, d), dtype=float)
            self.d = d

        if not sp.issparse(row):
            row = sp.csr_matrix(np.asarray(row).reshape(1, -1))
        else:
            row = row.tocsr()

        self.buffer_rows.append(row)
        self.buffer_nnz += row.nnz
        self.total_rows_seen += 1

        if self._buffer_is_full():
            self._flush_buffer()

    def get_sketch(self):
        if self.B is None:
            return None
        if self.buffer_rows:
            self._flush_buffer()
        return self.B

    def _buffer_is_full(self):
        if not self.buffer_rows:
            return False
        row_count = len(self.buffer_rows)
        return self.buffer_nnz >= self.l * self.d or row_count >= self.d

    def _flush_buffer(self):
        if not self.buffer_rows:
            return

        self.total_blocks += 1
        A_buffer = sp.vstack(self.buffer_rows, format="csr")

        A_adm = self._admit_block(A_buffer)

        # If nothing is admitted, just clear the buffer and keep current sketch.
        if A_adm.shape[0] == 0:
            self.buffer_rows.clear()
            self.buffer_nnz = 0
            return

        B_prime = self._sparse_shrink(A_adm)
        merged = np.vstack([self.B, B_prime])
        self.B = self._dense_shrink(merged)

        self.buffer_rows.clear()
        self.buffer_nnz = 0

    def _admit_block(self, A_block: sp.csr_matrix) -> sp.csr_matrix:
        """
        Compute cheap proxy scores for all rows in the current block,
        convert them to probabilities, then randomly admit rows.
        """
        m, d = A_block.shape
        if m == 0:
            return sp.csr_matrix((0, d), dtype=float)

        scores = self._score_block(A_block)
        probs = self._scores_to_probs(scores)

        keep = self.rng.random(m) < probs

        # Optionally force at least a few rows to survive if the block is nonempty.
        if keep.sum() < self.min_admitted_rows and m > 0 and self.min_admitted_rows > 0:
            k = min(self.min_admitted_rows, m)
            top_idx = np.argsort(-probs)[:k]
            keep[top_idx] = True

        admitted_count = int(keep.sum())
        self.total_rows_admitted += admitted_count
        self.total_rows_dropped += int(m - admitted_count)

        if admitted_count == 0:
            return sp.csr_matrix((0, d), dtype=float)

        A_keep = A_block[keep]

        if not self.rescale_admitted_rows:
            return A_keep

        # Importance-rescaling: divide each admitted row by sqrt(p_i)
        # to partially correct sampling bias.
        keep_probs = probs[keep]
        keep_probs = np.maximum(keep_probs, 1e-12)
        scale = 1.0 / np.sqrt(keep_probs)
        D = sp.diags(scale)
        return D @ A_keep

    def _score_block(self, A_block: sp.csr_matrix) -> np.ndarray:
        """
        Cheap proxy scores for each row.

        uniform:
            score_i = 1

        nnz:
            score_i = row.nnz

        l2:
            score_i = ||row||_2^2
        """
        m = A_block.shape[0]

        if self.admission_mode == "uniform":
            scores = np.ones(m, dtype=float)

        elif self.admission_mode == "nnz":
            # Number of nonzeros in each row
            row_nnz = np.diff(A_block.indptr)
            scores = row_nnz.astype(float)

        elif self.admission_mode == "l2":
            # Squared row norms
            scores = np.asarray(A_block.multiply(A_block).sum(axis=1)).ravel().astype(float)

        else:
            raise RuntimeError(f"Unsupported admission_mode: {self.admission_mode}")

        # Avoid all-zero score vector
        if not np.any(scores > 0):
            scores = np.ones(m, dtype=float)

        return scores

    def _scores_to_probs(self, scores: np.ndarray) -> np.ndarray:
        """
        Convert nonnegative scores into admission probabilities.

        We normalize by the block mean so that admission_scale is interpretable:
            prob_i = clip(admission_scale * score_i / mean(score), min_prob, max_prob)
        """
        scores = np.asarray(scores, dtype=float)
        mean_score = float(np.mean(scores))
        if mean_score <= 0:
            probs = np.full_like(scores, fill_value=self.max_prob, dtype=float)
        else:
            probs = self.admission_scale * (scores / mean_score)
            probs = np.clip(probs, self.min_prob, self.max_prob)

        return probs

    def _sparse_shrink(self, A_buffer: sp.csr_matrix):
        m, d = A_buffer.shape
        l_eff = min(self.l, m, d)
        if l_eff == 0:
            return np.zeros((self.l, d), dtype=float)

        Z = self._simultaneous_iteration(A_buffer, l_eff)
        P = Z.T @ A_buffer
        P = P.toarray() if sp.issparse(P) else np.asarray(P, dtype=float)

        _, s, vt = np.linalg.svd(P, full_matrices=False)
        delta = s[l_eff - 1] ** 2
        s_shrunk = np.sqrt(np.maximum(s[:l_eff] ** 2 - delta, 0.0))

        B_prime = np.zeros((self.l, d), dtype=float)
        B_prime[:l_eff, :] = np.diag(s_shrunk) @ vt[:l_eff, :]
        return B_prime

    def _dense_shrink(self, A_dense: np.ndarray):
        m, d = A_dense.shape
        l_eff = min(self.l, m, d)
        if l_eff == 0:
            return np.zeros((self.l, d), dtype=float)

        _, s, vt = np.linalg.svd(A_dense, full_matrices=False)
        delta = s[l_eff - 1] ** 2
        s_shrunk = np.sqrt(np.maximum(s[:l_eff] ** 2 - delta, 0.0))

        B = np.zeros((self.l, d), dtype=float)
        B[:l_eff, :] = np.diag(s_shrunk) @ vt[:l_eff, :]
        return B

    def _simultaneous_iteration(self, A: sp.csr_matrix, l_eff: int):
        m, d = A.shape
        G = self.rng.standard_normal(size=(d, l_eff))
        Y = A @ G
        Y = np.asarray(Y, dtype=float)

        for _ in range(self.n_iter):
            Y = A @ (A.T @ Y)
            Y = np.asarray(Y, dtype=float)

        Z, _ = np.linalg.qr(Y, mode="reduced")
        return Z

    def get_metadata(self) -> dict:
        return {
            "algorithm": "ASFD",
            "admission_mode": self.admission_mode,
            "admission_scale": self.admission_scale,
            "min_prob": self.min_prob,
            "max_prob": self.max_prob,
            "rescale_admitted_rows": self.rescale_admitted_rows,
            "total_rows_seen": self.total_rows_seen,
            "total_rows_admitted": self.total_rows_admitted,
            "total_rows_dropped": self.total_rows_dropped,
            "total_blocks": self.total_blocks,
            "admission_rate": (
                self.total_rows_admitted / self.total_rows_seen
                if self.total_rows_seen > 0 else np.nan
            ),
        }