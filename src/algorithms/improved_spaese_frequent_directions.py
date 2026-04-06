import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

class ImprovedSparseFrequentDirections:
    def __init__(
        self,
        l: int,
        n_iter: int = 2,
        random_state: int | None = 0,
        use_approx_p_svd: bool = False,
        use_ors_admission: bool = False,
        candidate_block_size: int = 256,
        sample_epsilon: float = 0.5,
        min_prob: float = 0.0,
        max_prob: float = 1.0,
        rescale_admitted_rows: bool = True,
        ridge_lambda: float = 1.0,
    ):
        if l <= 0:
            raise ValueError("Sketch size l must be positive.")
        if n_iter <= 0:
            raise ValueError("n_iter must be positive.")
        if candidate_block_size <= 0:
            raise ValueError("candidate_block_size must be positive.")
        if not (0.0 < sample_epsilon < 1.0):
            raise ValueError("sample_epsilon must be in (0, 1).")
        if not (0.0 <= min_prob <= 1.0):
            raise ValueError("min_prob must be in [0, 1].")
        if not (0.0 <= max_prob <= 1.0):
            raise ValueError("max_prob must be in [0, 1].")
        if min_prob > max_prob:
            raise ValueError("min_prob cannot exceed max_prob.")
        if ridge_lambda <= 0:
            raise ValueError("ridge_lambda must be positive.")

        self.l = l
        self.n_iter = n_iter
        self.use_approx_p_svd = use_approx_p_svd
        self.use_ors_admission = use_ors_admission
        self.candidate_block_size = candidate_block_size
        self.sample_epsilon = sample_epsilon
        self.min_prob = min_prob
        self.max_prob = max_prob
        self.rescale_admitted_rows = rescale_admitted_rows
        self.ridge_lambda = ridge_lambda

        self.rng = np.random.default_rng(random_state)

        self.B = None
        self.d = None

        self.candidate_rows: list[sp.csr_matrix] = []
        self.candidate_nnz = 0
        self.buffer_rows: list[sp.csr_matrix] = []
        self.buffer_nnz = 0

    def fit(self, A):
        n, d = A.shape

        if self.B is None:
            self.B = np.zeros((self.l, d), dtype=float)
            self.d = d

        iterator = tqdm(range(n), desc="Running Improved Sparse Frequent Directions")
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

        if self.use_ors_admission:
            self.candidate_rows.append(row)
            self.candidate_nnz += row.nnz

            if self._candidate_block_is_full():
                self._process_candidate_block()
            return

        self.buffer_rows.append(row)
        self.buffer_nnz += row.nnz

        if self._buffer_is_full():
            self._flush_buffer()

    def get_sketch(self):
        if self.B is None:
            return None

        if self.candidate_rows:
            self._process_candidate_block()

        if self.buffer_rows:
            self._flush_buffer()

        return self.B

    def _candidate_block_is_full(self):
        if not self.candidate_rows:
            return False
        return len(self.candidate_rows) >= self.candidate_block_size

    def _buffer_is_full(self):
        if not self.buffer_rows:
            return False

        row_count = len(self.buffer_rows)
        return self.buffer_nnz >= self.l * self.d or row_count >= self.d

    def _process_candidate_block(self):
        if not self.candidate_rows:
            return

        a_candidates = sp.vstack(self.candidate_rows, format="csr")
        scores = self._online_ridge_scores_batch(a_candidates)
        probs = self._scores_to_probs(scores)

        for row, prob in zip(self.candidate_rows, probs):
            admitted_row = self._admit_row_with_prob(row, float(prob))
            if admitted_row is None:
                continue

            self.buffer_rows.append(admitted_row)
            self.buffer_nnz += admitted_row.nnz

            if self._buffer_is_full():
                self._flush_buffer()

        self.candidate_rows.clear()
        self.candidate_nnz = 0

    def _flush_buffer(self):
        if not self.buffer_rows:
            return

        A_buffer = sp.vstack(self.buffer_rows, format="csr")
        B_prime = self._sparse_shrink(A_buffer)

        merged = np.vstack([self.B, B_prime])
        self.B = self._dense_shrink_fast(merged)

        self.buffer_rows.clear()
        self.buffer_nnz = 0

    def _admit_row_with_prob(self, row: sp.csr_matrix, prob: float):
        keep = self.rng.random() < prob
        if not keep:
            return None

        if not self.rescale_admitted_rows:
            return row

        prob = max(prob, 1e-12)
        scale = 1.0 / np.sqrt(prob)
        return row.multiply(scale)

    def _online_ridge_scores_batch(self, rows: sp.csr_matrix) -> np.ndarray:
        rows = rows.tocsr()
        row_norms2 = np.asarray(rows.multiply(rows).sum(axis=1)).reshape(-1)

        if self.B is None:
            return np.ones(rows.shape[0], dtype=float)

        b = self.B
        if b is None or b.size == 0:
            return np.ones(rows.shape[0], dtype=float)

        if np.allclose(b, 0.0):
            return np.ones(rows.shape[0], dtype=float)

        lam = self.ridge_lambda

        u = rows @ b.T
        u = np.asarray(u, dtype=float)

        bbt = b @ b.T
        m = np.eye(b.shape[0], dtype=float) + (1.0 / lam) * bbt

        try:
            x = np.linalg.solve(m, u.T).T
        except np.linalg.LinAlgError:
            x = (np.linalg.pinv(m) @ u.T).T

        scores = (row_norms2 / lam) - np.sum(u * x, axis=1) / (lam ** 2)
        return np.clip(scores, 0.0, 1.0)

    def _scores_to_probs(self, scores: np.ndarray) -> np.ndarray:
        probs = self._sampling_constant() * np.asarray(scores, dtype=float)
        return np.clip(probs, self.min_prob, self.max_prob)

    def _sampling_constant(self) -> float:
        d_eff = max(int(self.d or 1), 2)
        return 8.0 * np.log(d_eff) / (self.sample_epsilon ** 2)

    def _sparse_shrink(self, A_buffer: sp.csr_matrix):
        m, d = A_buffer.shape
        l_eff = min(self.l, m, d)

        if l_eff == 0:
            return np.zeros((self.l, d), dtype=float)

        Z = self._block_krylov_iteration(A_buffer, l_eff)

        P = Z.T @ A_buffer
        P = P.toarray() if sp.issparse(P) else np.asarray(P, dtype=float)

        if self.use_approx_p_svd and min(P.shape) > 16:
            s, vt = self._approx_top_svd_rows(P, l_eff)
        else:
            _, s, vt = np.linalg.svd(P, full_matrices=False)

        delta = s[l_eff - 1] ** 2
        s_shrunk = np.sqrt(np.maximum(s[:l_eff] ** 2 - delta, 0.0))

        B_prime = np.zeros((self.l, d), dtype=float)
        B_prime[:l_eff, :] = s_shrunk[:, None] * vt[:l_eff, :]
        return B_prime

    def _block_krylov_iteration(self, A: sp.csr_matrix, rank: int):
        _, d = A.shape
        Pi = self.rng.standard_normal(size=(d, rank))

        current_block = np.asarray(A @ Pi, dtype=float)
        blocks = [current_block]

        for _ in range(self.n_iter):
            current_block = np.asarray(A @ (A.T @ current_block), dtype=float)
            current_block, _ = np.linalg.qr(current_block, mode="reduced")
            blocks.append(current_block)

        K = np.hstack(blocks)
        Q, _ = np.linalg.qr(K, mode="reduced")

        AQ = A.T @ Q
        M = AQ.T @ AQ
        U_hat, _, _ = np.linalg.svd(M, full_matrices=False)

        return Q @ U_hat[:, :rank]

    def _dense_shrink_fast(self, A_dense: np.ndarray):
        m, d = A_dense.shape
        l_eff = min(self.l, m, d)

        if l_eff == 0:
            return np.zeros((self.l, d), dtype=float)

        G = A_dense @ A_dense.T
        evals, U = np.linalg.eigh(G)

        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        U = U[:, idx]

        s = np.sqrt(np.maximum(evals, 0.0))

        nonzero = s[:l_eff] > 1e-12
        vt = np.zeros((l_eff, d), dtype=float)
        if np.any(nonzero):
            U_keep = U[:, :l_eff][:, nonzero]
            s_keep = s[:l_eff][nonzero]
            vt[nonzero, :] = (U_keep.T @ A_dense) / s_keep[:, None]

        delta = s[l_eff - 1] ** 2
        s_shrunk = np.sqrt(np.maximum(s[:l_eff] ** 2 - delta, 0.0))

        B = np.zeros((self.l, d), dtype=float)
        B[:l_eff, :] = s_shrunk[:, None] * vt[:l_eff, :]
        return B

    def _approx_top_svd_rows(self, P: np.ndarray, rank: int):
        r, d = P.shape
        k = min(rank, r, d)

        Omega = self.rng.standard_normal((d, k))
        Y = P @ Omega
        Q, _ = np.linalg.qr(Y, mode="reduced")

        B_small = Q.T @ P
        _, s, vt = np.linalg.svd(B_small, full_matrices=False)

        return s[:rank], vt[:rank, :]
