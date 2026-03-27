import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


class ImprovedSparseFrequentDirections:
    def __init__(
        self,
        l: int,
        n_iter: int = 2,
        oversample: int = 5,
        random_state: int | None = 0,
    ):
        if l <= 0:
            raise ValueError("Sketch size l must be positive.")
        if n_iter <= 0:
            raise ValueError("n_iter must be positive.")
        if oversample < 0:
            raise ValueError("oversample must be non-negative.")

        self.l = l
        self.n_iter = n_iter
        self.oversample = oversample
        self.rng = np.random.default_rng(random_state)

        self.B = None
        self.d = None

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

        self.buffer_rows.append(row)
        self.buffer_nnz += row.nnz

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

        A_buffer = sp.vstack(self.buffer_rows, format="csr")
        B_prime = self._sparse_shrink(A_buffer)

        merged = np.vstack([self.B, B_prime])
        self.B = self._dense_shrink(merged)

        self.buffer_rows.clear()
        self.buffer_nnz = 0

    def _sparse_shrink(self, A_buffer: sp.csr_matrix):
        m, d = A_buffer.shape
        l_eff = min(self.l, m, d)

        if l_eff == 0:
            return np.zeros((self.l, d), dtype=float)

        work_rank = min(l_eff + self.oversample, m, d)

        Z = self._block_krylov_iteration(A_buffer, work_rank)

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

    def _block_krylov_iteration(self, A: sp.csr_matrix, rank: int):
        m, d = A.shape

        if rank <= 0:
            return np.zeros((m, 0), dtype=float)

        G = self.rng.standard_normal(size=(d, rank))

        Y0 = A @ G
        Y0 = np.asarray(Y0, dtype=float)

        Q, _ = np.linalg.qr(Y0, mode="reduced")
        krylov_blocks = [Q]

        for _ in range(self.n_iter):
            Y = A @ (A.T @ Q)
            Y = np.asarray(Y, dtype=float)

            Q, _ = np.linalg.qr(Y, mode="reduced")
            krylov_blocks.append(Q)

        K = np.hstack(krylov_blocks)
        Z, _ = np.linalg.qr(K, mode="reduced")

        return Z