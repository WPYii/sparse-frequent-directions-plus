import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


class ImprovedSparseFrequentDirections:
    def __init__(
        self,
        l: int,
        n_iter: int = 1,
        random_state: int | None = 0,
        buffer_mult: int = 4,
        max_buffer_rows: int | None = None,
        use_approx_p_svd: bool = True,
        p_oversample: int = 5,
    ):
        if l <= 0:
            raise ValueError("Sketch size l must be positive.")
        if n_iter <= 0:
            raise ValueError("n_iter must be positive.")
        if buffer_mult <= 0:
            raise ValueError("buffer_mult must be positive.")

        self.l = l
        self.n_iter = n_iter
        self.buffer_mult = buffer_mult
        self.max_buffer_rows = max_buffer_rows
        self.use_approx_p_svd = use_approx_p_svd
        self.p_oversample = p_oversample

        self.rng = np.random.default_rng(random_state)

        self.B = None
        self.d = None

        self.buffer_blocks: list[sp.csr_matrix] = []
        self.buffer_nnz = 0
        self.buffer_rows = 0

    def fit(self, A, input_block_size: int = 256):
        n, d = A.shape

        if self.B is None:
            self.B = np.zeros((self.l, d), dtype=float)
            self.d = d

        iterator = tqdm(range(0, n, input_block_size), desc="Running Fast Sparse Frequent Directions")
        for start in iterator:
            end = min(start + input_block_size, n)
            block = A[start:end] if sp.issparse(A) else sp.csr_matrix(A[start:end])
            self.process_block(block)

        return self.get_sketch()

    def process_block(self, block):
        if self.B is None:
            d = block.shape[1]
            self.B = np.zeros((self.l, d), dtype=float)
            self.d = d

        block = block.tocsr() if sp.issparse(block) else sp.csr_matrix(block)

        self.buffer_blocks.append(block)
        self.buffer_nnz += block.nnz
        self.buffer_rows += block.shape[0]

        if self._buffer_is_full():
            self._flush_buffer()

    def get_sketch(self):
        if self.B is None:
            return None

        if self.buffer_blocks:
            self._flush_buffer()

        return self.B

    def _buffer_is_full(self):
        if not self.buffer_blocks:
            return False

        nnz_limit = self.buffer_mult * self.l * self.d

        if self.buffer_nnz >= nnz_limit:
            return True

        if self.max_buffer_rows is not None and self.buffer_rows >= self.max_buffer_rows:
            return True

        return False

    def _flush_buffer(self):
        if not self.buffer_blocks:
            return

        A_buffer = sp.vstack(self.buffer_blocks, format="csr")
        B_prime = self._sparse_shrink(A_buffer)

        merged = np.vstack([self.B, B_prime])
        self.B = self._dense_shrink_fast(merged)

        self.buffer_blocks.clear()
        self.buffer_nnz = 0
        self.buffer_rows = 0

    def _sparse_shrink(self, A_buffer: sp.csr_matrix):
        m, d = A_buffer.shape
        l_eff = min(self.l, m, d)

        if l_eff == 0:
            return np.zeros((self.l, d), dtype=float)

        Z = self._simultaneous_iteration(A_buffer, l_eff)

        # P has shape (l_eff, d)
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

    def _dense_shrink_fast(self, A_dense: np.ndarray):
        m, d = A_dense.shape
        l_eff = min(self.l, m, d)

        if l_eff == 0:
            return np.zeros((self.l, d), dtype=float)

        # Faster than full SVD when d is large: eig on A A^T
        G = A_dense @ A_dense.T  # shape: (m, m)
        evals, U = np.linalg.eigh(G)

        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        U = U[:, idx]

        s = np.sqrt(np.maximum(evals, 0.0))

        # Recover V^T from A = U S V^T
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

    def _simultaneous_iteration(self, A: sp.csr_matrix, l_eff: int):
        _, d = A.shape
        G = self.rng.standard_normal(size=(d, l_eff))

        Y = np.asarray(A @ G, dtype=float)

        for _ in range(self.n_iter):
            Y = np.asarray(A @ (A.T @ Y), dtype=float)

        Z, _ = np.linalg.qr(Y, mode="reduced")
        return Z

    def _approx_top_svd_rows(self, P: np.ndarray, rank: int):
        """
        Approximate SVD for P (shape rank x d).
        We sketch the row-space/right singular vectors using a randomized projection.
        """
        r, d = P.shape
        k = min(rank + self.p_oversample, r, d)

        Omega = self.rng.standard_normal((d, k))
        Y = P @ Omega                      # shape: (r, k)
        Q, _ = np.linalg.qr(Y, mode="reduced")

        B_small = Q.T @ P                  # shape: (k, d)
        U_hat, s, vt = np.linalg.svd(B_small, full_matrices=False)

        # Left singular vectors of P would be Q @ U_hat, but we only need s, vt
        return s[:rank], vt[:rank, :]