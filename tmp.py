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

        # Oversampling is the paper's key recommendation for small spectral gaps [cite: 1037, 1044]
        # p > k is used to accelerate convergence when sigma_k / sigma_{k+1} is small [cite: 1034]
        work_rank = min(l_eff + self.oversample, m, d)

        # Obtain the near-optimal singular vectors from the Krylov subspace [cite: 661, 994]
        Z = self._block_krylov_iteration(A_buffer, work_rank)

        # Project A onto the optimized subspace [cite: 780, 789]
        # P = Z.T @ A captures nearly as much variance as true top k [cite: 632]
        P = Z.T @ A_buffer
        P = np.asarray(P, dtype=float)

        _, s, vt = np.linalg.svd(P, full_matrices=False)

        # Standard SFD shrinking step [cite: 162, 186]
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
        # Algorithm 2 Step 1: Initialize random matrix 
        G = self.rng.standard_normal(size=(d, rank))
        
        # We start with A * G 
        Y = A @ G
        Y = np.asarray(Y, dtype=float)
        
        # Step 2: Accumulate the blocks [A*G, (AA^T)A*G, ..., (AA^T)^q A*G] 
        # Orthonormalizing each block is critical for numerical stability [cite: 843]
        Q_block, _ = np.linalg.qr(Y, mode='reduced')
        krylov_blocks = [Q_block]
        
        for _ in range(self.n_iter):
            # Compute (AA^T) * current_block [cite: 618, 828]
            # Matrix-matrix multiplication is O(nnz(A) * rank) [cite: 663]
            Y = A @ (A.T @ Q_block)
            Y = np.asarray(Y, dtype=float)
            Q_block, _ = np.linalg.qr(Y, mode='reduced')
            krylov_blocks.append(Q_block)
            
        # Step 3: Combine all blocks to form the basis Q 
        K = np.hstack(krylov_blocks)
        Q, _ = np.linalg.qr(K, mode='reduced')
        
        # Step 4 & 5: Find the top singular vectors of the projected matrix M 
        # M = Q^T (AA^T) Q 
        # This Rayleigh-Ritz method finds the best approximation in the subspace [cite: 758]
        AQ = A.T @ Q
        M = AQ.T @ AQ # Equivalent to Q^T A A^T Q
        
        # Compute eigenvectors of M to get the top principal components 
        eigenvalues, U_hat = np.linalg.eigh(M)
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        U_hat = U_hat[:, idx]
        
        # Step 6: Return the basis Z = Q * U_hat_k [cite: 618, 819]
        # We only return the first 'rank' vectors to maintain efficiency
        return Q @ U_hat[:, :rank]