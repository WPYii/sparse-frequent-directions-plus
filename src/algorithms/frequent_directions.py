import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


class FrequentDirections:
    def __init__(self, l: int):
        if l <= 0:
            raise ValueError("Sketch size l must be positive.")
        self.l = l
        self.B = None          
        self.next_row = 0  

    def fit(self, A):
        n, d = A.shape

        if self.B is None:
            self.B = np.zeros((2 * self.l, d), dtype=float)

        for i in tqdm(range(n), desc="Running Frequent Directions"):
            row = A.getrow(i) if sp.issparse(A) else A[i]
            self.process_row(row)

        return self.get_sketch()

    def process_row(self, row):
        dense_row = self._to_dense_1d(row)

        if self.B is None:
            d = dense_row.shape[0]
            self.B = np.zeros((2 * self.l, d), dtype=float)

        self.B[self.next_row] = dense_row
        self.next_row += 1

        if self.next_row == 2 * self.l:
            self._shrink()

    def _shrink(self):
        U, s, Vt = np.linalg.svd(self.B, full_matrices=False)

        delta = s[self.l - 1] ** 2
        s_shrunk = np.sqrt(np.maximum(s[:self.l] ** 2 - delta, 0.0))

        B_top = np.diag(s_shrunk) @ Vt[:self.l, :]

        d = self.B.shape[1]
        self.B = np.zeros((2 * self.l, d), dtype=float)
        self.B[:self.l, :] = B_top
        self.next_row = self.l

    def get_sketch(self):
        if self.B is None:
            return None

        active = self.B[:self.next_row, :]

        if active.shape[0] <= self.l:
            out = np.zeros((self.l, self.B.shape[1]), dtype=float)
            out[:active.shape[0], :] = active
            return out

        U, s, Vt = np.linalg.svd(active, full_matrices=False)
        delta = s[self.l - 1] ** 2
        s_shrunk = np.sqrt(np.maximum(s[:self.l] ** 2 - delta, 0.0))
        return np.diag(s_shrunk) @ Vt[:self.l, :]

    @staticmethod
    def _to_dense_1d(row):
        if sp.issparse(row):
            return row.toarray().ravel().astype(float)
        return np.asarray(row, dtype=float).ravel()