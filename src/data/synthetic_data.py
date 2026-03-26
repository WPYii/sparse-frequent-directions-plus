import numpy as np
import scipy.sparse as sp


class SyntheticDataGenerator:
    def __init__(self, n, d, z, head_prob=0.9, seed=42):
        if z <= 0:
            raise ValueError("z must be positive.")
        if d <= 0 or n <= 0:
            raise ValueError("n and d must be positive.")
        if z > d:
            raise ValueError("z cannot be larger than d.")
        if not (0.0 <= head_prob <= 1.0):
            raise ValueError("head_prob must be between 0 and 1.")

        self.n = n
        self.d = d
        self.z = z
        self.head_prob = head_prob
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.head_size = min(int(1.5 * z), d)
        self.tail_start = self.head_size

    def generate(self):
        rows = []
        cols = []
        data = []

        for i in range(self.n):
            chosen_cols = self._sample_row_indices()
            chosen_vals = self.rng.choice([-1, 1], size=len(chosen_cols))

            rows.extend([i] * len(chosen_cols))
            cols.extend(chosen_cols)
            data.extend(chosen_vals)

        A = sp.csr_matrix((data, (rows, cols)), shape=(self.n, self.d))
        return A

    def _sample_row_indices(self):
        chosen = set()

        while len(chosen) < self.z:
            use_head = self.rng.random() < self.head_prob

            if use_head and self.head_size > 0:
                col = self.rng.integers(0, self.head_size)
            elif self.tail_start < self.d:
                col = self.rng.integers(self.tail_start, self.d)
            else:
                col = self.rng.integers(0, self.d)

            chosen.add(int(col))

        return list(chosen)

    def get_shape(self):
        return self.n, self.d

    def get_head_size(self):
        return self.head_size