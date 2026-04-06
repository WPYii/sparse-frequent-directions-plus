import argparse
import gzip
import os
import tempfile
import urllib.request
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.datasets import fetch_rcv1

from src.algorithms.improved_spaese_frequent_directions import (
    ImprovedSparseFrequentDirections,
)
from src.algorithms.sparse_frequent_directions import SparseFrequentDirections
from src.data.data_loader import DataLoader
from src.data.preprocessing import Preprocessor
from src.data.synthetic_data import SyntheticDataGenerator
from src.evaluation.experiment_runner import ExperimentRunner
from src.utils.config import Config
from src.utils.logger import logger


class ImprovedSFDWithBlockKrylovOnly(SparseFrequentDirections):
    """SFD with only the Block Krylov sparse shrink change."""

    def _sparse_shrink(self, A_buffer: sp.csr_matrix):
        m, d = A_buffer.shape
        l_eff = min(self.l, m, d)

        if l_eff == 0:
            return np.zeros((self.l, d), dtype=float)

        helper = ImprovedSparseFrequentDirections(
            l=self.l,
            n_iter=self.n_iter,
            random_state=None,
            use_approx_p_svd=False,
        )
        helper.rng = self.rng
        Z = helper._block_krylov_iteration(A_buffer, l_eff)

        P = Z.T @ A_buffer
        P = P.toarray() if sp.issparse(P) else np.asarray(P, dtype=float)

        _, s, vt = np.linalg.svd(P, full_matrices=False)

        delta = s[l_eff - 1] ** 2
        s_shrunk = np.sqrt(np.maximum(s[:l_eff] ** 2 - delta, 0.0))

        B_prime = np.zeros((self.l, d), dtype=float)
        B_prime[:l_eff, :] = np.diag(s_shrunk) @ vt[:l_eff, :]
        return B_prime


class ImprovedSFDWithDenseShrinkOnly(SparseFrequentDirections):
    """SFD with only the fast dense shrink change."""

    def _dense_shrink(self, A_dense: np.ndarray):
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


class ImprovedSFDWithORSOnly(ImprovedSparseFrequentDirections):
    """SFD with only the ORS admission change."""

    def _flush_buffer(self):
        if not self.buffer_rows:
            return

        a_buffer = sp.vstack(self.buffer_rows, format="csr")
        b_prime = SparseFrequentDirections._sparse_shrink(self, a_buffer)

        merged = np.vstack([self.B, b_prime])
        self.B = SparseFrequentDirections._dense_shrink(self, merged)

        self.buffer_rows.clear()
        self.buffer_nnz = 0

    def _sparse_shrink(self, A_buffer: sp.csr_matrix):
        return SparseFrequentDirections._sparse_shrink(self, A_buffer)

    def _simultaneous_iteration(self, A: sp.csr_matrix, l_eff: int):
        return SparseFrequentDirections._simultaneous_iteration(self, A, l_eff)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ablations for the current Improved Sparse Frequent Directions.",
    )
    parser.add_argument("--config", default="application.yaml", help="Path to config YAML.")
    parser.add_argument(
        "--dataset-mode",
        choices=["config", "synthetic"],
        default="config",
        help="Load the dataset from config or generate synthetic data.",
    )
    parser.add_argument(
        "--l-values",
        default=None,
        help="Comma-separated sketch sizes. Defaults to experiment.l_values from config.",
    )
    parser.add_argument(
        "--output",
        default="resources/excel/ablation_results.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--plot-output",
        default=None,
        help="Optional plot path. Defaults next to --output with .png suffix.",
    )
    parser.add_argument("--n", type=int, default=10000, help="Synthetic sample count.")
    parser.add_argument("--d", type=int, default=1000, help="Synthetic dimension.")
    parser.add_argument("--z", type=int, default=100, help="Synthetic nnz per row.")
    parser.add_argument("--head-prob", type=float, default=0.9, help="Synthetic head probability.")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic random seed.")
    parser.add_argument("--n-iter", type=int, default=2, help="Iteration count.")
    return parser.parse_args()


def parse_l_values(raw, config):
    if raw is None:
        return config.get("experiment", "l_values")
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def get_real_data(config):
    dataset_name = config.get("data", "dataset_name")

    if dataset_name == "20news_group":
        return get_real_data_20news_group(config)
    if dataset_name == "rcv1":
        return get_real_data_rcv1(config)
    if dataset_name == "enron":
        return get_real_data_enron_dataset(config)
    if dataset_name == "amazon0302":
        return get_real_data_amazon0302(config)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_real_data_20news_group(config):
    loader = DataLoader(config)
    documents = loader.load_documents()

    preprocessor = Preprocessor(
        lowercase=True,
        stop_words=None,
        max_features=None,
    )
    matrix = preprocessor.build_sparse_binary_matrix(documents)
    slice_to = config.get("experiment", "slice_to")
    real_data = preprocessor.transform(matrix, slice_to)
    logger.info("20news_group shape after preprocessing: %s", real_data.shape)
    return real_data


def get_real_data_rcv1(config):
    subset = config.get("data", "rcv1_subset") or "train"
    max_samples = config.get("experiment", "rcv1_max_samples")
    max_features = config.get("experiment", "rcv1_max_features")

    A = fetch_rcv1(subset=subset).data.tocsr()
    if max_samples is not None:
        A = A[: int(max_samples)]
    if max_features is not None:
        A = A[:, : int(max_features)]

    logger.info("RCV1 shape after slicing: %s", A.shape)
    return A


def get_real_data_enron_dataset(config):
    path = config.get("data", "enron_path")
    url = "https://snap.stanford.edu/data/email-Enron.txt.gz"

    if not os.path.exists(path):
        logger.info("Downloading SNAP/EMAIL-ENRON from %s", url)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)

    rows = []
    cols = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            src, dst = map(int, line.split())
            rows.append(src)
            cols.append(dst)

    max_id = max(max(rows), max(cols)) + 1
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(max_id, max_id)).tocsr()

    slice_to = config.get("experiment", "slice_to")
    if slice_to is not None:
        A = A[:, :slice_to]
    logger.info("Enron shape after slicing: %s", A.shape)
    return A


def get_real_data_amazon0302(config):
    path = config.get("data", "amazon_path")
    rows = []
    cols = []

    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            src, dst = map(int, line.split())
            rows.append(src)
            cols.append(dst)

    rows = np.array(rows)
    cols = np.array(cols)
    n = max(rows.max(), cols.max()) + 1
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()

    slice_to = config.get("experiment", "slice_to")
    if slice_to is not None:
        A = A[:, :slice_to]
    logger.info("amazon0302 shape after slicing: %s", A.shape)
    return A


def get_synthetic_matrix(args):
    generator = SyntheticDataGenerator(
        n=args.n,
        d=args.d,
        z=args.z,
        head_prob=args.head_prob,
        seed=args.seed,
    )
    return generator.generate()


def load_matrix(args, config):
    if args.dataset_mode == "config":
        return get_real_data(config), config.get("data", "dataset_name")
    return get_synthetic_matrix(args), "synthetic"


def build_algorithm_factories(args):
    return {
        "SFD": lambda l: SparseFrequentDirections(
            l=l,
            n_iter=args.n_iter,
            random_state=args.seed,
        ),
        "ImprovedSFD": lambda l: ImprovedSparseFrequentDirections(
            l=l,
            n_iter=args.n_iter,
            random_state=args.seed,
            use_approx_p_svd=False,
            use_ors_admission=True,
            candidate_block_size=256,
            sample_epsilon=0.5,
            ridge_lambda=1.0,
        ),
        "SFD+BK": lambda l: ImprovedSFDWithBlockKrylovOnly(
            l=l,
            n_iter=args.n_iter,
            random_state=args.seed,
        ),
        "SFD+FastDenseShrink": lambda l: ImprovedSFDWithDenseShrinkOnly(
            l=l,
            n_iter=args.n_iter,
            random_state=args.seed,
        ),
        "SFD+ORS": lambda l: ImprovedSFDWithORSOnly(
            l=l,
            n_iter=args.n_iter,
            random_state=args.seed,
            use_ors_admission=True,
            candidate_block_size=256,
            sample_epsilon=0.5,
            ridge_lambda=1.0,
        ),
    }


def summarize_results(df: pd.DataFrame):
    summary = (
        df.groupby("algorithm")[["runtime_sec", "projection_error", "covariance_error"]]
        .mean()
        .sort_index()
    )
    point_values = (
        df.sort_values(["algorithm", "sketch_size"])[
            ["algorithm", "sketch_size", "runtime_sec", "projection_error", "covariance_error"]
        ]
        .reset_index(drop=True)
    )

    print("\n===== Ablation Points =====")
    print(point_values.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\n===== Ablation Summary =====")
    print(summary.to_string(float_format=lambda x: f"{x:.6f}"))


def plot_results(df: pd.DataFrame, output_path: Path):
    metrics = [
        ("projection_error", "Projection Error"),
        ("covariance_error", "Covariance Error"),
        ("runtime_sec", "Run Time"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (metric, title) in zip(axes, metrics):
        for algorithm in sorted(df["algorithm"].unique()):
            subset = df[df["algorithm"] == algorithm].sort_values("sketch_size")
            ax.plot(subset["sketch_size"], subset[metric], marker="o", label=algorithm)

        ax.set_title(title)
        ax.set_xlabel("Sketch Size")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def main():
    args = parse_args()
    config = Config(args.config)
    l_values = parse_l_values(args.l_values, config)
    A, dataset_name = load_matrix(args, config)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_output = Path(args.plot_output) if args.plot_output else output_path.with_suffix(".png")
    plot_output.parent.mkdir(parents=True, exist_ok=True)

    runner = ExperimentRunner(
        k=config.get("experiment", "k"),
        algorithm_factories=build_algorithm_factories(args),
    )

    logger.info(
        "Running ablation on dataset '%s' with l_values=%s",
        dataset_name,
        l_values,
    )
    results = runner.run_sketch_size_experiment(
        A=A,
        l_values=l_values,
        dataset_name=f"ablation_{dataset_name}",
    )
    results.to_csv(output_path, index=False)
    logger.info("Saved ablation results to %s", output_path)
    plot_results(results, plot_output)
    logger.info("Saved ablation plot to %s", plot_output)
    summarize_results(results)


if __name__ == "__main__":
    main()
