from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_rcv1
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import config, load_dataset
from src.algorithms.frequent_directions import FrequentDirections
from src.algorithms.sparse_frequent_directions import SparseFrequentDirections
from src.algorithms.improved_spaese_frequent_directions import ImprovedSparseFrequentDirections

from src.utils.config import Config
from src.data.data_loader import DataLoader
from src.data.preprocessing import Preprocessor
from src.data.synthetic_data import SyntheticDataGenerator
from src.utils.logger import logger
import numpy as np
import os
import gzip
import urllib.request
import scipy.sparse as sp
import gzip
import numpy as np
import scipy.sparse as sp
from src.evaluation.experiment_runner import ExperimentRunner
from src.visualization.real_data_visualizer import RealDataVisualizer
from src.visualization.synthetic_data_visualizer import SyntheticDataVisualizer

def get_real_data(config):
    dataset_name = config.get("data", "dataset_name")

    if dataset_name == "20news_group":
        return get_real_data_20news_group(config)
    elif dataset_name == "rcv1":
        return get_real_data_rcv1(config)
    elif dataset_name == "enron":
        return get_real_data_enron_dataset(config)
    elif dataset_name == "amazon0302":
        return get_real_data_amazon0302(config)
    else:
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

    logger.info("Original 20news_group shape: %s", matrix.shape)
    logger.info("Vocab size: %d", preprocessor.get_vocab_size())

    slice_to    = config.get("experiment", "slice_to")
    real_data = preprocessor.transform(matrix, slice_to)

    logger.info("Transposed shape: %s", real_data.shape)

    return real_data

def get_real_data_rcv1(config):
    subset = config.get("data", "rcv1_subset") or "train"
    max_samples = config.get("experiment", "rcv1_max_samples")
    max_features = config.get("experiment", "rcv1_max_features")

    rcv1 = fetch_rcv1(subset=subset)
    A = rcv1.data.tocsr()

    if max_samples is not None:
        A = A[: int(max_samples)]

    if max_features is not None:
        A = A[:, : int(max_features)]

    logger.info("RCV1 subset: %s", subset)
    logger.info("RCV1 shape: %s", A.shape)
    logger.info("RCV1 nnz: %d", A.nnz)
    return A

def get_real_data_enron_dataset(config):
    url = "https://snap.stanford.edu/data/email-Enron.txt.gz"
    dest_path = "./resources/dattasets/email-Enron.txt.gz"
    if not os.path.exists(dest_path):
        logger.info("Downloading SNAP/EMAIL-ENRON from %s", url)
        urllib.request.urlretrieve(url, dest_path)
    
    rows = []
    cols = []
    
    logger.info("Parsing dataset from %s", dest_path)
    with gzip.open(dest_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            src, dst = map(int, line.split())
            rows.append(src)
            cols.append(dst)
            
    max_id = max(max(rows), max(cols)) + 1
    A = sp.coo_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(max_id, max_id)
    ).tocsr()

    logger.info("Dataset Loaded: SNAP/EMAIL-ENRON")
    logger.info("Shape: %s | Non-zeros: %d", A.shape, A.nnz)
    
    slice_from    = config.get("experiment", "slice_from")
    slice_to    = config.get("experiment", "slice_to")

    if slice_from is not None or slice_to is not None:
        row_end = slice_from if slice_from is not None else A.shape[0]
        col_end = slice_to if slice_to is not None else A.shape[1]
        A = A[:row_end, :col_end]

    logger.info("Final shape after slicing: %s", A.shape)
    return A

def get_real_data_amazon0302(config):
    path = config.get("data", "amazon_path")
    logger.info("Loading amazon0302 from %s", path)

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

    A = sp.coo_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n, n)
    ).tocsr()

    logger.info("Original amazon0302 shape: %s", A.shape)
    logger.info("Original amazon0302 nnz: %d", A.nnz)

    slice_from = config.get("experiment", "slice_from")
    slice_to = config.get("experiment", "slice_to")

    if slice_from is not None or slice_to is not None:
        row_end = slice_from if slice_from is not None else A.shape[0]
        col_end = slice_to if slice_to is not None else A.shape[1]
        A = A[:row_end, :col_end]

    logger.info("amazon0302 shape after slicing: %s", A.shape)
    logger.info("amazon0302 nnz after slicing: %d", A.nnz)
    
    return A

def get_synthetic_data(n, d, z, head_prob=0.9, seed=42):
    generator = SyntheticDataGenerator(
        n=n,
        d=d,
        z=z,
        head_prob=head_prob,
        seed=seed,
    )
    return generator.generate()

def run_synthetic_parameter_sweep(runner):
    all_synthetic_results = []

    default_n = 10000
    default_d = 1000
    default_l = 50
    default_z = 100

    sweeps = {
        "n": [10000, 20000, 30000, 40000, 50000, 60000],
        "d": [1000, 2000, 3000, 4000, 5000, 6000],
        "l": [5, 10, 15, 20, 50, 100],
        "z": [5, 100, 200, 300, 400, 500],
    }

    for varied_param, values in sweeps.items():
        for value in values:
            n = default_n
            d = default_d
            l = default_l
            z = default_z

            if varied_param == "n":
                n = value
            elif varied_param == "d":
                d = value
            elif varied_param == "l":
                l = value
            elif varied_param == "z":
                z = value

            logger.info(
                "Synthetic experiment: %s=%s (n=%d d=%d l=%d z=%d)",
                varied_param,
                value,
                n,
                d,
                l,
                z,
            )

            synthetic_data = get_synthetic_data(
                n=n,
                d=d,
                z=z,
                head_prob=0.9,
                seed=42,
            )

            df = runner.run_single_experiment(
                A=synthetic_data,
                l=l,
                dataset_name="synthetic",
                metadata={
                    "varied_param": varied_param,
                    "param_value": value,
                    "n": n,
                    "d": d,
                    "z": z,
                    "l": l,
                },
            )

            all_synthetic_results.append(df)

    return pd.concat(all_synthetic_results, ignore_index=True)

def numerical_comparison(results_df):
    required_cols = {
        "algorithm",
        "runtime_sec",
        "projection_error",
        "covariance_error",
    }

    missing = required_cols - set(results_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    avg_metrics = (
        results_df
        .groupby("algorithm")[["runtime_sec", "projection_error", "covariance_error"]]
        .mean()
    )

    if "ImprovedSFD" not in avg_metrics.index:
        raise ValueError("ImprovedSFD results not found in dataframe")

    def compute_improvement(baseline, improved):
        if baseline is None or improved is None:
            return None
        if baseline == 0:
            return None
        return (baseline - improved) / baseline * 100

    metrics = ["runtime_sec", "projection_error", "covariance_error"]
    baselines = ["FD", "SFD"]
    results = {}

    print("\n===== Numerical Comparison =====")

    for algo in avg_metrics.index:
        print(f"\n{algo}:")
        print(f"  Runtime: {avg_metrics.loc[algo, 'runtime_sec']:.6f}")
        print(f"  Projection Error: {avg_metrics.loc[algo, 'projection_error']:.6f}")
        print(f"  Covariance Error: {avg_metrics.loc[algo, 'covariance_error']:.6f}")

    improved_row = avg_metrics.loc["ImprovedSFD"]

    print("\n===== Improvements of ImprovedSFD =====")
    for baseline_algo in baselines:
        if baseline_algo not in avg_metrics.index:
            continue

        baseline_row = avg_metrics.loc[baseline_algo]
        print(f"\nAgainst {baseline_algo}:")

        for metric in metrics:
            improvement = compute_improvement(
                baseline_row[metric],
                improved_row[metric]
            )
            results[f"Improved_vs_{baseline_algo}_{metric}_%"] = improvement

            if improvement is not None:
                print(f"  {metric}: {improvement:.2f}%")
            else:
                print(f"  {metric}: N/A")

    for algo in avg_metrics.index:
        for metric in metrics:
            results[f"{algo}_avg_{metric}"] = avg_metrics.loc[algo, metric]

    return results

def main():
    config = Config("application.yaml")
    l_values    = config.get("experiment", "l_values")
    k           = config.get("experiment", "k")
    csv_path    = Path(config.get("output", "results_dir"))
    output_dir  = Path(config.get("output", "figures_dir"))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_data = get_real_data(config=config)

    runner = ExperimentRunner(
        k=k,
        algorithm_factories={
            "FD": lambda l: FrequentDirections(l=l),
            "SFD": lambda l: SparseFrequentDirections(
                l=l,
                n_iter=2,
                random_state=42,
            ),
            "ImprovedSFD": lambda l: ImprovedSparseFrequentDirections(
                l=l,
                n_iter=2,
                p_oversample=5,
                random_state=42,
            ),
        },
    )

    results_real = runner.run_sketch_size_experiment(
        A=real_data,
        l_values=l_values,
        dataset_name="real",
    )
    results_synthetic = run_synthetic_parameter_sweep(runner)
    results = pd.concat([results_real, results_synthetic], ignore_index=True)
    numerical_comparison(results)
    results.to_csv(csv_path, index=False)
    logger.info("Saved results to %s", csv_path)

    real_plotter = RealDataVisualizer(
        csv_path=csv_path,
        output_dir=output_dir,
    )

    synthetic_plotter = SyntheticDataVisualizer(
        csv_path=csv_path,
        output_dir=output_dir,
    )

    real_plot_path = real_plotter.plot()
    synthetic_plot_path = synthetic_plotter.plot()

    logger.info("Real plot saved to: %s", real_plot_path)
    logger.info("Synthetic plot saved to: %s", synthetic_plot_path)

if __name__ == "__main__":
    main()