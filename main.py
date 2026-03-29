from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_rcv1
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
from src.utils.config import Config
from src.data.data_loader import DataLoader
from src.data.preprocessing import Preprocessor
from src.data.synthetic_data import SyntheticDataGenerator
from src.utils.logger import logger

from src.evaluation.experiment_runner import ExperimentRunner
from src.visualization.real_data_visualizer import RealDataVisualizer
from src.visualization.synthetic_data_visualizer import SyntheticDataVisualizer


def get_real_data(config):
    loader = DataLoader(config)
    documents = loader.load_documents()

    preprocessor = Preprocessor(
        lowercase=True,
        stop_words=None,
        max_features=None,
    )

    matrix = preprocessor.build_sparse_binary_matrix(documents)

    logger.info("Original shape: %s", matrix.shape)
    logger.info("Vocab size: %d", preprocessor.get_vocab_size())

    real_data = preprocessor.transform(matrix, 3000)

    logger.info("Transposed shape: %s", real_data.shape)

    return real_data

def get_real_data_rcv1(config):
    subset = config.get("data", "rcv1_subset") or "train"
    max_samples = config.get("data", "rcv1_max_samples")
    max_features = config.get("data", "rcv1_max_features")

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

def get_real_data_wiki(n_docs=20000, max_features=50000, d_limit=20000):

    logger.info("Loading Wikipedia dataset...")

    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split=f"train[:{n_docs}]"
    )

    documents = [x["text"] for x in dataset]

    logger.info("Number of documents: %d", len(documents))

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=max_features
    )

    logger.info("Building TF-IDF matrix...")
    X = vectorizer.fit_transform(documents)

    logger.info("TF-IDF shape: %s", X.shape)
    logger.info("Vocabulary size: %d", len(vectorizer.vocabulary_))

    A = X.transpose().tocsr()

    logger.info("Transposed shape: %s", A.shape)

    if d_limit is not None:
        A = A[:, :d_limit]
        logger.info("Dimension limited to: %d", d_limit)

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

def main():
    config = Config("application.yaml")
    l_values = [5, 10, 15, 20, 50, 100]

    csv_path = Path(config.get("output", "results_dir"))
    output_dir = Path(config.get("output", "figures_dir"))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_data = get_real_data(config)

    runner = ExperimentRunner(
        k=10,
        sfd_n_iter=2,
        sfd_random_state=0,
        improved_sfd_n_iter=2,
        improved_sfd_oversample=5,
        improved_sfd_random_state=0,
    )

    results_real = runner.run_sketch_size_experiment(
        A=real_data,
        l_values=l_values,
        dataset_name="real",
    )

    # results_synthetic = run_synthetic_parameter_sweep(runner)

    # results = pd.concat([results_real, results_synthetic], ignore_index=True)
    # results.to_csv(csv_path, index=False)
    results=results_real.to_csv(csv_path, index=False)

    logger.info("Saved results to %s", csv_path)

    real_plotter = RealDataVisualizer(
        csv_path=csv_path,
        output_dir=output_dir,
    )

    # synthetic_plotter = SyntheticDataVisualizer(
    #     csv_path=csv_path,
    #     output_dir=output_dir,
    # )

    real_plot_path = real_plotter.plot()
    # synthetic_plot_path = synthetic_plotter.plot()

    logger.info("Real plot saved to: %s", real_plot_path)
    # logger.info("Synthetic plot saved to: %s", synthetic_plot_path)

if __name__ == "__main__":
    main()