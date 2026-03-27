import pandas as pd

from src.algorithms.frequent_directions import FrequentDirections
from src.algorithms.sparse_frequent_directions import SparseFrequentDirections
from src.algorithms.improved_spaese_frequent_directions import ImprovedSparseFrequentDirections
from src.evaluation.evaluator import SketchEvaluator
from src.utils.logger import logger


class ExperimentRunner:
    def __init__(
        self,
        k=10,
        sfd_n_iter=2,
        sfd_random_state=0,
        improved_sfd_n_iter=2,
        improved_sfd_oversample=5,
        improved_sfd_random_state=0,
    ):
        self.k = k
        self.sfd_n_iter = sfd_n_iter
        self.sfd_random_state = sfd_random_state
        self.improved_sfd_n_iter = improved_sfd_n_iter
        self.improved_sfd_oversample = improved_sfd_oversample
        self.improved_sfd_random_state = improved_sfd_random_state

    def run_sketch_size_experiment(self, A, l_values, dataset_name):
        evaluator = SketchEvaluator(A=A, k=self.k)
        all_results = []

        for l in l_values:
            logger.info(
                "Running %s experiment with sketch size l=%d",
                dataset_name,
                l,
            )

            fd = FrequentDirections(l=l)
            sfd = SparseFrequentDirections(
                l=l,
                n_iter=self.sfd_n_iter,
                random_state=self.sfd_random_state,
            )
            improved_sfd = ImprovedSparseFrequentDirections(
                l=l,
                n_iter=self.improved_sfd_n_iter,
                oversample=self.improved_sfd_oversample,
                random_state=self.improved_sfd_random_state,
            )

            result_fd = evaluator.evaluate(fd)
            result_sfd = evaluator.evaluate(sfd)
            result_improved_sfd = evaluator.evaluate(improved_sfd)

            all_results.append({
                "dataset": dataset_name,
                "algorithm": "FD",
                "sketch_size": l,
                "projection_error": result_fd.projection_error,
                "covariance_error": result_fd.covariance_error,
                "runtime_sec": result_fd.runtime_sec,
            })

            all_results.append({
                "dataset": dataset_name,
                "algorithm": "SFD",
                "sketch_size": l,
                "projection_error": result_sfd.projection_error,
                "covariance_error": result_sfd.covariance_error,
                "runtime_sec": result_sfd.runtime_sec,
            })

            all_results.append({
                "dataset": dataset_name,
                "algorithm": "ImprovedSFD",
                "sketch_size": l,
                "projection_error": result_improved_sfd.projection_error,
                "covariance_error": result_improved_sfd.covariance_error,
                "runtime_sec": result_improved_sfd.runtime_sec,
            })

        return pd.DataFrame(all_results)

    def run_single_experiment(self, A, l, dataset_name, metadata=None):
        evaluator = SketchEvaluator(A=A, k=self.k)

        fd = FrequentDirections(l=l)
        sfd = SparseFrequentDirections(
            l=l,
            n_iter=self.sfd_n_iter,
            random_state=self.sfd_random_state,
        )
        improved_sfd = ImprovedSparseFrequentDirections(
            l=l,
            n_iter=self.improved_sfd_n_iter,
            oversample=self.improved_sfd_oversample,
            random_state=self.improved_sfd_random_state,
        )

        result_fd = evaluator.evaluate(fd)
        result_sfd = evaluator.evaluate(sfd)
        result_improved_sfd = evaluator.evaluate(improved_sfd)

        rows = [
            {
                "dataset": dataset_name,
                "algorithm": "FD",
                "sketch_size": l,
                "projection_error": result_fd.projection_error,
                "covariance_error": result_fd.covariance_error,
                "runtime_sec": result_fd.runtime_sec,
            },
            {
                "dataset": dataset_name,
                "algorithm": "SFD",
                "sketch_size": l,
                "projection_error": result_sfd.projection_error,
                "covariance_error": result_sfd.covariance_error,
                "runtime_sec": result_sfd.runtime_sec,
            },
            {
                "dataset": dataset_name,
                "algorithm": "ImprovedSFD",
                "sketch_size": l,
                "projection_error": result_improved_sfd.projection_error,
                "covariance_error": result_improved_sfd.covariance_error,
                "runtime_sec": result_improved_sfd.runtime_sec,
            },
        ]

        if metadata is not None:
            for row in rows:
                row.update(metadata)

        return pd.DataFrame(rows)