import pandas as pd

from src.algorithms.frequent_directions import FrequentDirections
from src.algorithms.sparse_frequent_directions import SparseFrequentDirections
from src.algorithms.admission_sfd import AdmissionSparseFrequentDirections
from src.evaluation.evaluator import SketchEvaluator
from src.utils.logger import logger


class ExperimentRunner:

    def __init__(self, k=10, sfd_n_iter=2, sfd_random_state=0):
        self.k = k
        self.sfd_n_iter = sfd_n_iter
        self.sfd_random_state = sfd_random_state

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
            asfd = AdmissionSparseFrequentDirections(
                l=l,
                n_iter=self.sfd_n_iter,
                random_state=self.sfd_random_state,
                admission_mode="nnz",   # 可改: "uniform", "nnz", "l2"
                admission_scale=0.8,
                min_prob=0.05,
                max_prob=1.0,
                rescale_admitted_rows=True,
                min_admitted_rows=1,
            )

            result_fd = evaluator.evaluate(fd)
            result_sfd = evaluator.evaluate(sfd)
            result_asfd = evaluator.evaluate(asfd)

            asfd_meta = asfd.get_metadata()

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
                "algorithm": "ASFD",
                "sketch_size": l,
                "projection_error": result_asfd.projection_error,
                "covariance_error": result_asfd.covariance_error,
                "runtime_sec": result_asfd.runtime_sec,
                "admission_mode": asfd_meta["admission_mode"],
                "admission_scale": asfd_meta["admission_scale"],
                "admission_rate": asfd_meta["admission_rate"],
                "total_rows_seen": asfd_meta["total_rows_seen"],
                "total_rows_admitted": asfd_meta["total_rows_admitted"],
                "total_rows_dropped": asfd_meta["total_rows_dropped"],
                "total_blocks": asfd_meta["total_blocks"],
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
        asfd = AdmissionSparseFrequentDirections(
            l=l,
            n_iter=self.sfd_n_iter,
            random_state=self.sfd_random_state,
            admission_mode="nnz",
            admission_scale=0.8,
            min_prob=0.05,
            max_prob=1.0,
            rescale_admitted_rows=True,
            min_admitted_rows=1,
        )

        result_fd = evaluator.evaluate(fd)
        result_sfd = evaluator.evaluate(sfd)
        result_asfd = evaluator.evaluate(asfd)

        asfd_meta = asfd.get_metadata()

        fd_row = {
            "dataset": dataset_name,
            "algorithm": "FD",
            "sketch_size": l,
            "projection_error": result_fd.projection_error,
            "covariance_error": result_fd.covariance_error,
            "runtime_sec": result_fd.runtime_sec,
        }

        sfd_row = {
            "dataset": dataset_name,
            "algorithm": "SFD",
            "sketch_size": l,
            "projection_error": result_sfd.projection_error,
            "covariance_error": result_sfd.covariance_error,
            "runtime_sec": result_sfd.runtime_sec,
        }

        asfd_row = {
            "dataset": dataset_name,
            "algorithm": "ASFD",
            "sketch_size": l,
            "projection_error": result_asfd.projection_error,
            "covariance_error": result_asfd.covariance_error,
            "runtime_sec": result_asfd.runtime_sec,
            "admission_mode": asfd_meta["admission_mode"],
            "admission_scale": asfd_meta["admission_scale"],
            "admission_rate": asfd_meta["admission_rate"],
            "total_rows_seen": asfd_meta["total_rows_seen"],
            "total_rows_admitted": asfd_meta["total_rows_admitted"],
            "total_rows_dropped": asfd_meta["total_rows_dropped"],
            "total_blocks": asfd_meta["total_blocks"],
        }

        if metadata is not None:
            fd_row.update(metadata)
            sfd_row.update(metadata)
            asfd_row.update(metadata)

        return pd.DataFrame([fd_row, sfd_row, asfd_row])