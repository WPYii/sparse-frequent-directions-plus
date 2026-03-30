import pandas as pd

from src.algorithms.frequent_directions import FrequentDirections
from src.algorithms.sparse_frequent_directions import SparseFrequentDirections
from src.algorithms.improved_spaese_frequent_directions import ImprovedSparseFrequentDirections
from src.evaluation.evaluator import SketchEvaluator
from src.utils.logger import logger

class ExperimentRunner:
    def __init__(self,k=10,algorithm_factories=None,):
        self.k = k

        if algorithm_factories is None:
            self.algorithm_factories = {
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
            }
        else:
            self.algorithm_factories = algorithm_factories

    def _build_row(self, dataset_name, algorithm_name, l, result, metadata=None):
        row = {
            "dataset": dataset_name,
            "algorithm": algorithm_name,
            "sketch_size": l,
            "projection_error": result.projection_error,
            "covariance_error": result.covariance_error,
            "runtime_sec": result.runtime_sec,
        }

        if metadata is not None:
            row.update(metadata)

        return row

    def _run_one_l(self, A, l, dataset_name, metadata=None):
        evaluator = SketchEvaluator(A=A, k=self.k)
        rows = []

        for algorithm_name, factory in self.algorithm_factories.items():
            algorithm = factory(l)
            result = evaluator.evaluate(algorithm)

            rows.append(
                self._build_row(
                    dataset_name=dataset_name,
                    algorithm_name=algorithm_name,
                    l=l,
                    result=result,
                    metadata=metadata,
                )
            )

        return rows

    def run_sketch_size_experiment(self, A, l_values, dataset_name, metadata=None):
        all_results = []

        for l in l_values:
            logger.info(
                "Running %s experiment with sketch size l=%d",
                dataset_name,
                l,
            )
            all_results.extend(
                self._run_one_l(
                    A=A,
                    l=l,
                    dataset_name=dataset_name,
                    metadata=metadata,
                )
            )

        return pd.DataFrame(all_results)

    def run_single_experiment(self, A, l, dataset_name, metadata=None):
        rows = self._run_one_l(
            A=A,
            l=l,
            dataset_name=dataset_name,
            metadata=metadata,
        )
        return pd.DataFrame(rows)