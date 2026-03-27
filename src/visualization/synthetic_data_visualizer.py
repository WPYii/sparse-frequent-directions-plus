import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.visualization.BaseVisualizer import BaseVisualizer

class SyntheticDataVisualizer(BaseVisualizer):
    def __init__(self, csv_path, output_dir):
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = pd.read_csv(csv_path)

    def plot(self):
        df = self.df[self.df["dataset"] == "synthetic"].copy()

        fig, axes = plt.subplots(3, 4, figsize=(18, 10))

        metric_info = [
            ("projection_error", "Projection Error"),
            ("covariance_error", "Covariance Error"),
            ("runtime_sec", "Run Time"),
        ]

        param_info = [
            ("n", "number of data points"),
            ("d", "dimension"),
            ("l", "sketch size"),
            ("z", "nnz per row"),
        ]

        for row_idx, (metric, ylabel) in enumerate(metric_info):
            for col_idx, (param, xlabel) in enumerate(param_info):
                ax = axes[row_idx, col_idx]
                subset_param = df[df["varied_param"] == param].copy()

                for algorithm in ["SFD", "FD","ASFD"]:
                    subset = subset_param[subset_param["algorithm"] == algorithm].sort_values("param_value")
                    ax.plot(subset["param_value"], subset[metric], marker="o", label=algorithm)

                if row_idx == 0:
                    ax.set_title(xlabel)
                if col_idx == 0:
                    ax.set_ylabel(ylabel)
                if row_idx == 2:
                    ax.set_xlabel(xlabel)

                ax.legend(fontsize=8)

        output_path = self.output_dir / "synthetic_parameter_comparison.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path