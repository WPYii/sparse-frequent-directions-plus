import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.visualization.BaseVisualizer import BaseVisualizer

class RealDataVisualizer(BaseVisualizer):
    def __init__(self, csv_path, output_dir):
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = pd.read_csv(csv_path)

    def plot(self):
        df = self.df[self.df["dataset"] == "real"].copy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        metrics = [
            ("projection_error", "Projection Error"),
            ("covariance_error", "Covariance Error"),
            ("runtime_sec", "Run Time"),
        ]

        for ax, (metric, title) in zip(axes, metrics):
            for algorithm in ["SFD", "ImprovedSFD"]:
                subset = df[df["algorithm"] == algorithm].sort_values("sketch_size")
                ax.plot(subset["sketch_size"], subset[metric], marker="o", label=algorithm)

            ax.set_title(title)
            ax.set_xlabel("Sketch Size")
            ax.legend()

        output_path = self.output_dir / "real_sketch_size_comparison.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path
