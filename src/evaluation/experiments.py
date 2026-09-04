"""
Experiment runner for training baseline models, tracking execution metrics, exporting comparison tables, and plotting figures.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve
from src.evaluation.metrics import evaluate_classification
from src.models.baselines import BaseCyberModel
from src.utils.config import get_absolute_path
from src.utils.logging import setup_logger

logger = setup_logger("experiment_runner")


class ExperimentRunner:
    """Coordinates multi-model evaluation, computational latency benchmarking, and paper figure generation."""

    def __init__(self, results_dir: Union[str, Path] = "results"):
        self.results_dir = get_absolute_path(results_dir)
        self.metrics_dir = self.results_dir / "metrics"
        self.tables_dir = self.results_dir / "tables"
        self.figures_dir = self.results_dir / "figures"

        for d in [self.metrics_dir, self.tables_dir, self.figures_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def run_baseline_comparison(
        self,
        models: Dict[str, BaseCyberModel],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        experiment_name: str = "baseline_comparison",
    ) -> pd.DataFrame:
        """Trains and evaluates a dictionary of models, storing timing & performance metrics.

        Args:
            models: Dictionary mapping model_id -> BaseCyberModel instance.
            X_train: Training feature DataFrame.
            y_train: Training label Series.
            X_test: Testing feature DataFrame.
            y_test: Testing label Series.
            experiment_name: Identifier for saved metric artifacts.

        Returns:
            pandas DataFrame comparing models across metrics.
        """
        logger.info(f"Running baseline comparison experiment '{experiment_name}' across {len(models)} models...")
        results_list = []
        roc_data = {}

        for model_id, model in models.items():
            logger.info(f"--- Evaluating Model: {model_id} ---")
            
            # 1. Train model & measure runtime
            t_train_start = time.time()
            model.fit(X_train, y_train)
            train_time_sec = time.time() - t_train_start

            # 2. Measure inference latency
            t_inf_start = time.time()
            y_pred = model.predict(X_test)
            inf_time_total = time.time() - t_inf_start
            inf_ms_per_1k = (inf_time_total / max(len(X_test), 1)) * 1000.0 * 1000.0

            # Predict probabilities if supported
            y_prob = None
            try:
                y_prob = model.predict_proba(X_test)
                if y_prob.ndim == 2:
                    y_prob_pos = y_prob[:, 1]
                else:
                    y_prob_pos = y_prob
                if len(np.unique(y_test)) <= 2:
                    fpr, tpr, _ = roc_curve(y_test, y_prob_pos)
                    roc_data[model_id] = (fpr, tpr)
            except Exception as e:
                logger.warning(f"Probability estimation omitted for {model_id}: {e}")

            # 3. Compute metrics
            metrics = evaluate_classification(y_test, y_pred, y_prob=y_prob)
            metrics["model"] = model_id
            metrics["training_time_sec"] = round(train_time_sec, 4)
            metrics["inference_ms_per_1k"] = round(inf_ms_per_1k, 4)

            results_list.append(metrics)

        summary_df = pd.DataFrame(results_list)
        cols_order = [
            "model", "accuracy", "precision_macro", "recall_macro", "macro_f1",
            "weighted_f1", "mcc", "fpr", "fnr", "roc_auc", "training_time_sec", "inference_ms_per_1k"
        ]
        summary_df = summary_df[[c for c in cols_order if c in summary_df.columns]]

        # 4. Save results to CSV, JSON, Markdown
        csv_path = self.metrics_dir / f"{experiment_name}_metrics.csv"
        json_path = self.metrics_dir / f"{experiment_name}_metrics.json"
        md_table_path = self.tables_dir / f"{experiment_name}_table.md"

        summary_df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_list, f, indent=2)

        with open(md_table_path, "w", encoding="utf-8") as f:
            f.write(f"# Baseline Models Comparison Table [{experiment_name}]\n\n")
            f.write(summary_df.to_markdown(index=False))

        logger.info(f"Metrics saved to {csv_path} and {md_table_path}")

        # 5. Plot ROC Curves figure
        if roc_data:
            self._plot_roc_curves(roc_data, experiment_name)

        # 6. Plot Metrics Comparison Bar Chart
        self._plot_metrics_barchart(summary_df, experiment_name)

        return summary_df

    def _plot_roc_curves(self, roc_data: Dict[str, tuple], experiment_name: str) -> None:
        """Plots publication-quality ROC curves."""
        fig, ax = plt.subplots(figsize=(8, 6))
        for model_id, (fpr, tpr) in roc_data.items():
            ax.plot(fpr, tpr, label=f"{model_id}")
        ax.plot([0, 1], [0, 1], "k--", label="Random Chance")
        ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
        ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=12)
        ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        fig_path = self.figures_dir / f"{experiment_name}_roc_curves.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        logger.info(f"Saved ROC figure to: {fig_path}")

    def _plot_metrics_barchart(self, df: pd.DataFrame, experiment_name: str) -> None:
        """Plots bar chart comparison across models."""
        metrics_to_plot = ["accuracy", "macro_f1", "precision_macro", "recall_macro", "mcc"]
        available_metrics = [m for m in metrics_to_plot if m in df.columns]

        df_melt = df.melt(id_vars=["model"], value_vars=available_metrics, var_name="Metric", value_name="Score")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_melt, x="Metric", y="Score", hue="model", palette="Set2", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_title("Baseline Classifier Metrics Comparison", fontsize=14, fontweight="bold")
        ax.set_ylabel("Metric Score", fontsize=12)
        ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        fig_path = self.figures_dir / f"{experiment_name}_metrics_barchart.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        logger.info(f"Saved metrics barchart to: {fig_path}")
