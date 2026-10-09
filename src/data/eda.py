"""
Exploratory data analysis plots for the CIC-IDS2017 audit: class balance, per-file composition
and per-column quality problems.
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
from src.utils.logging import setup_logger
from src.utils.plotting import new_figure, save_figure

logger = setup_logger("cicids_eda")


def plot_class_distribution(class_distribution: pd.DataFrame, path: Path) -> Path:
    """Horizontal bar chart of row counts per label (log scale, because BENIGN dominates)."""
    df = class_distribution.sort_values("count")
    fig = new_figure((9, 6))
    ax = fig.add_subplot()
    ax.barh(df["label"], df["count"], color="#4C72B0")
    ax.set_xscale("log")
    ax.set_xlabel("Rows (log scale)")
    ax.set_title("CIC-IDS2017 class distribution (merged, before cleaning)")
    for y, v in enumerate(df["count"]):
        ax.text(v, y, f" {int(v):,}", va="center", fontsize=8)
    return save_figure(fig, path)


def plot_class_by_source_file(composition: pd.DataFrame, path: Path) -> Path:
    """Stacked horizontal bars: which labels each CSV file contributes."""
    fig = new_figure((11, 6))
    ax = fig.add_subplot()
    left = np.zeros(len(composition))
    labels = list(composition.columns)
    colors = _palette(len(labels))
    for color, label in zip(colors, labels):
        values = composition[label].to_numpy()
        ax.barh(composition.index.astype(str), values, left=left, color=color, label=label)
        left = left + values
    ax.set_xscale("log")
    ax.set_xlabel("Rows (log scale)")
    ax.set_title("Label composition by source CSV file")
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    return save_figure(fig, path)


def plot_column_quality(column_quality: pd.DataFrame, path: Path, top_n: int = 20) -> Path:
    """Bar chart of the columns with the most missing, infinite or negative values."""
    issues = column_quality[["missing", "infinite", "negative"]].sum(axis=1)
    top = column_quality.loc[issues.sort_values(ascending=False).index[:top_n]]
    fig = new_figure((10, 6))
    ax = fig.add_subplot()
    if top[["missing", "infinite", "negative"]].to_numpy().sum() == 0:
        ax.text(0.5, 0.5, "No missing, infinite or negative values in any column",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        y = np.arange(len(top))
        height = 0.27
        ax.barh(y - height, top["missing"], height, label="missing (NaN)", color="#DD8452")
        ax.barh(y, top["infinite"], height, label="infinite", color="#55A868")
        ax.barh(y + height, top["negative"], height, label="negative", color="#8172B3")
        ax.set_yticks(y, labels=[str(c) for c in top.index])
        ax.invert_yaxis()
        ax.set_xlabel("Cells")
        ax.set_xscale("symlog", linthresh=1)
        ax.legend(fontsize=8)
    ax.set_title(f"Columns with the most data-quality issues (top {top_n})")
    return save_figure(fig, path)


def run_eda_plots(audit_result: Dict, figures_dir: Path) -> Dict[str, Path]:
    """Writes all audit-related EDA figures and returns their paths."""
    figures_dir = Path(figures_dir)
    paths = {
        "class_distribution": plot_class_distribution(
            audit_result["class_distribution"], figures_dir / "cicids2017_class_distribution.png"
        ),
        "column_quality": plot_column_quality(
            audit_result["column_quality"], figures_dir / "cicids2017_column_quality.png"
        ),
    }
    composition: Optional[pd.DataFrame] = audit_result.get("class_composition")
    if composition is not None:
        paths["class_by_source_file"] = plot_class_by_source_file(
            composition, figures_dir / "cicids2017_class_by_source_file.png"
        )
    logger.info(f"Saved EDA figures to {figures_dir}")
    return paths


def _palette(n: int):
    """Returns n colors from the tab20 colormap (cycled if n > 20)."""
    from matplotlib import colormaps

    cmap = colormaps["tab20"]
    return [cmap(i % 20) for i in range(n)]
