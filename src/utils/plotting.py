"""
Headless figure helpers. Uses the object-oriented Agg canvas, so no GUI backend or pyplot state is needed.
"""

from pathlib import Path
from typing import Tuple, Union
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def new_figure(size: Tuple[float, float] = (8, 5)) -> Figure:
    """Creates a Figure bound to an Agg canvas."""
    fig = Figure(figsize=size, constrained_layout=True)
    FigureCanvasAgg(fig)
    return fig


def save_figure(fig: Figure, path: Union[str, Path], dpi: int = 150) -> Path:
    """Saves a figure to `path` as PNG, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    return path
