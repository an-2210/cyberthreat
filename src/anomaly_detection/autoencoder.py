"""
Autoencoder anomaly detector for network flows, trained on benign traffic only.

The network learns to reconstruct normal flows through a narrow bottleneck. Flows it cannot reconstruct well
(high mean squared error) are unusual compared with the training traffic and get a high anomaly score.
Like the Isolation Forest, it never sees attack labels during training.
"""

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import nn

from src.anomaly_detection.common import signed_log1p
from src.utils.config import get_absolute_path
from src.utils.logging import setup_logger

logger = setup_logger("autoencoder")


class AutoencoderDetector:
    """Reconstruction-error anomaly detector. Inputs are signed-log transformed and standardized using benign training rows."""

    def __init__(self, hidden_dims: Sequence[int] = (48, 16), epochs: int = 10, batch_size: int = 1024,
                 learning_rate: float = 1e-3, seed: int = 42):
        self.hidden_dims = tuple(hidden_dims)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.network: Optional[nn.Module] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.loss_history_: list = []

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return ((signed_log1p(X) - self.mean_) / self.std_).astype(np.float32)

    def _build(self, n_features: int) -> nn.Module:
        dims = [n_features, *self.hidden_dims]
        layers = []
        for i in range(len(dims) - 1):  # encoder
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        for i in range(len(dims) - 1, 0, -1):  # decoder mirrors the encoder
            layers += [nn.Linear(dims[i], dims[i - 1])]
            if i > 1:
                layers += [nn.ReLU()]
        return nn.Sequential(*layers)

    def fit(self, X_benign: np.ndarray) -> "AutoencoderDetector":
        """Trains on benign rows only."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        transformed = signed_log1p(X_benign)
        self.mean_ = transformed.mean(axis=0)
        self.std_ = np.maximum(transformed.std(axis=0), 1e-6)
        data = torch.from_numpy(self._standardize(X_benign))

        self.network = self._build(data.shape[1])
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        self.network.train()
        self.loss_history_ = []
        for epoch in range(self.epochs):
            order = torch.randperm(len(data))
            total = 0.0
            for start in range(0, len(data), self.batch_size):
                batch = data[order[start:start + self.batch_size]]
                optimizer.zero_grad()
                loss = loss_fn(self.network(batch), batch)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
            self.loss_history_.append(total / len(data))
            logger.info(f"Autoencoder epoch {epoch + 1}/{self.epochs}: benign reconstruction MSE {self.loss_history_[-1]:.5f}")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Per-row mean squared reconstruction error. Higher means more anomalous."""
        if self.network is None:
            raise RuntimeError("Call fit() before score().")
        self.network.eval()
        data = self._standardize(X)
        out = np.empty(len(data), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(data), 65536):
                chunk = torch.from_numpy(data[start:start + 65536])
                err = ((self.network(chunk) - chunk) ** 2).mean(dim=1)
                out[start:start + len(chunk)] = err.numpy()
        return out

    def save(self, path: Union[str, Path]) -> Path:
        path = get_absolute_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.network.state_dict(),
            "n_features": int(self.mean_.shape[0]),
            "hidden_dims": self.hidden_dims,
            "mean": self.mean_,
            "std": self.std_,
        }, path)
        return path
