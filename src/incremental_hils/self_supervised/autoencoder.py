"""
Simple Self-Supervised Autoencoder for HILS anomaly detection.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from collections import deque


class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dims: List[int] = [64, 32]):
        super().__init__()
        # Encoder
        encoder_layers = []
        prev = input_dim
        for h in hidden_dims:
            encoder_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        encoder_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class OnlineAutoencoderDetector:
    """
    Incremental-style Autoencoder detector.
    Uses a sliding window buffer + periodic fine-tuning.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: List[int] = [64, 32],
        lr: float = 1e-3,
        buffer_size: int = 500,
        train_every: int = 50,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = SimpleAutoencoder(input_dim, latent_dim, hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.buffer = deque(maxlen=buffer_size)
        self.train_every = train_every
        self._n_seen = 0
        self.feature_names: List[str] = []
        self.reconstruction_errors: List[float] = []

        # For root-cause (feature-wise error)
        self.last_feature_errors: Optional[np.ndarray] = None

    def _to_tensor(self, x: Dict[str, float]) -> torch.Tensor:
        if not self.feature_names:
            self.feature_names = sorted(x.keys())
        vec = np.array([x[f] for f in self.feature_names], dtype=np.float32)
        return torch.from_numpy(vec).to(self.device)

    def learn_one(self, x: Dict[str, float]) -> float:
        self._n_seen += 1
        tensor = self._to_tensor(x)
        self.buffer.append(tensor.cpu().numpy())

        self.model.eval()
        with torch.no_grad():
            recon = self.model(tensor.unsqueeze(0)).squeeze(0)
            error = torch.mean((recon - tensor) ** 2).item()
            # Feature-wise absolute error for root-cause
            self.last_feature_errors = torch.abs(recon - tensor).cpu().numpy()

        self.reconstruction_errors.append(error)

        # Periodic training on buffer
        if self._n_seen % self.train_every == 0 and len(self.buffer) >= 32:
            self._train_on_buffer()

        return float(error)

    def _train_on_buffer(self, epochs: int = 3):
        self.model.train()
        data = torch.tensor(np.array(self.buffer), dtype=torch.float32).to(self.device)

        for _ in range(epochs):
            self.optimizer.zero_grad()
            recon = self.model(data)
            loss = self.criterion(recon, data)
            loss.backward()
            self.optimizer.step()

        self.model.eval()

    def basic_root_cause(self, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.last_feature_errors is None or not self.feature_names:
            return []
        scores = list(zip(self.feature_names, self.last_feature_errors.tolist()))
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[:top_k]