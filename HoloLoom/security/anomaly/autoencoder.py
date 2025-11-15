"""
Autoencoder Anomaly Detector

Deep learning-based anomaly detection using autoencoders.

Features:
- Learns compressed representation of normal behavior
- Anomaly score = reconstruction error
- High reconstruction error = anomalous (model can't reconstruct unfamiliar patterns)
- Good for capturing complex non-linear patterns

Requires: PyTorch

Author: HoloLoom Security Team
Created: 2025-11-15
"""

import logging
from typing import List, Optional
import numpy as np

# Optional torch dependency (graceful degradation)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("⚠️  PyTorch not installed. Autoencoder detector unavailable.")
    nn = None

logger = logging.getLogger(__name__)


if TORCH_AVAILABLE:
    class AutoencoderModel(nn.Module):
        """
        Autoencoder model for anomaly detection.

        Encoder: Compresses input to latent representation
        Decoder: Reconstructs input from latent representation

        Trained on normal data, anomalies have high reconstruction error.
        """

        def __init__(self, input_size: int, latent_size: int = 8):
            super().__init__()

            # Encoder: input -> latent
            self.encoder = nn.Sequential(
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, latent_size),
                nn.ReLU()
            )

            # Decoder: latent -> input
            self.decoder = nn.Sequential(
                nn.Linear(latent_size, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, input_size),
                nn.Sigmoid()  # Normalize output to 0-1
            )

        def forward(self, x):
            """
            Forward pass: encode then decode.

            Args:
                x: Input tensor (batch_size, input_size)

            Returns:
                Reconstructed input (batch_size, input_size)
            """
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed

        def encode(self, x):
            """Get latent representation."""
            return self.encoder(x)


    class EventDataset(Dataset):
        """Dataset for event feature vectors."""

        def __init__(self, features: List[np.ndarray]):
            self.features = [torch.FloatTensor(f) for f in features]

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx]


class AutoencoderDetector:
    """
    Autoencoder-based anomaly detector.

    Learns to reconstruct normal event features.
    Anomaly score = reconstruction error (MSE).

    Features:
    - Deep learning-based
    - Captures complex non-linear patterns
    - No labeled data required (unsupervised)
    """

    def __init__(
        self,
        latent_size: int = 8,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32
    ):
        """
        Initialize Autoencoder detector.

        Args:
            latent_size: Size of latent representation (bottleneck)
            learning_rate: Learning rate for training
            epochs: Training epochs
            batch_size: Batch size for training
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for Autoencoder detector. "
                              "Install with: pip install torch")

        self.latent_size = latent_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # Model (lazy-initialized)
        self._model: Optional['AutoencoderModel'] = None
        self._is_fitted = False
        self._feature_size: Optional[int] = None

        # Reconstruction error statistics (for normalization)
        self._mean_reconstruction_error: float = 0.0
        self._std_reconstruction_error: float = 1.0

        # Device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"AutoencoderDetector initialized (latent_size={latent_size}, "
                    f"device={self._device})")

    async def fit(self, events: List['SecurityEvent']):
        """
        Train autoencoder on normal events.

        Args:
            events: List of security events (should be normal behavior)
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Skipping Autoencoder training.")
            return

        logger.info(f"Training Autoencoder on {len(events)} events")

        # Convert events to feature vectors
        features = [event.to_feature_vector() for event in events]
        self._feature_size = len(features[0])

        # Create dataset and dataloader
        dataset = EventDataset(features)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model
        self._model = AutoencoderModel(
            input_size=self._feature_size,
            latent_size=self.latent_size
        ).to(self._device)

        # Training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self._model.parameters(), lr=self.learning_rate)

        self._model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_features in dataloader:
                batch_features = batch_features.to(self._device)

                # Forward pass (reconstruct)
                reconstructed = self._model(batch_features)
                loss = criterion(reconstructed, batch_features)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

        # Compute reconstruction error statistics on training data
        self._model.eval()
        reconstruction_errors = []
        with torch.no_grad():
            for batch_features in dataloader:
                batch_features = batch_features.to(self._device)
                reconstructed = self._model(batch_features)
                errors = torch.mean((reconstructed - batch_features) ** 2, dim=1)
                reconstruction_errors.extend(errors.cpu().numpy())

        self._mean_reconstruction_error = float(np.mean(reconstruction_errors))
        self._std_reconstruction_error = float(np.std(reconstruction_errors))

        self._is_fitted = True
        logger.info(f"Autoencoder training complete "
                    f"(mean_error={self._mean_reconstruction_error:.4f}, "
                    f"std_error={self._std_reconstruction_error:.4f})")

    async def predict(self, event: 'SecurityEvent') -> float:
        """
        Predict anomaly score for an event.

        Args:
            event: Security event to score

        Returns:
            Anomaly score (0.0-1.0, based on reconstruction error)
        """
        if not self._is_fitted or not TORCH_AVAILABLE:
            return 0.0

        # Convert to feature vector
        features = torch.FloatTensor(event.to_feature_vector()).unsqueeze(0).to(self._device)

        # Reconstruct
        with torch.no_grad():
            reconstructed = self._model(features)

        # Compute reconstruction error (MSE)
        error = torch.mean((reconstructed - features) ** 2).item()

        # Normalize using z-score (how many std devs from mean)
        if self._std_reconstruction_error > 0:
            z_score = (error - self._mean_reconstruction_error) / self._std_reconstruction_error
            # Convert z-score to 0-1 range (z > 3 = very anomalous)
            normalized_score = min(1.0, max(0.0, z_score / 3.0))
        else:
            normalized_score = 0.0

        return float(normalized_score)

    async def predict_batch(self, events: List['SecurityEvent']) -> List[float]:
        """
        Predict anomaly scores for multiple events (efficient batch processing).

        Args:
            events: List of security events

        Returns:
            List of anomaly scores (0.0-1.0)
        """
        if not self._is_fitted or not TORCH_AVAILABLE:
            return [0.0] * len(events)

        # Convert to feature matrix
        features = torch.FloatTensor([e.to_feature_vector() for e in events]).to(self._device)

        # Reconstruct
        with torch.no_grad():
            reconstructed = self._model(features)

        # Compute reconstruction errors
        errors = torch.mean((reconstructed - features) ** 2, dim=1).cpu().numpy()

        # Normalize using z-score
        if self._std_reconstruction_error > 0:
            z_scores = (errors - self._mean_reconstruction_error) / self._std_reconstruction_error
            normalized_scores = np.clip(z_scores / 3.0, 0.0, 1.0)
        else:
            normalized_scores = np.zeros_like(errors)

        return normalized_scores.tolist()

    def get_latent_representation(self, event: 'SecurityEvent') -> Optional[np.ndarray]:
        """
        Get latent (compressed) representation of an event.

        Useful for visualization and analysis.

        Args:
            event: Security event

        Returns:
            Latent vector (latent_size,)
        """
        if not self._is_fitted or not TORCH_AVAILABLE:
            return None

        features = torch.FloatTensor(event.to_feature_vector()).unsqueeze(0).to(self._device)

        with torch.no_grad():
            latent = self._model.encode(features)

        return latent.cpu().numpy().squeeze()

    def save_model(self, path: str):
        """Save trained model to disk."""
        if not self._is_fitted or not TORCH_AVAILABLE:
            logger.warning("Model not fitted. Cannot save.")
            return

        torch.save({
            'model_state_dict': self._model.state_dict(),
            'feature_size': self._feature_size,
            'mean_reconstruction_error': self._mean_reconstruction_error,
            'std_reconstruction_error': self._std_reconstruction_error,
            'config': {
                'latent_size': self.latent_size
            }
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model from disk."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Cannot load model.")
            return

        checkpoint = torch.load(path, map_location=self._device)

        self._feature_size = checkpoint['feature_size']
        self._mean_reconstruction_error = checkpoint['mean_reconstruction_error']
        self._std_reconstruction_error = checkpoint['std_reconstruction_error']
        config = checkpoint['config']

        self._model = AutoencoderModel(
            input_size=self._feature_size,
            latent_size=config['latent_size']
        ).to(self._device)

        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.eval()
        self._is_fitted = True

        logger.info(f"Model loaded from {path}")
