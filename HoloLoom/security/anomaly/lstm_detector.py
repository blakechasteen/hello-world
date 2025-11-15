"""
LSTM Anomaly Detector

Temporal anomaly detection using LSTM (Long Short-Term Memory) networks.

Features:
- Detects sequential patterns and temporal anomalies
- Learns normal behavior sequences
- Predicts next event, scores based on prediction error
- Good for detecting unusual sequences of actions

Requires: PyTorch

Author: HoloLoom Security Team
Created: 2025-11-15
"""

import logging
from typing import List, Optional, Tuple
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
    logging.warning("⚠️  PyTorch not installed. LSTM detector unavailable.")
    # Create dummy classes for type hints
    nn = None

logger = logging.getLogger(__name__)


if TORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        """
        LSTM model for sequence prediction.

        Predicts next event features based on previous sequence.
        """

        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0.0
            )

            self.fc = nn.Linear(hidden_size, input_size)

        def forward(self, x):
            """
            Forward pass.

            Args:
                x: Input tensor (batch_size, sequence_length, input_size)

            Returns:
                Predictions (batch_size, input_size)
            """
            # LSTM forward
            lstm_out, _ = self.lstm(x)

            # Take last output
            last_output = lstm_out[:, -1, :]

            # Predict next event
            prediction = self.fc(last_output)

            return prediction


    class EventSequenceDataset(Dataset):
        """Dataset for event sequences."""

        def __init__(self, sequences: List[np.ndarray], targets: List[np.ndarray]):
            self.sequences = [torch.FloatTensor(s) for s in sequences]
            self.targets = [torch.FloatTensor(t) for t in targets]

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            return self.sequences[idx], self.targets[idx]


class LSTMDetector:
    """
    LSTM-based anomaly detector for temporal patterns.

    Learns to predict next event based on previous sequence.
    Anomaly score = prediction error (higher error = more anomalous).

    Features:
    - Temporal pattern learning
    - Sequence-based anomaly detection
    - Configurable sequence length
    """

    def __init__(
        self,
        sequence_length: int = 10,  # How many previous events to consider
        hidden_size: int = 64,
        num_layers: int = 2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32
    ):
        """
        Initialize LSTM detector.

        Args:
            sequence_length: Number of previous events to consider
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            learning_rate: Learning rate for training
            epochs: Training epochs
            batch_size: Batch size for training
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for LSTM detector. "
                              "Install with: pip install torch")

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # Model (lazy-initialized)
        self._model: Optional['LSTMModel'] = None
        self._is_fitted = False
        self._feature_size: Optional[int] = None

        # Device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"LSTMDetector initialized (sequence_length={sequence_length}, "
                    f"hidden_size={hidden_size}, device={self._device})")

    async def fit(self, events: List['SecurityEvent']):
        """
        Train LSTM on normal event sequences.

        Args:
            events: List of security events (should be chronologically ordered)
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Skipping LSTM training.")
            return

        logger.info(f"Training LSTM on {len(events)} events")

        # Convert events to feature vectors
        features = [event.to_feature_vector() for event in events]
        self._feature_size = len(features[0])

        # Create sequences (sliding window)
        sequences, targets = self._create_sequences(features)

        if len(sequences) < 10:
            logger.warning(f"Too few sequences ({len(sequences)}) for LSTM training. Need at least 10.")
            return

        # Create dataset and dataloader
        dataset = EventSequenceDataset(sequences, targets)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model
        self._model = LSTMModel(
            input_size=self._feature_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self._device)

        # Training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self._model.parameters(), lr=self.learning_rate)

        self._model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_sequences, batch_targets in dataloader:
                batch_sequences = batch_sequences.to(self._device)
                batch_targets = batch_targets.to(self._device)

                # Forward pass
                predictions = self._model(batch_sequences)
                loss = criterion(predictions, batch_targets)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

        self._model.eval()
        self._is_fitted = True
        logger.info("LSTM training complete")

    async def predict(self, event: 'SecurityEvent', history: List['SecurityEvent']) -> float:
        """
        Predict anomaly score for an event given recent history.

        Args:
            event: Current event to score
            history: Recent events (last N events before current)

        Returns:
            Anomaly score (0.0-1.0, based on prediction error)
        """
        if not self._is_fitted or not TORCH_AVAILABLE:
            return 0.0

        # Need enough history
        if len(history) < self.sequence_length:
            logger.warning(f"Insufficient history ({len(history)} < {self.sequence_length}). "
                           f"Returning default score.")
            return 0.0

        # Take last sequence_length events
        recent_history = history[-self.sequence_length:]

        # Convert to feature vectors
        sequence_features = [e.to_feature_vector() for e in recent_history]
        target_features = event.to_feature_vector()

        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence_features).unsqueeze(0).to(self._device)
        target_tensor = torch.FloatTensor(target_features).to(self._device)

        # Predict
        with torch.no_grad():
            prediction = self._model(sequence_tensor).squeeze(0)

        # Compute prediction error (MSE)
        error = torch.mean((prediction - target_tensor) ** 2).item()

        # Normalize error to 0-1 range (heuristic: errors typically < 0.1)
        normalized_score = min(1.0, error / 0.1)

        return float(normalized_score)

    def _create_sequences(
        self,
        features: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Create sliding window sequences for training.

        Args:
            features: List of feature vectors

        Returns:
            Tuple of (sequences, targets)
        """
        sequences = []
        targets = []

        for i in range(len(features) - self.sequence_length):
            sequence = features[i:i + self.sequence_length]
            target = features[i + self.sequence_length]

            sequences.append(np.array(sequence))
            targets.append(target)

        return sequences, targets

    def save_model(self, path: str):
        """Save trained model to disk."""
        if not self._is_fitted or not TORCH_AVAILABLE:
            logger.warning("Model not fitted. Cannot save.")
            return

        torch.save({
            'model_state_dict': self._model.state_dict(),
            'feature_size': self._feature_size,
            'config': {
                'sequence_length': self.sequence_length,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers
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
        config = checkpoint['config']

        self._model = LSTMModel(
            input_size=self._feature_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers']
        ).to(self._device)

        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.eval()
        self._is_fitted = True

        logger.info(f"Model loaded from {path}")
