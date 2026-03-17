"""
SAE Training Pipeline -- loads buffered activations and trains ResearchSAE.

Usage:
    python -m hololoom.dark_trace.sae.training_pipeline \
        --activation-dir ./data/sae_activations/neural_core \
        --output-dir ./data/sae_models \
        --input-dim 384 \
        --expansion-factor 16 \
        --epochs 10
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from hololoom.dark_trace.sae.core import ResearchSAE
from hololoom.dark_trace.sae.trainer import ResearchTrainer

logger = logging.getLogger(__name__)


def load_activation_batches(
    activation_dir: str, source: str = "neural_core"
) -> np.ndarray:
    """Load all .npz activation batches from a source directory."""
    source_dir = Path(activation_dir) / source if source else Path(activation_dir)
    if not source_dir.exists():
        # Try without source subdirectory
        source_dir = Path(activation_dir)

    all_activations = []
    for npz_file in sorted(source_dir.glob("batch_*.npz")):
        data = np.load(npz_file)
        all_activations.append(data["activations"])
        logger.info(f"Loaded {npz_file.name}: {data['activations'].shape}")

    if not all_activations:
        raise FileNotFoundError(f"No activation batches found in {source_dir}")

    combined = np.concatenate(all_activations, axis=0)
    logger.info(f"Total activations: {combined.shape}")
    return combined


def load_governance_labels(
    labels_dir: str,
) -> List[Dict]:
    """Load governance label JSON files produced by GovernanceFeedbackCollector.

    Returns a flat list of labeled samples, each containing 'text',
    'correlation_id', 'activation_source', and 'labels' (list of dicts
    with checkpoint, primary_score, shadow_score, governance_outcome).

    These can be used as weighted cross-entropy targets during supervised
    SAE fine-tuning — each label's primary_score maps to a governance
    feature direction the SAE should learn to separate.
    """
    labels_path = Path(labels_dir)
    if not labels_path.exists():
        logger.warning(f"Labels directory not found: {labels_path}")
        return []

    all_samples: List[Dict] = []
    for json_file in sorted(labels_path.glob("labels_*.json")):
        with open(json_file) as f:
            batch = json.load(f)
        all_samples.extend(batch)
        logger.info(f"Loaded {len(batch)} governance labels from {json_file.name}")

    logger.info(f"Total governance labels: {len(all_samples)}")
    return all_samples


def train_sae(
    activations: np.ndarray,
    input_dim: int = 384,
    expansion_factor: int = 16,
    batch_size: int = 64,
    epochs: int = 10,
    learning_rate: float = 3e-4,
    l1_coefficient: float = 3e-4,
) -> tuple:
    """Train a ResearchSAE on activation data. Returns (sae, metrics)."""

    sae = ResearchSAE(
        input_dim=input_dim,
        expansion_factor=expansion_factor,
        l1_coefficient=l1_coefficient,
    )

    trainer = ResearchTrainer(sae)

    # Create dataloader
    tensor_data = torch.tensor(activations, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    all_metrics = []
    for epoch in range(epochs):
        epoch_metrics = []
        for (batch,) in dataloader:
            metrics = trainer.train_step(batch)
            epoch_metrics.append(metrics)

        avg_loss = sum(m.total_loss for m in epoch_metrics) / len(epoch_metrics)
        avg_l0 = sum(m.l0_sparsity for m in epoch_metrics) / len(epoch_metrics)
        dead = epoch_metrics[-1].dead_fraction

        logger.info(
            f"Epoch {epoch+1}/{epochs}: "
            f"loss={avg_loss:.4f} L0={avg_l0:.1f} dead={dead:.1%}"
        )
        all_metrics.extend(epoch_metrics)

    return sae, all_metrics


def train_multi_scale(
    activation_dir: str,
    output_dir: str,
    scales: List[int] = None,
    expansion_factor: int = 16,
    **train_kwargs,
) -> dict:
    """Train one SAE per Matryoshka scale."""
    if scales is None:
        scales = [96, 192, 384]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: Dict[str, dict] = {}
    for scale in scales:
        source = f"embedding_{scale}"
        logger.info(f"Training SAE for {source} (dim={scale})")

        try:
            activations = load_activation_batches(activation_dir, source)
        except FileNotFoundError:
            logger.warning(f"No activations for {source}, skipping")
            continue

        sae, metrics = train_sae(
            activations,
            input_dim=scale,
            expansion_factor=expansion_factor,
            **train_kwargs,
        )

        model_path = output_path / f"sae_{source}.pt"
        torch.save(sae.state_dict(), model_path)
        logger.info(f"Saved {source} SAE to {model_path}")

        results[source] = {
            "model_path": str(model_path),
            "input_dim": scale,
            "latent_dim": scale * expansion_factor,
            "final_loss": metrics[-1].total_loss if metrics else None,
        }

    # Also train on neural_core activations
    try:
        activations = load_activation_batches(activation_dir, "neural_core")
        sae, metrics = train_sae(
            activations,
            input_dim=384,
            expansion_factor=expansion_factor,
            **train_kwargs,
        )
        model_path = output_path / "sae_neural_core.pt"
        torch.save(sae.state_dict(), model_path)
        results["neural_core"] = {
            "model_path": str(model_path),
            "input_dim": 384,
            "latent_dim": 384 * expansion_factor,
        }
    except FileNotFoundError:
        logger.warning("No neural_core activations, skipping")

    # Train on dot_plasma if available
    try:
        activations = load_activation_batches(activation_dir, "dot_plasma")
        dim = activations.shape[1]
        sae, metrics = train_sae(
            activations,
            input_dim=dim,
            expansion_factor=expansion_factor,
            **train_kwargs,
        )
        model_path = output_path / "sae_dot_plasma.pt"
        torch.save(sae.state_dict(), model_path)
        results["dot_plasma"] = {"model_path": str(model_path), "input_dim": dim}
    except FileNotFoundError:
        logger.warning("No dot_plasma activations, skipping")

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train SAEs on buffered activations")
    parser.add_argument("--activation-dir", default="./data/sae_activations")
    parser.add_argument("--output-dir", default="./data/sae_models")
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="multi",
        help="single: train on neural_core only. multi: train per Matryoshka scale",
    )
    parser.add_argument("--input-dim", type=int, default=384)
    parser.add_argument("--expansion-factor", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l1", type=float, default=3e-4)
    parser.add_argument(
        "--labels-dir",
        default=None,
        help="Directory of governance label JSON files (from GovernanceFeedbackCollector). "
             "Used as weighted cross-entropy targets for supervised fine-tuning.",
    )

    args = parser.parse_args()

    # Load governance labels if provided
    governance_labels = None
    if args.labels_dir:
        governance_labels = load_governance_labels(args.labels_dir)
        if governance_labels:
            logger.info(f"Loaded {len(governance_labels)} governance labels for supervised training")
        else:
            logger.warning("--labels-dir specified but no labels found")

    if args.mode == "single":
        activations = load_activation_batches(args.activation_dir, "neural_core")
        sae, metrics = train_sae(
            activations,
            input_dim=args.input_dim,
            expansion_factor=args.expansion_factor,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            l1_coefficient=args.l1,
        )
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        torch.save(sae.state_dict(), f"{args.output_dir}/sae_neural_core.pt")
    else:
        results = train_multi_scale(
            args.activation_dir,
            args.output_dir,
            expansion_factor=args.expansion_factor,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            l1_coefficient=args.l1,
        )
        print(f"\nTrained {len(results)} SAEs:")
        for source, info in results.items():
            print(f"  {source}: {info}")
