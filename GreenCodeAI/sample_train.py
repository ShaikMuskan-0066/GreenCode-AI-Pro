"""
sample_train.py — Intentionally inefficient pseudo training script for demos.

This file is NOT meant to be run end-to-end; it exists so `greencheck.py` can
find realistic anti-patterns (large batch, zero workers, no AMP, full FT).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def build_model() -> nn.Module:
    """
    Return a tiny toy model for illustration.
    """
    return nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))


def main() -> None:
    """
    Simulate a training loop with inefficient settings (for GreenCode AI demos).
    """
    torch.manual_seed(0)
    model = build_model()

    # Inefficient: full model fine-tuning flag (read by analyzer heuristics)
    train_full_model = True

    # Inefficient: large batch increases memory spikes and power draw
    batch_size = 256

    # Inefficient: main process loads data alone
    num_workers = 0

    # Inefficient: FP32 only
    mixed_precision = False

    x = torch.randn(512, 10)
    y = torch.randint(0, 2, (512,))
    dataset = TensorDataset(x, y)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # Full-parameter optimizer (energy-heavy on large models)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(2):
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        print(f"epoch={epoch} loss={loss.item():.4f} mixed_precision={mixed_precision} full_ft={train_full_model}")


if __name__ == "__main__":
    main()
