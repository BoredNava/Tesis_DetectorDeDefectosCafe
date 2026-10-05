"""Small reusable helpers for supervised training loops."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


class EarlyStopping:
    """Stop after ``patience`` validation checks without enough improvement."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        if patience < 1:
            raise ValueError("patience debe ser al menos 1")
        if min_delta < 0:
            raise ValueError("min_delta no puede ser negativo")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.bad_epochs = 0

    def update(self, metric: float) -> tuple[bool, bool]:
        """Return (meaningful_improvement, should_stop). Lower is better."""
        value = float(metric)
        if not torch.isfinite(torch.tensor(value)).item():
            raise ValueError(f"La métrica de validación no es finita: {value}")
        if value < self.best - self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return True, False
        self.bad_epochs += 1
        return False, self.bad_epochs >= self.patience

    def state_dict(self) -> dict[str, float | int]:
        return {"patience": self.patience, "min_delta": self.min_delta,
                "best": self.best, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["patience"]) != self.patience:
            raise ValueError("La patience del checkpoint no coincide con la configuración actual")
        if float(state["min_delta"]) != self.min_delta:
            raise ValueError("El min_delta del checkpoint no coincide con la configuración actual")
        self.best = float(state["best"])
        self.bad_epochs = int(state["bad_epochs"])


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write a checkpoint atomically so interruption cannot leave a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temp)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_json_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def prepare_run_directory(output: Path, resume: bool) -> None:
    """Refuse to overwrite an earlier run; resume only from its last checkpoint."""
    if resume:
        last = output / "last.pt"
        if not last.is_file():
            raise FileNotFoundError(f"No se puede reanudar: falta {last}")
        return
    existing = [name for name in ("best.pt", "last.pt", "history.json")
                if (output / name).exists()]
    if existing:
        raise FileExistsError(
            f"{output} ya contiene {', '.join(existing)}. Usa otra ruta --output "
            "para una corrida nueva o --resume para continuar desde last.pt."
        )
