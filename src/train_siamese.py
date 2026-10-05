"""Train Siamese ResNet18 with early stopping and resumable checkpoints."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data.siamese_dataset import SiamesePairDataset
from src.models.siamese_resnet18 import SiameseResNet18
from src.utils.seed import set_seed
from src.utils.training import (EarlyStopping, atomic_json_save, atomic_torch_save,
                                prepare_run_directory)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/siamese.yaml"
PAIRS = ROOT / "data/siamese_pairs"


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, label):
        distance = nn.functional.pairwise_distance(embedding1, embedding2)
        return (label * distance.pow(2) + (1 - label) *
                torch.clamp(self.margin - distance, min=0).pow(2)).mean()


def best_threshold(distances, labels):
    candidates = sorted({0.0, *[float(x) for x in distances], float(max(distances) + 1e-8)})
    return max(candidates, key=lambda threshold: sum(
        (distance < threshold) == bool(label)
        for distance, label in zip(distances, labels)
    ) / len(labels))


def run_epoch(model, loader, criterion, device, optimizer=None,
              epoch: int = 1, epochs: int = 1, phase: str = "train",
              log_every: int = 10, global_step: int = 0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    count = 0
    distances = []
    labels = []
    with torch.set_grad_enabled(training):
        for iteration, (image1, image2, label) in enumerate(loader, start=1):
            image1 = image1.to(device, non_blocking=device.type == "cuda")
            image2 = image2.to(device, non_blocking=device.type == "cuda")
            label = label.float().to(device, non_blocking=device.type == "cuda")
            if training:
                optimizer.zero_grad(set_to_none=True)
            z1, z2 = model(image1, image2)
            loss = criterion(z1, z2, label)
            if training:
                loss.backward()
                optimizer.step()
                global_step += 1
            distance = nn.functional.pairwise_distance(z1, z2)
            total_loss += loss.item() * label.size(0)
            count += label.size(0)
            distances.extend(distance.detach().cpu().tolist())
            labels.extend(label.int().detach().cpu().tolist())
            if training and (iteration % log_every == 0 or iteration == len(loader)):
                print(f"  época {epoch}/{epochs} · iteración {iteration}/{len(loader)} "
                      f"(global {global_step}) · pérdida lote={loss.item():.4f}", flush=True)
    if count == 0:
        raise ValueError(f"El manifiesto {phase} no contiene pares")
    return total_loss / count, distances, labels, global_step


def train(pair_dir: Path, output: Path, epochs: int, batch: int, lr: float,
          image_size: int, embedding_dim: int, margin: float, seed: int,
          workers: int, pretrained: bool, patience: int, min_delta: float,
          log_every: int, resume: bool = False):
    if epochs < 1 or batch < 1 or log_every < 1:
        raise ValueError("epochs, batch y log_every deben ser al menos 1")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {split: SiamesePairDataset(pair_dir / f"{split}_pairs.csv", image_size,
                                          split == "train")
                for split in ("train", "val")}
    loaders = {split: DataLoader(ds, batch_size=batch, shuffle=(split == "train"),
                                 num_workers=workers, pin_memory=device.type == "cuda")
               for split, ds in datasets.items()}
    model = SiameseResNet18(embedding_dim, pretrained and not resume).to(device)
    criterion = ContrastiveLoss(margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )
    stopper = EarlyStopping(patience=patience, min_delta=min_delta)
    prepare_run_directory(output, resume)
    output.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    best_threshold_value = 0.5
    if resume:
        state = torch.load(output / "last.pt", map_location=device, weights_only=False)
        if state.get("architecture") != "siamese_resnet18":
            raise ValueError("El checkpoint de reanudación no pertenece a ResNet18 siamesa")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        stopper.load_state_dict(state["early_stopping_state"])
        history = state["history"]
        global_step = int(state["global_step"])
        best_val_loss = float(state["best_val_loss"])
        best_threshold_value = float(state["best_threshold"])
        start_epoch = int(state["epoch"]) + 1
        print(f"Reanudando desde época {start_epoch}; iteración global {global_step}.")
        if stopper.bad_epochs >= patience:
            print("La corrida ya alcanzó el criterio de early stopping; no hay entrenamiento pendiente.")
            return
    if start_epoch > epochs:
        print(f"El checkpoint ya alcanzó el límite configurado ({epochs} épocas).")
        return

    if device.type == "cuda":
        print(f"Dispositivo: cuda:0 ({torch.cuda.get_device_name(0)}); "
              f"GPU visibles: {torch.cuda.device_count()}")
    else:
        print("Dispositivo: CPU; CUDA no está disponible para este proceso.")
    print(f"Pares/época: train={len(datasets['train'])}, val={len(datasets['val'])}; "
          f"iteraciones train/época={len(loaders['train'])}; máximo={epochs}; paciencia={patience}.")

    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        train_loss, _, _, global_step = run_epoch(
            model, loaders["train"], criterion, device, optimizer,
            epoch, epochs, "train", log_every, global_step
        )
        val_loss, val_distances, val_labels, _ = run_epoch(
            model, loaders["val"], criterion, device,
            epoch=epoch, epochs=epochs, phase="val", log_every=log_every
        )
        threshold = best_threshold(val_distances, val_labels)
        pair_accuracy = sum((d < threshold) == bool(y)
                            for d, y in zip(val_distances, val_labels)) / len(val_labels)
        row = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_pair_accuracy": pair_accuracy, "val_threshold": threshold,
            "train_iterations": len(loaders["train"]), "global_step": global_step,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.perf_counter() - started,
        }
        scheduler.step(val_loss)
        # Record the rate that will be used for the next epoch.
        row["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        history.append(row)
        _, should_stop = stopper.update(val_loss)
        print(f"época {epoch}/{epochs}: train_loss={train_loss:.4f}; val_loss={val_loss:.4f}; "
              f"val_pair_accuracy={pair_accuracy:.4f}; lr={row['learning_rate']:.2g}; "
              f"tiempo={row['epoch_seconds']:.1f}s; paciencia={stopper.bad_epochs}/{patience}")

        checkpoint = {
            "architecture": "siamese_resnet18", "model_state": model.state_dict(),
            "image_size": image_size, "embedding_dim": embedding_dim,
            "margin": margin, "threshold": threshold, "seed": seed,
            "epoch": epoch, "global_step": global_step,
            "val_loss": val_loss, "val_pair_accuracy": pair_accuracy,
        }
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_threshold_value = threshold
            atomic_torch_save(checkpoint, output / "best.pt")
        last_checkpoint = {
            **checkpoint, "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "early_stopping_state": stopper.state_dict(),
            "best_val_loss": best_val_loss, "best_threshold": best_threshold_value,
            "history": history,
        }
        atomic_torch_save(last_checkpoint, output / "last.pt")
        atomic_json_save(history, output / "history.json")
        with (output / "history.csv.tmp").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        (output / "history.csv.tmp").replace(output / "history.csv")
        if should_stop:
            print(f"Early stopping: val_loss no mejoró al menos {min_delta:g} durante "
                  f"{patience} épocas consecutivas.")
            break

    print(f"Mejor checkpoint: {output / 'best.pt'} (val_loss={best_val_loss:.4f}; "
          f"threshold={best_threshold_value:.4f})")
    print(f"Último checkpoint/reanudación: {output / 'last.pt'}")
    print(f"Historiales: {output / 'history.csv'} y {output / 'history.json'}")


if __name__ == "__main__":
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=PAIRS)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/siamese_resnet18")
    parser.add_argument("--epochs", type=int, default=cfg["epochs"])
    parser.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    parser.add_argument("--lr", type=float, default=cfg["learning_rate"])
    parser.add_argument("--image-size", type=int, default=cfg["img_size"])
    parser.add_argument("--embedding-dim", type=int, default=cfg["embedding_dim"])
    parser.add_argument("--margin", type=float, default=cfg["margin"])
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    parser.add_argument("--workers", type=int, default=cfg["workers"])
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction,
                        default=cfg["pretrained"])
    parser.add_argument("--patience", type=int, default=cfg["early_stopping_patience"])
    parser.add_argument("--min-delta", type=float, default=cfg["early_stopping_min_delta"])
    parser.add_argument("--log-every", type=int, default=cfg["log_every_n_batches"])
    parser.add_argument("--resume", action="store_true", help="Reanuda desde <output>/last.pt")
    args = parser.parse_args()
    train(args.pairs, args.output, args.epochs, args.batch_size, args.lr,
          args.image_size, args.embedding_dim, args.margin, args.seed,
          args.workers, args.pretrained, args.patience, args.min_delta,
          args.log_every, args.resume)
