"""Train Slim-CNN with validation-based early stopping and resumable checkpoints."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import yaml
from torch import nn

from src.data.classification import ROOT, make_loaders
from src.models.slim_cnn import SlimCNN
from src.utils.seed import set_seed
from src.utils.training import (EarlyStopping, atomic_json_save, atomic_torch_save,
                                prepare_run_directory)

CONFIG = ROOT / "config/slim_cnn.yaml"


def train(data_dir: Path, output: Path, epochs: int, batch_size: int, lr: float,
          image_size: int, seed: int, workers: int, patience: int,
          min_delta: float, log_every: int, resume: bool = False) -> None:
    if epochs < 1:
        raise ValueError("epochs debe ser al menos 1")
    if log_every < 1:
        raise ValueError("log_every debe ser al menos 1")

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, class_map = make_loaders(data_dir, image_size, batch_size, workers)
    model = SlimCNN(len(class_map), cfg["channels"], cfg["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss()
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
    if resume:
        state = torch.load(output / "last.pt", map_location=device, weights_only=False)
        if state.get("architecture") != "slim_cnn":
            raise ValueError("El checkpoint de reanudación no pertenece a Slim-CNN")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        stopper.load_state_dict(state["early_stopping_state"])
        history = state["history"]
        global_step = int(state["global_step"])
        best_val_loss = float(state["best_val_loss"])
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
    print(f"Imágenes/época: train={len(loaders['train'].dataset)}, val={len(loaders['val'].dataset)}; "
          f"iteraciones train/época={len(loaders['train'])}; máximo={epochs}; paciencia={patience}.")

    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.perf_counter()
        row = {"epoch": epoch}
        for phase in ("train", "val"):
            training = phase == "train"
            model.train(training)
            loss_sum = 0.0
            correct = 0
            count = 0
            loader = loaders[phase]
            with torch.set_grad_enabled(training):
                for iteration, (images, labels) in enumerate(loader, start=1):
                    images = images.to(device, non_blocking=device.type == "cuda")
                    labels = labels.to(device, non_blocking=device.type == "cuda")
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    if training:
                        loss.backward()
                        optimizer.step()
                        global_step += 1
                    batch_count = len(labels)
                    loss_sum += loss.item() * batch_count
                    correct += (logits.argmax(1) == labels).sum().item()
                    count += batch_count
                    if training and (iteration % log_every == 0 or iteration == len(loader)):
                        print(f"  época {epoch}/{epochs} · iteración {iteration}/{len(loader)} "
                              f"(global {global_step}) · pérdida lote={loss.item():.4f}", flush=True)
            if count == 0:
                raise ValueError(f"La partición {phase} está vacía")
            row[f"{phase}_loss"] = loss_sum / count
            row[f"{phase}_accuracy"] = correct / count
            if training:
                row["train_iterations"] = len(loader)

        scheduler.step(row["val_loss"])
        row["global_step"] = global_step
        row["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        row["epoch_seconds"] = time.perf_counter() - epoch_started
        history.append(row)
        improved, should_stop = stopper.update(row["val_loss"])
        print(f"época {epoch}/{epochs}: train_loss={row['train_loss']:.4f}; "
              f"val_loss={row['val_loss']:.4f}; val_accuracy={row['val_accuracy']:.4f}; "
              f"lr={row['learning_rate']:.2g}; tiempo={row['epoch_seconds']:.1f}s; "
              f"paciencia={stopper.bad_epochs}/{patience}")

        checkpoint = {
            "architecture": "slim_cnn", "model_state": model.state_dict(),
            "class_to_idx": class_map, "image_size": image_size,
            "channels": cfg["channels"], "dropout": cfg["dropout"], "seed": seed,
            "epoch": epoch, "global_step": global_step,
            "val_loss": row["val_loss"], "val_accuracy": row["val_accuracy"],
        }
        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            atomic_torch_save(checkpoint, output / "best.pt")

        last_checkpoint = {
            **checkpoint, "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "early_stopping_state": stopper.state_dict(),
            "best_val_loss": best_val_loss, "history": history,
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

    print(f"Mejor checkpoint: {output / 'best.pt'} (val_loss={best_val_loss:.4f})")
    print(f"Último checkpoint/reanudación: {output / 'last.pt'}")
    print(f"Historiales: {output / 'history.csv'} y {output / 'history.json'}")


if __name__ == "__main__":
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/splits")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/slim_cnn")
    parser.add_argument("--epochs", type=int, default=cfg["epochs"])
    parser.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    parser.add_argument("--lr", type=float, default=cfg["learning_rate"])
    parser.add_argument("--image-size", type=int, default=cfg["img_size"])
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    parser.add_argument("--workers", type=int, default=cfg["workers"])
    parser.add_argument("--patience", type=int, default=cfg["early_stopping_patience"])
    parser.add_argument("--min-delta", type=float, default=cfg["early_stopping_min_delta"])
    parser.add_argument("--log-every", type=int, default=cfg["log_every_n_batches"])
    parser.add_argument("--resume", action="store_true", help="Reanuda desde <output>/last.pt")
    args = parser.parse_args()
    train(args.data, args.output, args.epochs, args.batch_size, args.lr,
          args.image_size, args.seed, args.workers, args.patience,
          args.min_delta, args.log_every, args.resume)
