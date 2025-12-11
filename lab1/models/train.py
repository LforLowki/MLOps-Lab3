# lab1/models/train.py
"""
Simple transfer-learning training script that logs runs to MLflow and registers the model.
Designed to be short (few epochs) for lab testing.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Tuple
import shutil
import time


import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import OxfordIIITPet

# reproducibility
SEED = 1243
torch.manual_seed(SEED)
random.seed(SEED)

DEFAULT_DATA_DIR = "data/oxford_pets"  # script will download / prepare here
DEFAULT_MLRUNS = "mlruns"



def prepare_datasets(data_root, img_size=224):
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    # Download the dataset
    dataset = OxfordIIITPet(root=str(data_root), download=True)

    train_dir = data_root / "train"
    val_dir = data_root / "val"
    for d in [train_dir, val_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # Correct images directory
    images_dir = data_root / "oxford-iiit-pet" / "images"

    classes = dataset.classes  # list of breed names

    for i, img_path in enumerate(dataset._images):
        label = dataset._labels[i]
        class_name = classes[label]

        split_dir = train_dir if i % 5 != 0 else val_dir
        class_dir = split_dir / class_name
        class_dir.mkdir(exist_ok=True, parents=True)

        # Use only the filename to avoid doubling the path
        src = images_dir / Path(img_path).name
        dst = class_dir / Path(img_path).name

        if not dst.exists():
            shutil.copy(src, dst)

    return train_dir, val_dir, classes




def make_dataloaders(train_dir, val_dir, img_size=224, batch_size=16, num_workers=2):
    tr_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_ds = datasets.ImageFolder(root=str(train_dir), transform=tr_transform)
    val_ds = datasets.ImageFolder(root=str(val_dir), transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_ds.classes


def build_model(num_classes, model_name="mobilenet_v2", pretrained=True):
    if model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        # freeze features
        for p in model.features.parameters():
            p.requires_grad = False
        # replace classifier
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    else:
        raise ValueError("Unknown model_name")


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            running_loss += float(loss.item()) * xb.size(0)
            preds = out.argmax(dim=1)
            correct += int((preds == yb).sum())
            total += xb.size(0)
    acc = correct / total if total > 0 else 0.0
    avg_loss = running_loss / total if total > 0 else 0.0
    return acc, avg_loss


def train_one_run(
    run_name,
    experiment_name="lab3-experiment",
    model_name="mobilenet_v2",
    epochs=1,
    lr=1e-3,
    batch_size=16,
    img_size=224,
    data_root=DEFAULT_DATA_DIR,
):
    mlflow.set_tracking_uri("file://" + str(Path("mlruns").resolve()))
    mlflow.set_experiment(experiment_name)

    train_dir, val_dir, _ = prepare_datasets(data_root, img_size=img_size)
    train_loader, val_loader, classes = make_dataloaders(train_dir, val_dir, img_size=img_size, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(classes), model_name=model_name, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # params and meta
    params = dict(model_name=model_name, epochs=epochs, lr=lr, batch_size=batch_size, seed=SEED, img_size=img_size)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            total = 0
            correct = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item()) * xb.size(0)
                total += xb.size(0)
                correct += int((out.argmax(dim=1) == yb).sum())

            train_loss = running_loss / total if total > 0 else 0.0
            train_acc = correct / total if total > 0 else 0.0

            val_acc, val_loss = evaluate(model, val_loader, device)

            # log metrics per epoch
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

            print(f"Epoch {epoch}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        # save class labels as artifact JSON
        labels_path = Path("labels.json")
        with open(labels_path, "w") as fh:
            json.dump(list(classes), fh)
        mlflow.log_artifact(str(labels_path), artifact_path="labels")

        # log model and register it (same model name so versions increment)
        mlflow.pytorch.log_model(pytorch_model=model.cpu(), artifact_path="model", registered_model_name="lab3_pet_model")

    print("Run complete, registered model (if registration enabled).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--data-root", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-name", default="mobilenet_v2")
    args = parser.parse_args()

    # Create a unique run name based on model, batch size, lr, and timestamp
    run_name = f"{args.model_name}_bs{args.batch_size}_lr{args.lr}_{int(time.time())}"

    train_one_run(
        run_name=run_name,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        img_size=args.img_size,
        data_root=args.data_root,
        model_name=args.model_name,
    )

