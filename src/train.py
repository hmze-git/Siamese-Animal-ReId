import argparse
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from DogsData import DogsData, get_train_transforms, get_val_transforms
from Embedding import Embedding
from losses import get_loss
from utils import seed_everything, get_device


def train(cfg):
    seed_everything(cfg.get("seed", 42))
    device = get_device()
    #print("using device:", device)

    #load everything first
    full_dataset = DogsData(cfg["data_root"])
    all_identities = list(full_dataset.identity_to_idx.keys())

    #simple split, 70% of dogs for training
    n_train = int(0.7 * len(all_identities))
    train_ids = set(all_identities[:n_train])
    val_ids = set(all_identities[n_train:])

    train_ds = DogsData(cfg["data_root"], transform=get_train_transforms(cfg["image_size"]), identities=train_ids)
    val_ds = DogsData(cfg["data_root"], transform=get_val_transforms(cfg["image_size"]), identities=val_ids)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)

    #print(f"train dogs: {len(train_ids)} | val dogs: {len(val_ids)}")

    model = Embedding(
        backbone=cfg["backbone"],
        embedding_dim=cfg["embedding_dim"],
        pretrained=True
    ).to(device)

    criterion = get_loss(
        loss_name=cfg["loss"],
        embedding_dim=cfg["embedding_dim"],
        num_classes=len(train_ds.identity_to_idx),
        margin=cfg.get("margin", 0.5),
        scale=cfg.get("scale", 64)
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 1e-4)
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    run_dir = Path("runs") / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg['epochs']}")
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)

            embeddings = model(images)
            loss = criterion(embeddings, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=round(loss.item(), 4))

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"epoch {epoch + 1} | loss: {avg_loss:.4f}")

        #save stuff
        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "cfg": cfg
        }
        torch.save(ckpt, run_dir / "last.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, run_dir / "best.pt")
            #print("saved new best model")

    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"done. checkpoints are in {run_dir}")


