import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json

from DogsData import DogsData, get_val_transforms
from Embedding import Embedding
from utils import get_device


def extract(cfg, checkpoint_path, output_dir="embeddings"):
    device = get_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = Embedding(
        backbone=ckpt["cfg"]["backbone"],
        embedding_dim=ckpt["cfg"]["embedding_dim"],
        pretrained=False
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dataset = DogsData(
        cfg["data_root"],
        transform=get_val_transforms(cfg.get("image_size", 224))
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    all_embeddings = []
    all_labels = []
    all_identities = []
    all_paths = []

    with torch.no_grad():
        for batch_idx, (images, labels, identities) in enumerate(tqdm(loader, desc="extracting")):
            images = images.to(device)
            emb = model(images)

            all_embeddings.append(emb.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_identities.extend(identities)

            #get the actual file paths from the dataset
            start = batch_idx * loader.batch_size
            end = start + len(images)
            for i in range(start, min(end, len(dataset.samples))):
                all_paths.append(dataset.samples[i][0])

    embeddings = np.concatenate(all_embeddings, axis=0)

    np.save(output_dir / "embeddings.npy", embeddings)
    np.save(output_dir / "labels.npy", np.array(all_labels))

    with open(output_dir / "identities.json", "w") as f:
        json.dump(all_identities, f)

    with open(output_dir / "paths.json", "w") as f:
        json.dump(all_paths, f)

    print(f"saved {len(embeddings)} embeddings + paths to {output_dir}")

