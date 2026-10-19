import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import json
from sklearn.metrics.pairwise import cosine_similarity

from DogsData import get_val_transforms
from Embedding import Embedding
from utils import get_device


def load_gallery(emb_dir="embeddings"):
    emb_dir = Path(emb_dir)
    embeddings = np.load(emb_dir / "embeddings.npy")
    labels = np.load(emb_dir / "labels.npy")

    with open(emb_dir / "identities.json") as f:
        identities = json.load(f)

    return embeddings, labels, identities


def search(query_path, checkpoint, emb_dir="embeddings", top_k=5):
    device = get_device()

    #load model
    ckpt = torch.load(checkpoint, map_location=device)
    model = Embedding(
        backbone=ckpt["cfg"]["backbone"],
        embedding_dim=ckpt["cfg"]["embedding_dim"],
        pretrained=False
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    #load gallery
    gallery_embs, gallery_labels, gallery_ids = load_gallery(emb_dir)

    #process query image
    transform = get_val_transforms(ckpt["cfg"].get("image_size", 224))
    img = np.array(Image.open(query_path).convert("RGB"))
    img = transform(image=img)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        query_emb = model(img).cpu().numpy()

    #cosine similarity (since we already l2-normalized)
    sims = cosine_similarity(query_emb, gallery_embs)[0]

    #get top k
    top_indices = np.argsort(sims)[::-1][:top_k]

    print(f"\nquery: {query_path}")
    print("-" * 40)
    for rank, idx in enumerate(top_indices, 1):
        print(f"{rank}. {gallery_ids[idx]}  |  score: {sims[idx]:.4f}")

    return top_indices, sims[top_indices]

