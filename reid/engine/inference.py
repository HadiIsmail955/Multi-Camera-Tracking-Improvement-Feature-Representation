# src/engine/inference.py

import torch
from tqdm import tqdm


@torch.no_grad()
def extract_embeddings(
    model,
    loader,
    device,
    show_progress: bool = False,
):
    """
    Extract embeddings for a full DataLoader.

    Returns:
        embeddings: torch.Tensor [N, D]
        pids: list[int]
        camids: list[int]
    """
    model.eval()

    all_embs = []
    all_pids = []
    all_camids = []

    iterator = loader
    if show_progress:
        iterator = tqdm(loader, desc="Extracting embeddings")

    for imgs, pids, camids in iterator:
        imgs = imgs.to(device)
        embs = model(imgs)

        all_embs.append(embs.cpu())
        all_pids.extend(pids.tolist())
        all_camids.extend(camids.tolist())

    return torch.cat(all_embs), all_pids, all_camids
