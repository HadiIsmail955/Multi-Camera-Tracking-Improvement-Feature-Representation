import os
import csv
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


IMG_H = 384
IMG_W = 192
MEAN  = [0.485, 0.456, 0.406]
STD   = [0.229, 0.224, 0.225]

def load_records(csv_path):
    records = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            records.append({
                'filepath': row['filepath'],
                'pid'     : int(row['pid']),
                'camid'   : int(row['camid']),
            })
    return records


class ImageDataset(Dataset):
    def __init__(self, records, transform):
        self.records   = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r   = self.records[idx]
        img = Image.open(r['filepath']).convert('RGB')
        return self.transform(img), r['pid'], r['camid']


def build_loader(records, batch_size=64):
    tf = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    ds = ImageDataset(records, tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True)


def load_model(args, device):
    """
    Two modes:

    1. --pretrained  (zero-shot baseline)
       Loads the backbone with its original pretrained weights.
       No fine-tuning applied. Good for measuring how much training helps.

    2. --checkpoint path  (fine-tuned model, default)
       Loads the backbone architecture then restores the saved state_dict.
       backbone and lora_rank must match those used during training.
    """
    from models import build_model

    if args.pretrained:
        # Build with pretrained weights — num_classes is a dummy value here
        # since we only use the embedding (model.forward), not the classifier
        print(f"\n  Mode     : pretrained weights only (zero-shot baseline)")
        print(f"  Backbone : {args.backbone}")
        model = build_model(
            backbone    = args.backbone,
            num_classes = 751,        # needed to build the head; classifier unused at inference
            lora_rank   = args.lora_rank,
            osnet_weight_path = args.osnet_weights,
        ).to(device)
        if args.backbone == 'osnet' and hasattr(model, 'use_raw_inference'):
            model.use_raw_inference = True
        print(f"  [OK] Pretrained {args.backbone} loaded — no fine-tuning applied.")

    else:
        # Fine-tuned checkpoint
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(
                f"Checkpoint not found: {args.checkpoint}\n"
                f"Either run step5_training.py first, or use --pretrained "
                f"for zero-shot baseline."
            )
        print(f"\n  Mode       : fine-tuned checkpoint")
        print(f"  Checkpoint : {args.checkpoint}")

        ckpt = torch.load(args.checkpoint, map_location=device)

        # Respect the backbone saved in the checkpoint if present
        ckpt_backbone = ckpt.get('backbone', args.backbone)
        if ckpt_backbone != args.backbone:
            print(f"  [NOTE] Checkpoint backbone '{ckpt_backbone}' differs from "
                  f"--backbone '{args.backbone}'. Using checkpoint value.")
            args.backbone = ckpt_backbone

        model = build_model(
            backbone    = args.backbone,
            num_classes = 751,
            lora_rank   = args.lora_rank,
            osnet_weight_path = args.osnet_weights,
        ).to(device)
        model.load_state_dict(ckpt['state_dict'])
        print(f"  [OK] Loaded  epoch={ckpt.get('epoch','?')}  "
              f"Rank-1={ckpt.get('rank1', 0):.4f}  "
              f"mAP={ckpt.get('mAP', 0):.4f}")

    return model

@torch.no_grad()
def extract_embeddings(model, loader, device):
    """Returns (embeddings [N, D], pids [N], camids [N])."""
    model.eval()
    all_embs, all_pids, all_cams = [], [], []
    for imgs, pids, camids in tqdm(loader, desc="Extracting embeddings"):
        embs = model(imgs.to(device)).cpu()
        all_embs.append(embs)
        all_pids.extend(pids.tolist())
        all_cams.extend(camids.tolist())
    return torch.cat(all_embs), all_pids, all_cams

def cosine_distance_matrix(q_embs: torch.Tensor,
                            g_embs: torch.Tensor) -> torch.Tensor:
    """
    Pairwise cosine distance: d = 1 - cosine_similarity ∈ [0, 2]
    Both inputs assumed L2-normalised.
    Returns [Nq, Ng]
    """
    print("Query embeddings shape:", q_embs.shape)
    print("Gallery embeddings shape:", g_embs.shape)
    return (1.0 - q_embs @ g_embs.T).clamp(min=0.0)

def k_reciprocal_rerank(
    q_embs : torch.Tensor,
    g_embs : torch.Tensor,
    k1     : int   = 20,
    k2     : int   = 6,
    lam    : float = 0.3,
) -> torch.Tensor:
    """
    Re-ranking via k-reciprocal encoding.
    Returns re-ranked distance matrix [Nq, Ng].
    """
    Nq = q_embs.shape[0]
    N  = Nq + g_embs.shape[0]

    all_embs  = torch.cat([q_embs, g_embs], dim=0)      # [N, D]
    orig_dist = cosine_distance_matrix(all_embs, all_embs)  # [N, N]
    sorted_idx = torch.argsort(orig_dist, dim=1)

    V = torch.zeros(N, N)

    for i in range(N):
        fwd = sorted_idx[i, 1 : k1 + 1].tolist()
        R   = [j for j in fwd if i in sorted_idx[j, 1:k1+1].tolist()]

        R_expanded = list(R)
        for r in R:
            r_k2 = sorted_idx[r, 1 : k2 // 2 + 1].tolist()
            for r2 in r_k2:
                if r in sorted_idx[r2, 1 : k2 // 2 + 1].tolist():
                    R_expanded.append(r2)
        R_expanded = list(set(R_expanded))

        if R_expanded:
            dists_R = orig_dist[i, R_expanded]
            weights = torch.exp(-dists_R)
            weights = weights / weights.sum()
            for idx_r, w in zip(R_expanded, weights.tolist()):
                V[i, idx_r] = w

    jaccard_sim  = V[:Nq] @ V[Nq:].T
    denom        = (V[:Nq].sum(dim=1, keepdim=True) +
                    V[Nq:].sum(dim=1, keepdim=True).T + 1e-6)
    jaccard_dist = (1.0 - 2.0 * jaccard_sim / denom).clamp(min=0.0)

    orig_qg      = orig_dist[:Nq, Nq:]
    return (1.0 - lam) * jaccard_dist + lam * orig_qg


def rank_gallery(dist_matrix, q_pids, q_cids, g_pids, g_cids,
                 remove_junk=True):
    ranked_results = []
    for i in range(len(q_pids)):
        order = torch.argsort(dist_matrix[i])
        if remove_junk:
            valid = [j for j in order.tolist()
                     if not (g_pids[j] == q_pids[i] and g_cids[j] == q_cids[i])]
        else:
            valid = order.tolist()
        ranked_results.append([g_pids[j] for j in valid])
    return ranked_results

def compute_metrics(ranked_results, q_pids, q_cids, g_pids, g_cids):
    rank1_hits = sum(1 for i, ranked in enumerate(ranked_results)
                     if ranked and ranked[0] == q_pids[i])
    rank1 = rank1_hits / len(q_pids)

    ap_list = []
    for i, ranked in enumerate(ranked_results):
        # Exclude same-camera same-identity gallery images (junk)
        num_gt = sum(
            1
            for p, c in zip(g_pids, g_cids)
            if p == q_pids[i] and not (p == q_pids[i] and c == q_cids[i])
        )
        if num_gt == 0:
            continue
        hits, prec_sum = 0, 0.0
        for rank, pid in enumerate(ranked):
            if pid == q_pids[i]:
                hits += 1
                prec_sum += hits / (rank + 1)
        ap_list.append(prec_sum / num_gt)
    mAP = sum(ap_list) / len(ap_list) if ap_list else 0.0
    return rank1, mAP


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nStep 8: Cross-Camera Matching  |  device={device}")

    model = load_model(args, device)

    query_recs   = load_records(args.query)
    gallery_recs = load_records(args.gallery)

    print(f"\n  Extracting embeddings...")
    print(f"    Query   : {len(query_recs):,} images")
    print(f"    Gallery : {len(gallery_recs):,} images")

    q_embs, q_pids, q_cids = extract_embeddings(
        model, build_loader(query_recs),   device)
    g_embs, g_pids, g_cids = extract_embeddings(
        model, build_loader(gallery_recs), device)

    print(f"    Embedding dim : {q_embs.shape[1]}")

    print(f"\n  Computing distance matrix...")
    if args.rerank:
        print(f"  Applying k-reciprocal re-ranking  "
              f"(k1={args.k1}, k2={args.k2}, lam={args.lam})...")
        dist_matrix = k_reciprocal_rerank(
            q_embs, g_embs, k1=args.k1, k2=args.k2, lam=args.lam)
    else:
        dist_matrix = cosine_distance_matrix(q_embs, g_embs)

    print(f"  Distance matrix : {list(dist_matrix.shape)}  "
          f"min={dist_matrix.min():.4f}  max={dist_matrix.max():.4f}")

    ranked_results = rank_gallery(dist_matrix, q_pids, q_cids,
                                   g_pids, g_cids, remove_junk=True)
    rank1, mAP = compute_metrics(ranked_results, q_pids, q_cids, g_pids, g_cids)

    mode_tag = 'pretrained' if args.pretrained else f'checkpoint'
    rr_tag   = '+rerank' if args.rerank else ''
    print(f"\n  ── Results [{mode_tag}{rr_tag}] ──────────────────")
    print(f"  Rank-1 : {rank1:.4f}  ({rank1*100:.2f}%)")
    print(f"  mAP    : {mAP:.4f}  ({mAP*100:.2f}%)")
    print(f"  ──────────────────────────────────────────")

    torch.save({
        'dist_matrix': dist_matrix,
        'q_pids'     : q_pids, 'q_cids': q_cids,
        'g_pids'     : g_pids, 'g_cids': g_cids,
        'reranked'   : args.rerank,
        'mode'       : mode_tag,
    }, 'distance_matrix.pt')

    out_csv = 'matching_results.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['query_pid', 'query_cam',
                         'top1','top2','top3','top4','top5',
                         'top6','top7','top8','top9','top10',
                         'top1_correct'])
        for i, ranked in enumerate(ranked_results):
            top10   = ranked[:10] + [-1] * max(0, 10 - len(ranked))
            correct = int(top10[0] == q_pids[i]) if top10 else 0
            writer.writerow([q_pids[i], q_cids[i]] + top10 + [correct])

    print(f"\n  Saved: distance_matrix.pt")
    print(f"  Saved: {out_csv}")
    print("\n[Step 8 Complete]")
    print("Next step: python step9_evaluation.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Step 8: Cross-Camera Matching')
    parser.add_argument('--backbone',    type=str,   default='osnet',
                        choices=['osnet', 'dinov3', 'dinov2'])
    parser.add_argument('--lora_rank',   type=int,   default=16)
    parser.add_argument('--osnet_weights', type=str, default=None,
                        help='Optional path to local OSNet ReID checkpoint '
                             '(loads via torchreid.utils.load_pretrained_weights)')
    parser.add_argument('--query',       type=str,   default='query_records.csv')
    parser.add_argument('--gallery',     type=str,   default='gallery_records.csv')

    # Model loading — mutually exclusive
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--checkpoint', type=str,
                             default='checkpoints/best_model.pth',
                             help='Path to fine-tuned checkpoint (.pth)')
    model_group.add_argument('--pretrained', action='store_true',
                             help='Use pretrained backbone weights only '
                                  '(zero-shot baseline, no fine-tuning)')

    # Re-ranking
    parser.add_argument('--rerank', action='store_true',
                        help='Enable k-reciprocal re-ranking')
    parser.add_argument('--k1',  type=int,   default=20)
    parser.add_argument('--k2',  type=int,   default=6)
    parser.add_argument('--lam', type=float, default=0.3)
    args = parser.parse_args()
    main(args)