import os
import csv
import math
import argparse
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image

# Local imports from previous steps
from models import build_model

IMG_H  = 384
IMG_W  = 192
MEAN   = [0.485, 0.456, 0.406]
STD    = [0.229, 0.224, 0.225]

class PKSampler(Sampler):
    """
    Samples P identities, then K images per identity per batch.
    Ensures every batch has exactly P*K items with P distinct PIDs.

    Args:
        pid_to_indices : dict {pid -> [dataset_index, ...]}
        P              : number of identities per batch
        K              : number of images per identity
    """

    def __init__(self, pid_to_indices: dict, P: int = 16, K: int = 4):
        self.pid_to_indices = {
            pid: list(idxs) for pid, idxs in pid_to_indices.items()
        }
        self.pids = list(self.pid_to_indices.keys())
        self.P    = P
        self.K    = K
        # How many complete batches can we form?
        self.num_batches = len(self.pids) // P

    def __iter__(self):
        import random
        pids_shuffled = self.pids.copy()
        random.shuffle(pids_shuffled)

        for i in range(self.num_batches):
            batch_pids = pids_shuffled[i * self.P : (i + 1) * self.P]
            indices = []
            for pid in batch_pids:
                pool = self.pid_to_indices[pid]
                if len(pool) >= self.K:
                    chosen = random.sample(pool, self.K)
                else:
                    # sample with replacement if fewer than K images
                    chosen = random.choices(pool, k=self.K)
                indices.extend(chosen)
            # With DataLoader(batch_sampler=...), each yield must be a full batch
            # of sample indices, not individual indices.
            yield indices

    def __len__(self):
        # For batch_sampler, __len__ should be the number of yielded batches.
        return self.num_batches

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


class ReIDDataset(Dataset):
    def __init__(self, records, transform, relabel=True):
        self.records   = records
        self.transform = transform

        all_pids       = sorted(set(r['pid'] for r in records))
        self.pid2label = {p: i for i, p in enumerate(all_pids)} if relabel else {}
        self.num_classes = len(all_pids)

        # Build pid -> [dataset indices] mapping for PK sampler
        self.pid_to_indices = defaultdict(list)
        for idx, r in enumerate(records):
            self.pid_to_indices[r['pid']].append(idx)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        img   = Image.open(rec['filepath']).convert('RGB')
        img   = self.transform(img)
        label = self.pid2label.get(rec['pid'], rec['pid'])
        return img, label, rec['camid']


class BHTripletLoss(nn.Module):
    """
    Batch Hard Triplet Loss.

    For each anchor i:
        hardest positive: j with same label, max(d_ij)
        hardest negative: k with diff label, min(d_ik)
    Loss: mean( max(0, d_pos - d_neg + margin) )

    Uses squared Euclidean distance on L2-normalised embeddings,
    which is equivalent to 2*(1 - cosine_similarity).
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        # embeddings: [B, D]  already L2-normalised
        # Pairwise squared Euclidean distance
        dot   = embeddings @ embeddings.T                     # [B, B]
        dist  = 2.0 - 2.0 * dot                              # [B, B]  in [0, 4]
        dist  = dist.clamp(min=0.0)                           # numerical safety

        B     = labels.size(0)
        label_mat = labels.unsqueeze(1) == labels.unsqueeze(0)  # [B, B] bool

        # Hardest positive: same label, max distance
        pos_dist = dist.clone()
        pos_dist[~label_mat] = -1e9
        hardest_pos = pos_dist.max(dim=1).values              # [B]

        # Hardest negative: diff label, min distance
        neg_dist = dist.clone()
        neg_dist[label_mat]  = 1e9
        hardest_neg = neg_dist.min(dim=1).values              # [B]

        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return loss.mean()


def cosine_with_warmup(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    all_embs   = []
    all_pids   = []
    all_camids = []
    for imgs, pids, camids in loader:
        imgs = imgs.to(device)
        embs = model(imgs)
        all_embs.append(embs.cpu())
        all_pids.extend(pids.tolist())
        all_camids.extend(camids.tolist())
    return torch.cat(all_embs), all_pids, all_camids


def evaluate(q_embs, q_pids, q_cids, g_embs, g_pids, g_cids):
    """
    Cosine similarity matching, single-query evaluation.
    Returns rank1 (float) and mAP (float).
    """
    # Similarity matrix [Nq, Ng]
    sim    = q_embs @ g_embs.T         # both L2-normalised → cosine sim

    rank1_hits = 0
    ap_list    = []

    for i in range(len(q_pids)):
        qpid, qcid = q_pids[i], q_cids[i]
        scores     = sim[i]            # [Ng]

        # Sort gallery by descending similarity
        order      = torch.argsort(scores, descending=True)

        # Remove same-camera same-identity (junk) matches
        valid_mask = torch.tensor([
            not (g_pids[j] == qpid and g_cids[j] == qcid)
            for j in order.tolist()
        ])
        order_valid = order[valid_mask]

        ranked_pids = [g_pids[j] for j in order_valid.tolist()]

        # Rank-1
        if ranked_pids and ranked_pids[0] == qpid:
            rank1_hits += 1

        # AP
        num_gt = sum(
            1
            for p, c in zip(g_pids, g_cids)
            if p == qpid and not (p == qpid and c == qcid)
        )
        if num_gt == 0:
            continue
        hits = 0
        prec_sum = 0.0
        for rank, pid in enumerate(ranked_pids):
            if pid == qpid:
                hits += 1
                prec_sum += hits / (rank + 1)
        ap_list.append(prec_sum / num_gt)

    rank1 = rank1_hits / len(q_pids)
    mAP   = sum(ap_list) / len(ap_list) if ap_list else 0.0
    return rank1, mAP

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nStep 5: Training  |  backbone={args.backbone}  device={device}")

    # ── Transforms 
    train_tf = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.Pad(10),
        transforms.RandomCrop((IMG_H, IMG_W)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # ── Datasets & loaders 
    train_recs   = load_records(args.train)
    query_recs   = load_records(args.query)
    gallery_recs = load_records(args.gallery)

    train_ds  = ReIDDataset(train_recs,   train_tf, relabel=True)
    query_ds  = ReIDDataset(query_recs,   val_tf,   relabel=False)
    gallery_ds= ReIDDataset(gallery_recs, val_tf,   relabel=False)

    pk_sampler = PKSampler(
        pid_to_indices=train_ds.pid_to_indices,
        P=args.P, K=args.K,
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=pk_sampler,
        num_workers=4, pin_memory=True,
    )
    query_loader = DataLoader(
        query_ds,   batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    gallery_loader = DataLoader(
        gallery_ds, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    print(f"  Train : {len(train_ds):,} imgs | {train_ds.num_classes} IDs | "
          f"{len(train_loader)} batches/epoch  (P={args.P}, K={args.K})")

    # ── Model 
    model = build_model(
        backbone    = args.backbone,
        num_classes = train_ds.num_classes,
        lora_rank   = args.lora_rank,
        osnet_weight_path = args.osnet_weights,
    ).to(device)

    # ── Losses 
    ce_loss      = nn.CrossEntropyLoss(label_smoothing=0.1)
    triplet_loss = BHTripletLoss(margin=args.margin)

    # ── Optimiser — separate LRs for LoRA/head vs backbone 
    lora_head_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and ('lora' in n or 'head' in n or 'classifier' in n)
    ]
    backbone_params  = [
        p for n, p in model.named_parameters()
        if p.requires_grad and ('lora' not in n and 'head' not in n
                                and 'classifier' not in n)
    ]
    optimizer = torch.optim.AdamW([
        {'params': lora_head_params, 'lr': args.lr},
        {'params': backbone_params,  'lr': args.lr * 0.01},
    ], weight_decay=1e-4)

    scheduler = cosine_with_warmup(optimizer, args.warmup, args.epochs)

    # ── Checkpoint dir 
    os.makedirs('checkpoints', exist_ok=True)
    log_rows  = []
    best_mAP  = 0.0

    print("Training started...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0          = time.time()
        total_loss  = 0.0
        ce_running  = 0.0
        tri_running = 0.0
        n_batches   = 0

        for imgs, labels, _ in train_loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)

            embs, logits = model.forward_train(imgs)

            loss_ce  = ce_loss(logits, labels)
            loss_tri = triplet_loss(embs, labels)
            loss     = loss_ce + loss_tri

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss  += loss.item()
            ce_running  += loss_ce.item()
            tri_running += loss_tri.item()
            n_batches   += 1
            print(f"\n  Epoch {epoch:03d}  Batch {n_batches:03d}  "
                  f"loss={loss.item():.4f}  ce={loss_ce.item():.4f}  "
                  f"tri={loss_tri.item():.4f}", end='')

        scheduler.step()

        avg_loss = total_loss  / n_batches
        avg_ce   = ce_running  / n_batches
        avg_tri  = tri_running / n_batches
        lr_now   = optimizer.param_groups[0]['lr']
        elapsed  = time.time() - t0
        print(f"Epoch [{epoch:03d}/{args.epochs}]  "
              f"loss={avg_loss:.4f}  ce={avg_ce:.4f}  tri={avg_tri:.4f}  "
              f"lr={lr_now:.2e}  [{elapsed:.1f}s]")

        # ── Validation every 5 epochs (or last epoch) ────────────────────────
        rank1, mAP = 0.0, 0.0
        if epoch % 100 == 0 or epoch == args.epochs:
            q_embs, q_pids, q_cids = extract_embeddings(model, query_loader,   device)
            g_embs, g_pids, g_cids = extract_embeddings(model, gallery_loader, device)
            rank1, mAP = evaluate(q_embs, q_pids, q_cids, g_embs, g_pids, g_cids)

            if mAP > best_mAP:
                best_mAP = mAP
                torch.save({
                    'epoch'     : epoch,
                    'backbone'  : args.backbone,
                    'state_dict': model.state_dict(),
                    'rank1'     : rank1,
                    'mAP'       : mAP,
                }, 'checkpoints/best_model.pth')

            print(f"Epoch [{epoch:03d}/{args.epochs}]  "
                  f"loss={avg_loss:.4f}  ce={avg_ce:.4f}  tri={avg_tri:.4f}  "
                  f"lr={lr_now:.2e}  Rank-1={rank1:.4f}  mAP={mAP:.4f}  "
                  f"[{elapsed:.1f}s]  ★" if mAP == best_mAP else
                  f"Epoch [{epoch:03d}/{args.epochs}]  "
                  f"loss={avg_loss:.4f}  ce={avg_ce:.4f}  tri={avg_tri:.4f}  "
                  f"lr={lr_now:.2e}  Rank-1={rank1:.4f}  mAP={mAP:.4f}  "
                  f"[{elapsed:.1f}s]")
        else:
            print(f"Epoch [{epoch:03d}/{args.epochs}]  "
                  f"loss={avg_loss:.4f}  ce={avg_ce:.4f}  tri={avg_tri:.4f}  "
                  f"lr={lr_now:.2e}  [{elapsed:.1f}s]")

        log_rows.append({
            'epoch': epoch, 'loss': round(avg_loss, 6),
            'loss_ce': round(avg_ce, 6), 'loss_triplet': round(avg_tri, 6),
            'lr': lr_now, 'rank1': rank1, 'mAP': mAP,
        })

    # ── Save final checkpoint & log 
    torch.save({
        'epoch'     : args.epochs,
        'backbone'  : args.backbone,
        'state_dict': model.state_dict(),
        'rank1'     : rank1,
        'mAP'       : mAP,
    }, 'checkpoints/last_model.pth')

    with open('training_log.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\n[Training Complete]")
    print(f"  Best mAP : {best_mAP:.4f}")
    print(f"  Checkpoints saved in: checkpoints/")
    print(f"  Log saved: training_log.csv")
    print(f"\nNext step: python step6_augmentation.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Step 5: Training')
    parser.add_argument('--backbone',  type=str,   default='dinov3',
                        choices=['osnet', 'dinov3', 'dinov2'])
    parser.add_argument('--train',     type=str,   default='train_records.csv')
    parser.add_argument('--query',     type=str,   default='query_records.csv')
    parser.add_argument('--gallery',   type=str,   default='gallery_records.csv')
    parser.add_argument('--epochs',    type=int,   default=1000)
    parser.add_argument('--warmup',    type=int,   default=10)
    parser.add_argument('--P',         type=int,   default=16,
                        help='Identities per batch')
    parser.add_argument('--K',         type=int,   default=4,
                        help='Images per identity per batch')
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--lam',       type=float, default=1.0,
                        help='Weight of triplet loss')
    parser.add_argument('--margin',    type=float, default=0.3,
                        help='Triplet loss margin')
    parser.add_argument('--lora_rank', type=int,   default=16)
    parser.add_argument('--osnet_weights', type=str, default=None,
                        help='Optional path to local OSNet ReID checkpoint '
                             '(loads via torchreid.utils.load_pretrained_weights)')
    args = parser.parse_args()
    train(args)