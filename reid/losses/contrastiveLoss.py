import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = float(margin)

    def forward(self, embeddings, labels):
        labels = labels.long()
        embeddings = F.normalize(embeddings, p=2, dim=1)

        dist = torch.cdist(embeddings, embeddings, p=2)

        labels = labels.view(-1, 1)

        same_id = labels.eq(labels.T)
        self_mask = torch.eye(
            embeddings.size(0),
            device=embeddings.device,
            dtype=torch.bool,
        )

        positive_mask = same_id & ~self_mask
        negative_mask = ~same_id

        has_positive = positive_mask.any(dim=1)
        has_negative = negative_mask.any(dim=1)
        valid = has_positive & has_negative

        if valid.sum() == 0:
            return embeddings.sum() * 0.0

        hardest_positive = dist.masked_fill(
            ~positive_mask,
            -1e9,
        ).max(dim=1)[0]

        hardest_negative = dist.masked_fill(
            ~negative_mask,
            1e9,
        ).min(dim=1)[0]

        loss = F.relu(
            hardest_positive - hardest_negative + self.margin
        )

        return loss[valid].mean()


class CameraWeightedSupConLoss(nn.Module):
    def __init__(
        self,
        temperature=0.07,
        cross_camera_weight=1.0,
        same_camera_weight=0.5,
        occlusion_positive_weight=1.5,
    ):
        super().__init__()

        self.temperature = float(temperature)
        self.cross_camera_weight = float(cross_camera_weight)
        self.same_camera_weight = float(same_camera_weight)
        self.occlusion_positive_weight = float(occlusion_positive_weight)

    def forward(self, embeddings, labels, cameras, is_occluded=None):
        device = embeddings.device

        labels = labels.long()
        cameras = cameras.long()

        embeddings = F.normalize(embeddings, p=2, dim=1)

        logits = torch.matmul(embeddings, embeddings.T)
        logits = logits / self.temperature

        labels = labels.view(-1, 1)
        cameras = cameras.view(-1, 1)

        same_id = labels.eq(labels.T)
        same_camera = cameras.eq(cameras.T)

        self_mask = torch.eye(
            embeddings.size(0),
            device=device,
            dtype=torch.bool,
        )

        positive_mask = same_id & ~self_mask

        cross_camera_positive = positive_mask & ~same_camera
        same_camera_positive = positive_mask & same_camera

        positive_weights = torch.zeros_like(logits)
        positive_weights[cross_camera_positive] = self.cross_camera_weight
        positive_weights[same_camera_positive] = self.same_camera_weight

        if is_occluded is not None and self.occlusion_positive_weight > 0:
            is_occluded = is_occluded.long().view(-1, 1)
            different_occ_state = is_occluded.ne(is_occluded.T)
            clean_occluded_positive = positive_mask & different_occ_state

            # Multiplicative boost preserves cross/same-camera meaning while
            # making clean-occluded positives more important.
            positive_weights[clean_occluded_positive] = (
                positive_weights[clean_occluded_positive]
                * self.occlusion_positive_weight
            )

        positives_per_anchor = positive_weights.sum(dim=1)
        valid = positives_per_anchor > 0

        if valid.sum() == 0:
            return embeddings.sum() * 0.0

        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        exp_logits = torch.exp(logits) * (~self_mask).float()

        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True) + 1e-12
        )

        loss = -(
            positive_weights * log_prob
        ).sum(dim=1) / positives_per_anchor.clamp(min=1e-12)

        return loss[valid].mean()


class OcclusionConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embeddings, labels, is_occluded):
        if is_occluded is None:
            return embeddings.sum() * 0.0

        labels = labels.long().view(-1, 1)
        is_occluded = is_occluded.long().view(-1, 1)

        embeddings = F.normalize(embeddings, p=2, dim=1)
        sim = torch.matmul(embeddings, embeddings.T)

        same_id = labels.eq(labels.T)
        different_occ_state = is_occluded.ne(is_occluded.T)
        self_mask = torch.eye(
            embeddings.size(0),
            device=embeddings.device,
            dtype=torch.bool,
        )

        pair_mask = same_id & different_occ_state & ~self_mask

        if pair_mask.sum() == 0:
            return embeddings.sum() * 0.0

        return (1.0 - sim[pair_mask]).mean()


class ReIDLoss(nn.Module):
    def __init__(
        self,
        id_weight=1.0,
        triplet_weight=1.0,
        contrastive_weight=0.2,
        occlusion_consistency_weight=0.1,
        triplet_margin=0.3,
        temperature=0.07,
        label_smoothing=0.1,
        cross_camera_weight=1.0,
        same_camera_weight=0.5,
        occlusion_positive_weight=1.5,
        metric_embedding_key="bn_embedding",
    ):
        super().__init__()

        self.id_weight = float(id_weight)
        self.triplet_weight = float(triplet_weight)
        self.contrastive_weight = float(contrastive_weight)
        self.occlusion_consistency_weight = float(occlusion_consistency_weight)
        self.metric_embedding_key = metric_embedding_key

        self.id_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
        )

        self.triplet_loss = BatchHardTripletLoss(
            margin=triplet_margin,
        )

        self.contrastive_loss = CameraWeightedSupConLoss(
            temperature=temperature,
            cross_camera_weight=cross_camera_weight,
            same_camera_weight=same_camera_weight,
            occlusion_positive_weight=occlusion_positive_weight,
        )

        self.occlusion_consistency_loss = OcclusionConsistencyLoss()

    def forward(self, outputs, labels, cameras, is_occluded=None):
        labels = labels.long()
        cameras = cameras.long()

        logits = outputs["logits"]

        if self.metric_embedding_key not in outputs:
            raise KeyError(
                f"metric_embedding_key='{self.metric_embedding_key}' not in model outputs. "
                f"Available keys: {list(outputs.keys())}"
            )

        embeddings = outputs[self.metric_embedding_key]

        loss_id = self.id_loss(logits, labels)

        loss_triplet = self.triplet_loss(
            embeddings,
            labels,
        )

        loss_contrastive = self.contrastive_loss(
            embeddings,
            labels,
            cameras,
            is_occluded=is_occluded,
        )

        loss_occlusion = self.occlusion_consistency_loss(
            embeddings,
            labels,
            is_occluded,
        )

        total_loss = (
            self.id_weight * loss_id
            + self.triplet_weight * loss_triplet
            + self.contrastive_weight * loss_contrastive
            + self.occlusion_consistency_weight * loss_occlusion
        )

        return {
            "loss": total_loss,
            "loss_id": loss_id.detach(),
            "loss_triplet": loss_triplet.detach(),
            "loss_contrastive": loss_contrastive.detach(),
            "loss_occlusion": loss_occlusion.detach(),
        }
