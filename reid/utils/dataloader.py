from torch.utils.data import DataLoader, Subset, Dataset
from ..dataLoader.sampler.camera_aware_pk_sampler import CameraAwarePKBatchSampler, log_info
import torch

def build_train_loader(train_dataset, args, logger=None):
    use_pk_sampler = getattr(args, "use_pk_sampler", True)

    if use_pk_sampler:
        expected_batch_size = args.pk_identities * args.pk_instances

        if args.batch_size != expected_batch_size:
            log_info(
                logger,
                (
                    f"[WARN] batch_size={args.batch_size}, but PK sampler gives "
                    f"{args.pk_identities} x {args.pk_instances} = "
                    f"{expected_batch_size}. Using PK batch size."
                ),
            )

        batch_sampler = CameraAwarePKBatchSampler(
            dataset=train_dataset,
            num_ids_per_batch=args.pk_identities,
            num_instances_per_id=args.pk_instances,
            camera_aware=True,
            same_camera_instances=args.same_camera_instances,
            occlusion_aware=args.occlusion_aware_sampler,
            normal_instances_per_id=args.normal_instances_per_id,
            occlusion_instances_per_id=args.occlusion_instances_per_id,
        )

        return DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )

    return DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )


def build_eval_loader(dataset, args):
    return DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.workers > 0,
    )


def maybe_subset_eval_dataset(dataset, max_samples: int = 0, seed: int = 42):
    if max_samples is None or max_samples <= 0:
        return dataset

    if len(dataset) <= max_samples:
        return dataset

    generator = torch.Generator()
    generator.manual_seed(seed)

    indices = torch.randperm(len(dataset), generator=generator)[:max_samples]
    indices = indices.tolist()

    return Subset(dataset, indices)


def make_query_gallery_subsets(dataset, max_queries_per_id: int = 1):
    if not hasattr(dataset, "df"):
        raise ValueError("Dataset must expose dataset.df for query/gallery splitting.")

    df = dataset.df.reset_index(drop=True)

    query_indices = []
    gallery_indices = []

    for _, group in df.groupby("label", sort=False):
        indices = list(group.index)

        if len(indices) < 2:
            continue

        num_query = min(max_queries_per_id, len(indices) - 1)
        query = indices[:num_query]
        gallery = indices[num_query:]

        query_indices.extend(query)
        gallery_indices.extend(gallery)

    if len(query_indices) == 0 or len(gallery_indices) == 0:
        raise RuntimeError(
            "Could not build query/gallery subsets. "
            "Validation identities need at least two images each."
        )

    return Subset(dataset, query_indices), Subset(dataset, gallery_indices)
