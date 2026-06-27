import random
from collections import defaultdict

from torch.utils.data import Sampler


def log_info(logger, message):
    if logger is not None:
        logger.info(message)
    else:
        print(message)


class CameraAwarePKBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        num_ids_per_batch=8,
        num_instances_per_id=4,
        num_batches=None,
        camera_aware=True,
        same_camera_instances=2,
        occlusion_aware=True,
        normal_instances_per_id=1,
        occlusion_instances_per_id=1,
    ):
        self.dataset = dataset
        self.num_ids_per_batch = int(num_ids_per_batch)
        self.num_instances_per_id = int(num_instances_per_id)
        self.camera_aware = bool(camera_aware)
        self.same_camera_instances = int(same_camera_instances)
        self.occlusion_aware = bool(occlusion_aware)
        self.normal_instances_per_id = int(normal_instances_per_id)
        self.occlusion_instances_per_id = int(occlusion_instances_per_id)

        if not hasattr(dataset, "df"):
            raise ValueError("Dataset must have a dataframe attribute: dataset.df")

        if "label" not in dataset.df.columns:
            raise ValueError("dataset.df must contain a 'label' column")

        if "camera_id" not in dataset.df.columns:
            raise ValueError("dataset.df must contain a 'camera_id' column")

        if self.num_ids_per_batch <= 0:
            raise ValueError("num_ids_per_batch must be > 0")

        if self.num_instances_per_id <= 0:
            raise ValueError("num_instances_per_id must be > 0")

        if self.same_camera_instances <= 0:
            raise ValueError("same_camera_instances must be > 0")

        if self.same_camera_instances > self.num_instances_per_id:
            raise ValueError(
                "same_camera_instances cannot be larger than num_instances_per_id"
            )

        if self.normal_instances_per_id < 0:
            raise ValueError("normal_instances_per_id must be >= 0")

        if self.occlusion_instances_per_id < 0:
            raise ValueError("occlusion_instances_per_id must be >= 0")

        required_occ = self.normal_instances_per_id + self.occlusion_instances_per_id
        if required_occ > self.num_instances_per_id:
            raise ValueError(
                "normal_instances_per_id + occlusion_instances_per_id cannot be "
                "larger than num_instances_per_id"
            )

        self.label_to_indices = defaultdict(list)
        self.label_camera_to_indices = defaultdict(lambda: defaultdict(list))
        self.label_occ_to_indices = defaultdict(lambda: defaultdict(list))
        self.label_occ_camera_to_indices = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        self.path_to_pos = {}

        for pos, (_, row) in enumerate(dataset.df.iterrows()):
            label = int(row["label"])
            camera = int(row["camera_id"])
            is_occluded = int(row.get("is_occluded", 0))

            self.label_to_indices[label].append(pos)
            self.label_camera_to_indices[label][camera].append(pos)
            self.label_occ_to_indices[label][is_occluded].append(pos)
            self.label_occ_camera_to_indices[label][is_occluded][camera].append(pos)

            resolved_path = row.get("crop_path_resolved", None)
            if resolved_path is not None:
                self.path_to_pos[str(resolved_path)] = pos

        self.labels = sorted(self.label_to_indices.keys())

        if len(self.labels) == 0:
            raise ValueError("No valid labels found for PK sampler.")

        batch_size = self.num_ids_per_batch * self.num_instances_per_id

        if num_batches is None:
            self.num_batches = max(len(dataset) // batch_size, 1)
        else:
            self.num_batches = int(num_batches)

    def __len__(self):
        return self.num_batches

    @staticmethod
    def _safe_sample(pool, k):
        pool = list(pool)

        if k <= 0:
            return []

        if len(pool) == 0:
            return []

        if len(pool) >= k:
            return random.sample(pool, k)

        return random.choices(pool, k=k)

    @staticmethod
    def _safe_sample_without_used(pool, k, used):
        candidates = [idx for idx in pool if idx not in used]

        if k <= 0:
            return []

        if len(candidates) == 0:
            return []

        if len(candidates) >= k:
            return random.sample(candidates, k)

        return candidates

    def _try_add_source_clean_pair_for_occlusion(self, label, selected, used):
        if "source_crop_path_resolved" not in self.dataset.df.columns:
            return False

        occ_pool = list(self.label_occ_to_indices[label].get(1, []))
        if len(occ_pool) == 0:
            return False

        random.shuffle(occ_pool)

        for occ_idx in occ_pool:
            if occ_idx in used:
                continue

            row = self.dataset.df.iloc[occ_idx]
            source_path = str(row.get("source_crop_path_resolved", ""))

            if source_path == "" or source_path.lower() == "nan":
                continue

            clean_idx = self.path_to_pos.get(source_path)

            if clean_idx is None or clean_idx in used:
                continue

            clean_row = self.dataset.df.iloc[clean_idx]
            if int(clean_row["label"]) != int(label):
                continue
            if int(clean_row.get("is_occluded", 0)) != 0:
                continue

            selected.extend([clean_idx, occ_idx])
            used.add(clean_idx)
            used.add(occ_idx)
            return True

        return False

    def _sample_required_occlusion_mix(self, label, selected, used):
        if not self.occlusion_aware:
            return

        # Best case: exact clean source + its occluded copy.
        if (
            self.normal_instances_per_id >= 1
            and self.occlusion_instances_per_id >= 1
            and len(selected) + 2 <= self.num_instances_per_id
        ):
            paired = self._try_add_source_clean_pair_for_occlusion(
                label=label,
                selected=selected,
                used=used,
            )
            if paired:
                return

        # Fallback: sample clean and occluded samples from the same identity.
        normal_pool = self.label_occ_to_indices[label].get(0, [])
        occlusion_pool = self.label_occ_to_indices[label].get(1, [])

        if len(normal_pool) > 0:
            need = min(
                self.normal_instances_per_id,
                self.num_instances_per_id - len(selected),
            )
            samples = self._safe_sample_without_used(normal_pool, need, used)
            selected.extend(samples)
            used.update(samples)

        if len(occlusion_pool) > 0:
            need = min(
                self.occlusion_instances_per_id,
                self.num_instances_per_id - len(selected),
            )
            samples = self._safe_sample_without_used(occlusion_pool, need, used)
            selected.extend(samples)
            used.update(samples)

    def _fill_camera_aware(self, label, selected, used):
        all_indices = self.label_to_indices[label]
        camera_to_indices = self.label_camera_to_indices[label]
        cameras = list(camera_to_indices.keys())

        remaining = self.num_instances_per_id - len(selected)
        if remaining <= 0:
            return

        if not self.camera_aware or len(cameras) <= 1:
            extra = self._safe_sample_without_used(all_indices, remaining, used)
            selected.extend(extra)
            used.update(extra)
            return

        # Choose main camera. Prefer a camera that can provide the requested
        # same-camera group after avoiding already selected samples.
        valid_main_cameras = []
        for cam in cameras:
            available = [idx for idx in camera_to_indices[cam] if idx not in used]
            if len(available) >= self.same_camera_instances:
                valid_main_cameras.append(cam)

        main_camera = random.choice(valid_main_cameras) if valid_main_cameras else random.choice(cameras)

        same_cam_k = min(
            self.same_camera_instances,
            self.num_instances_per_id - len(selected),
        )

        same_camera_samples = self._safe_sample_without_used(
            camera_to_indices[main_camera],
            same_cam_k,
            used,
        )
        selected.extend(same_camera_samples)
        used.update(same_camera_samples)

        # Fill remaining from other cameras one-by-one to encourage cross-camera positives.
        remaining = self.num_instances_per_id - len(selected)
        other_cameras = [cam for cam in cameras if cam != main_camera]
        random.shuffle(other_cameras)

        while remaining > 0 and len(other_cameras) > 0:
            added_any = False

            for cam in other_cameras:
                if remaining <= 0:
                    break

                candidates = [idx for idx in camera_to_indices[cam] if idx not in used]

                if len(candidates) == 0:
                    continue

                chosen = random.choice(candidates)
                selected.append(chosen)
                used.add(chosen)
                remaining -= 1
                added_any = True

            if not added_any:
                break

    def _sample_indices_for_label(self, label):
        all_indices = self.label_to_indices[label]

        selected = []
        used = set()

        self._sample_required_occlusion_mix(label, selected, used)
        self._fill_camera_aware(label, selected, used)

        # Fill any remaining slots using unused images from this identity.
        remaining = self.num_instances_per_id - len(selected)

        if remaining > 0:
            extra_samples = self._safe_sample_without_used(
                all_indices,
                remaining,
                used,
            )
            selected.extend(extra_samples)
            used.update(extra_samples)

        # Final fallback: use replacement if the identity has too few images.
        remaining = self.num_instances_per_id - len(selected)

        if remaining > 0:
            selected.extend(random.choices(all_indices, k=remaining))

        # Defensive truncation if a path-pair plus fill overshot.
        return selected[: self.num_instances_per_id]

    def __iter__(self):
        for _ in range(self.num_batches):
            if len(self.labels) >= self.num_ids_per_batch:
                batch_labels = random.sample(
                    self.labels,
                    self.num_ids_per_batch,
                )
            else:
                batch_labels = random.choices(
                    self.labels,
                    k=self.num_ids_per_batch,
                )

            batch_indices = []

            for label in batch_labels:
                indices = self._sample_indices_for_label(label)
                batch_indices.extend(indices)

            random.shuffle(batch_indices)

            yield batch_indices
