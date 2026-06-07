from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class MTMCCSVDataset(Dataset):
    """
    CSV dataset for MTMC/ReID crop training.

    Subproject-2 additions:
      - supports normal crops and occlusion crops from one combined metadata.csv
      - resolves both crops/ and occlusion_crops/ paths correctly
      - exposes is_occluded, data_subset, augmentation, and source_crop_path
      - supports clean-only validation by setting include_occlusion_crops=False
      - keeps scene-aware identity labels to avoid mixing IDs across scenes
    """

    def __init__(
        self,
        root,
        split="train",
        scene_folders=None,
        transform=None,
        min_images_per_id=2,
        min_normal_images_per_id=1,
        object_types=None,
        base_path=".",
        debug=True,
        verify_paths=False,
        max_rows=None,
        scene_aware_ids=True,
        min_cameras_per_id=1,
        include_occlusion_crops=True,
        only_occlusion_crops=False,
        metadata_filename="metadata.csv",
    ):
        self.root = Path(root)
        self.split = split
        self.split_root = self.root / split
        self.transform = transform
        self.base_path = Path(base_path)
        self.debug = debug
        self.verify_paths = verify_paths
        self.scene_aware_ids = scene_aware_ids
        self.min_cameras_per_id = int(min_cameras_per_id)
        self.include_occlusion_crops = bool(include_occlusion_crops)
        self.only_occlusion_crops = bool(only_occlusion_crops)
        self.metadata_filename = metadata_filename

        if self.only_occlusion_crops and not self.include_occlusion_crops:
            raise ValueError(
                "only_occlusion_crops=True requires include_occlusion_crops=True"
            )

        if not self.split_root.exists():
            raise FileNotFoundError(f"Split folder not found: {self.split_root}")

        scene_paths = self._get_scene_paths(scene_folders)
        metadata_files = self._get_metadata_files(scene_paths)

        if len(metadata_files) == 0:
            raise RuntimeError(
                f"No {self.metadata_filename} files found under {self.split_root}"
            )

        df = self._load_metadata(metadata_files)

        if max_rows is not None:
            df = df.head(max_rows).copy()

        required_cols = [
            "crop_path",
            "global_id",
            "camera",
            "frame",
            "object_type",
            "scene_root",
            "scene",
        ]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df = self._normalize_optional_columns(df)

        if debug:
            print("=" * 80, flush=True)
            print("DATASET INITIAL DEBUG", flush=True)
            print("=" * 80, flush=True)
            print("root:", self.root, flush=True)
            print("split:", self.split, flush=True)
            print("split_root:", self.split_root, flush=True)
            print("base_path:", self.base_path, flush=True)
            print("metadata_filename:", self.metadata_filename, flush=True)
            print("rows after csv load:", len(df), flush=True)
            print("object_types argument:", object_types, flush=True)
            print("min_images_per_id:", min_images_per_id, flush=True)
            print("min_normal_images_per_id:", min_normal_images_per_id, flush=True)
            print("min_cameras_per_id:", self.min_cameras_per_id, flush=True)
            print("include_occlusion_crops:", self.include_occlusion_crops, flush=True)
            print("only_occlusion_crops:", self.only_occlusion_crops, flush=True)
            print("verify_paths:", verify_paths, flush=True)
            print("scene_aware_ids:", scene_aware_ids, flush=True)
            print("=" * 80, flush=True)

        df = df.dropna(
            subset=[
                "crop_path",
                "global_id",
                "camera",
                "frame",
                "object_type",
                "scene_root",
                "scene",
            ]
        ).copy()

        df["global_id"] = df["global_id"].astype(str)
        df["camera"] = df["camera"].astype(str)
        df["scene"] = df["scene"].astype(str)
        df["object_type"] = df["object_type"].astype(str)

        if debug:
            print("[DATASET] rows after dropna:", len(df), flush=True)
            print("[DATASET] data_subset counts before filtering:", flush=True)
            print(df["data_subset"].value_counts(dropna=False), flush=True)
            print("[DATASET] is_occluded counts before filtering:", flush=True)
            print(df["is_occluded"].value_counts(dropna=False), flush=True)

        if object_types is not None:
            if isinstance(object_types, str):
                object_types = [object_types]

            object_types = [
                str(x).strip()
                for x in object_types
                if x is not None and str(x).strip() != ""
            ]

            if len(object_types) > 0:
                before = len(df)
                df = df[df["object_type"].isin(object_types)].copy()

                if debug:
                    print(
                        f"[DATASET] object_types filter {object_types}: "
                        f"{before} -> {len(df)}",
                        flush=True,
                    )
            elif debug:
                print(
                    "[DATASET] Empty object_types list detected. "
                    "Skipping object type filter.",
                    flush=True,
                )

        if len(df) == 0:
            raise RuntimeError(
                "No samples left after object type filtering. "
                f"object_types={object_types}"
            )

        # Occlusion inclusion filter.
        if self.only_occlusion_crops:
            before = len(df)
            df = df[df["is_occluded"] == 1].copy()
            if debug:
                print(
                    f"[DATASET] only occlusion crops filter: {before} -> {len(df)}",
                    flush=True,
                )
        elif not self.include_occlusion_crops:
            before = len(df)
            df = df[df["is_occluded"] == 0].copy()
            if debug:
                print(
                    f"[DATASET] clean-only filter: {before} -> {len(df)}",
                    flush=True,
                )

        if len(df) == 0:
            raise RuntimeError(
                "No samples left after occlusion filtering. "
                f"include_occlusion_crops={self.include_occlusion_crops}, "
                f"only_occlusion_crops={self.only_occlusion_crops}"
            )

        if debug:
            print("[DATASET] available object types:", flush=True)
            print(df["object_type"].value_counts(), flush=True)

        # Scene-aware identity prevents Person_0001 from different scenes
        # being treated as the same real person/object.
        if scene_aware_ids:
            df["identity_key"] = (
                df["scene"].astype(str) + "__" + df["global_id"].astype(str)
            )
        else:
            df["identity_key"] = df["global_id"].astype(str)

        # Resolve image paths after scene_root is known.
        df["crop_path_resolved"] = df.apply(
            self._resolve_crop_path_from_row,
            axis=1,
        )
        df["source_crop_path_resolved"] = df.apply(
            self._resolve_source_crop_path_from_row,
            axis=1,
        )

        if debug:
            print("=" * 80, flush=True)
            print("DATASET PATH DEBUG", flush=True)
            print("=" * 80, flush=True)
            print("scene folders:", [p.name for p in scene_paths], flush=True)
            print("metadata files:", len(metadata_files), flush=True)
            print("rows before path verification:", len(df), flush=True)

            print("Example paths:", flush=True)
            for _, row in df.head(5).iterrows():
                p = row["crop_path_resolved"]
                print("raw:", row["crop_path"], flush=True)
                print("scene_root:", row["scene_root"], flush=True)
                print("resolved:", p, flush=True)
                print("exists:", Path(p).exists(), flush=True)
                print("data_subset:", row["data_subset"], flush=True)
                print("is_occluded:", row["is_occluded"], flush=True)
                print("source_crop_path_resolved:", row["source_crop_path_resolved"], flush=True)
                print("-" * 40, flush=True)

            print("=" * 80, flush=True)

        if verify_paths:
            before = len(df)

            exists_mask = df["crop_path_resolved"].apply(
                lambda p: Path(p).exists()
            )

            valid_count = int(exists_mask.sum())
            invalid_count = int((~exists_mask).sum())

            if debug:
                print("=" * 80, flush=True)
                print("FULL PATH VERIFICATION", flush=True)
                print("=" * 80, flush=True)
                print("before:", before, flush=True)
                print("valid:", valid_count, flush=True)
                print("invalid:", invalid_count, flush=True)

                if invalid_count > 0:
                    print("First missing paths:", flush=True)
                    missing_df = df[~exists_mask].head(10)

                    for _, row in missing_df.iterrows():
                        print("raw:", row["crop_path"], flush=True)
                        print("resolved:", row["crop_path_resolved"], flush=True)
                        print("data_subset:", row["data_subset"], flush=True)
                        print("-" * 40, flush=True)

                print("=" * 80, flush=True)

            df = df[exists_mask].copy()

        if len(df) == 0:
            raise RuntimeError(
                "No valid samples after path verification. "
                "Check crop_path, scene_root, and verify_paths."
            )

        if debug:
            print("[DATASET] rows before ID filter:", len(df), flush=True)
            print(
                "[DATASET] unique global IDs before ID filter:",
                df["global_id"].nunique(),
                flush=True,
            )
            print(
                "[DATASET] unique identity keys before ID filter:",
                df["identity_key"].nunique(),
                flush=True,
            )
            print("[DATASET] top identity counts:", flush=True)
            print(df["identity_key"].value_counts().head(10), flush=True)

        # Filter identities by total sample count.
        counts = df["identity_key"].value_counts()
        valid_ids = counts[counts >= int(min_images_per_id)].index

        before_id_filter = len(df)
        df = df[df["identity_key"].isin(valid_ids)].copy()

        if debug:
            print(
                f"[DATASET] ID image-count filter min_images_per_id={min_images_per_id}: "
                f"{before_id_filter} -> {len(df)}",
                flush=True,
            )

        # Prevent synthetic occlusion copies from being the only reason an ID is valid.
        if int(min_normal_images_per_id) > 0:
            before_normal_filter = len(df)
            normal_counts = (
                df[df["is_occluded"] == 0]["identity_key"].value_counts()
            )
            valid_normal_ids = normal_counts[
                normal_counts >= int(min_normal_images_per_id)
            ].index
            df = df[df["identity_key"].isin(valid_normal_ids)].copy()

            if debug:
                print(
                    f"[DATASET] normal-image filter min_normal_images_per_id={min_normal_images_per_id}: "
                    f"{before_normal_filter} -> {len(df)}",
                    flush=True,
                )

        if self.min_cameras_per_id > 1:
            before_camera_filter = len(df)

            # Use clean crops when available so occlusion duplicates do not inflate camera coverage.
            clean_df = df[df["is_occluded"] == 0]
            camera_source_df = clean_df if len(clean_df) > 0 else df

            camera_counts = (
                camera_source_df.groupby("identity_key")["camera"]
                .nunique()
            )

            valid_camera_ids = camera_counts[
                camera_counts >= self.min_cameras_per_id
            ].index

            df = df[df["identity_key"].isin(valid_camera_ids)].copy()

            if debug:
                print(
                    f"[DATASET] ID camera-count filter min_cameras_per_id={self.min_cameras_per_id}: "
                    f"{before_camera_filter} -> {len(df)}",
                    flush=True,
                )

        if len(df) == 0:
            raise RuntimeError(
                "No valid samples after filtering.\n"
                f"object_types={object_types}\n"
                f"min_images_per_id={min_images_per_id}\n"
                f"min_normal_images_per_id={min_normal_images_per_id}\n"
                f"min_cameras_per_id={self.min_cameras_per_id}\n"
                f"include_occlusion_crops={self.include_occlusion_crops}\n"
                f"only_occlusion_crops={self.only_occlusion_crops}\n"
                f"verify_paths={verify_paths}\n"
                f"scene_aware_ids={scene_aware_ids}\n"
                "Most common causes:\n"
                "1. verify_paths=True removed all rows.\n"
                "2. object_types filter removed all rows.\n"
                "3. min_images_per_id/min_normal_images_per_id is too high.\n"
                "4. min_cameras_per_id is too high.\n"
                "5. crop paths do not match crops/ or occlusion_crops/.\n"
            )

        unique_ids = sorted(df["identity_key"].unique())

        self.id_to_label = {
            identity_key: idx
            for idx, identity_key in enumerate(unique_ids)
        }

        self.label_to_id = {
            idx: identity_key
            for identity_key, idx in self.id_to_label.items()
        }

        df["label"] = df["identity_key"].map(self.id_to_label)

        unique_cameras = sorted(df["camera"].unique())

        self.camera_to_id = {
            camera: idx
            for idx, camera in enumerate(unique_cameras)
        }

        df["camera_id"] = df["camera"].map(self.camera_to_id)

        self.df = df.reset_index(drop=True)
        self.num_classes = len(self.id_to_label)
        self.scene_folders = [p.name for p in scene_paths]

        if debug:
            print("=" * 80, flush=True)
            print("FINAL DATASET READY", flush=True)
            print("=" * 80, flush=True)
            print("samples:", len(self.df), flush=True)
            print("classes:", self.num_classes, flush=True)
            print("cameras:", len(self.camera_to_id), flush=True)
            print("scene_folders:", self.scene_folders, flush=True)
            print("scene_aware_ids:", self.scene_aware_ids, flush=True)
            print("object types:", flush=True)
            print(self.df["object_type"].value_counts(), flush=True)
            print("data subsets:", flush=True)
            print(self.df["data_subset"].value_counts(dropna=False), flush=True)
            print("is_occluded:", flush=True)
            print(self.df["is_occluded"].value_counts(dropna=False), flush=True)
            print("top final identities:", flush=True)
            print(self.df["identity_key"].value_counts().head(10), flush=True)
            print("=" * 80, flush=True)

    def _normalize_optional_columns(self, df):
        df = df.copy()

        if "data_subset" not in df.columns:
            df["data_subset"] = "normal"

        if "augmentation" not in df.columns:
            df["augmentation"] = "none"

        if "source_crop_path" not in df.columns:
            df["source_crop_path"] = ""

        df["data_subset"] = (
            df["data_subset"]
            .fillna("normal")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        df["augmentation"] = (
            df["augmentation"]
            .fillna("none")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        df["source_crop_path"] = df["source_crop_path"].fillna("").astype(str)

        crop_path_lower = df["crop_path"].fillna("").astype(str).str.replace("\\\\", "/", regex=False).str.lower()

        # Robust occlusion detection. This supports old metadata and new metadata.
        is_occ = (
            df["data_subset"].eq("occlusion")
            | df["augmentation"].str.contains("occlusion", na=False)
            | crop_path_lower.str.contains("occlusion_crops/", na=False)
        )

        df["is_occluded"] = is_occ.astype(int)

        # Normalize empty or unknown subset labels.
        df.loc[df["is_occluded"].eq(1), "data_subset"] = "occlusion"
        df.loc[df["is_occluded"].eq(0), "data_subset"] = "normal"

        return df

    def _get_scene_paths(self, scene_folders):
        if scene_folders is None:
            return sorted([
                p for p in self.split_root.iterdir()
                if p.is_dir()
            ])

        if isinstance(scene_folders, str):
            scene_folders = [scene_folders]

        scene_paths = [self.split_root / name for name in scene_folders]

        missing = [str(p) for p in scene_paths if not p.exists()]

        if missing:
            raise FileNotFoundError(f"Scene folders not found: {missing}")

        return scene_paths

    def _get_metadata_files(self, scene_paths):
        metadata_files = []

        for scene_path in scene_paths:
            metadata_file = scene_path / self.metadata_filename

            if metadata_file.exists():
                metadata_files.append(metadata_file)
            else:
                print(
                    f"[WARNING] {self.metadata_filename} not found: {scene_path}",
                    flush=True,
                )

        return metadata_files

    def _load_metadata(self, metadata_files):
        dfs = []

        for metadata_file in metadata_files:
            df_part = pd.read_csv(metadata_file)

            print(
                f"[INFO] Loaded metadata: {metadata_file} rows={len(df_part)}",
                flush=True,
            )

            if len(df_part) == 0:
                print(
                    f"[WARNING] Empty metadata skipped: {metadata_file}",
                    flush=True,
                )
                continue

            df_part["source_csv"] = str(metadata_file)
            df_part["scene"] = metadata_file.parent.name
            df_part["scene_root"] = str(metadata_file.parent.resolve())

            dfs.append(df_part)

        if len(dfs) == 0:
            raise RuntimeError("All metadata.csv files are empty.")

        return pd.concat(dfs, ignore_index=True)

    def _resolve_named_path_inside_scene(self, raw_path, scene_root):
        raw_path = "" if raw_path is None else str(raw_path)
        scene_root = Path(scene_root)

        if raw_path.strip() == "" or raw_path.lower() == "nan":
            return ""

        normalized = raw_path.replace("\\", "/").replace("./", "")
        lower_normalized = normalized.lower()

        # IMPORTANT: check occlusion_crops before crops because
        # 'occlusion_crops/' also contains the substring 'crops/'.
        token = "occlusion_crops/"
        if token in lower_normalized:
            start = lower_normalized.index(token) + len(token)
            suffix = normalized[start:]
            return str(scene_root / "occlusion_crops" / suffix)

        token = "crops/"
        if token in lower_normalized:
            start = lower_normalized.index(token) + len(token)
            suffix = normalized[start:]
            return str(scene_root / "crops" / suffix)

        p = Path(normalized)

        if p.is_absolute():
            return str(p)

        return str(self.base_path / p)

    def _resolve_crop_path_from_row(self, row):
        return self._resolve_named_path_inside_scene(
            row["crop_path"],
            row["scene_root"],
        )

    def _resolve_source_crop_path_from_row(self, row):
        source_path = row.get("source_crop_path", "")
        return self._resolve_named_path_inside_scene(
            source_path,
            row["scene_root"],
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = Path(row["crop_path_resolved"])

        if not image_path.exists():
            raise FileNotFoundError(
                "Image not found during DataLoader access.\n"
                f"Resolved path: {image_path}\n"
                f"Raw CSV path: {row['crop_path']}\n"
                f"Scene root: {row['scene_root']}\n"
                f"Current working directory: {Path.cwd()}\n"
                f"base_path: {self.base_path}\n"
            )

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "camera_id": torch.tensor(int(row["camera_id"]), dtype=torch.long),
            "is_occluded": torch.tensor(int(row["is_occluded"]), dtype=torch.long),
            "identity_key": row["identity_key"],
            "global_id": row["global_id"],
            "camera": row["camera"],
            "frame": int(row["frame"]),
            "object_type": row["object_type"],
            "scene": row["scene"],
            "data_subset": row["data_subset"],
            "augmentation": row["augmentation"],
            "crop_path": str(image_path),
            "source_crop_path": str(row.get("source_crop_path_resolved", "")),
        }
