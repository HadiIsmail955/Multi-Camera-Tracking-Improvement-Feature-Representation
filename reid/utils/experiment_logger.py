import os
import json
import csv
import yaml
import time
import logging
import subprocess
import torch
import sys
import traceback
from datetime import datetime
from pathlib import Path

class ExperimentLogger:
    def __init__(self, base_dir="outputs", exp_name="reid", log_to_console=True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_dir = os.path.join(base_dir, f"{exp_name}_{timestamp}")
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.viz_dir = os.path.join(self.run_dir, "viz")

        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)

        self.closed = False

        self.logger = logging.getLogger(f"experiment_{timestamp}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()

        fmt = logging.Formatter("[%(asctime)s] %(message)s")

        file_handler = logging.FileHandler(
            os.path.join(self.run_dir, "train.log"),
            mode="a",
        )
        file_handler.setFormatter(fmt)
        self.logger.addHandler(file_handler)

        if log_to_console:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(fmt)
            self.logger.addHandler(stream_handler)

        self.csv_file = open(
            os.path.join(self.run_dir, "history.csv"),
            "w",
            newline="",
        )
        self.csv_writer = None

        self.metrics_path = os.path.join(self.run_dir, "metrics.jsonl")

        self.info("=" * 80)
        self.info(f"Run directory: {self.run_dir}")
        self.info("=" * 80)

        self._log_system_info()

    def info(self, message):
        if self.closed:
            return

        self.logger.info(message)

        for handler in self.logger.handlers:
            handler.flush()

    def exception(self, message):
        if self.closed:
            return

        self.logger.error(message)
        self.logger.error(traceback.format_exc())

        for handler in self.logger.handlers:
            handler.flush()

    def _log_system_info(self):
        try:
            python_version = subprocess.check_output(
                ["python", "--version"],
                stderr=subprocess.STDOUT,
            ).decode().strip()
        except Exception:
            python_version = "unknown"

        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            git_commit = "unknown"

        info = {
            "time": time.asctime(),
            "python": python_version,
            "pytorch": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "git_commit": git_commit,
        }

        with open(os.path.join(self.run_dir, "system.txt"), "w") as f:
            for k, v in info.items():
                f.write(f"{k}: {v}\n")

        self.info("System info:")
        for k, v in info.items():
            self.info(f"  {k}: {v}")

    def save_config(self, cfg):
        cfg = dict(cfg)

        with open(os.path.join(self.run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        self.info("Saved config.yaml")

    def log_dataset(self, train_dataset=None, val_dataset=None):
        if train_dataset is not None:
            self.info("Train dataset:")
            self.info(f"  samples: {len(train_dataset)}")
            self.info(f"  classes: {getattr(train_dataset, 'num_classes', 'unknown')}")
            self.info(f"  cameras: {len(getattr(train_dataset, 'camera_to_id', {}))}")
            self.info(f"  scenes: {getattr(train_dataset, 'scene_folders', 'unknown')}")

        if val_dataset is not None:
            self.info("Val dataset:")
            self.info(f"  samples: {len(val_dataset)}")
            self.info(f"  classes: {getattr(val_dataset, 'num_classes', 'unknown')}")
            self.info(f"  cameras: {len(getattr(val_dataset, 'camera_to_id', {}))}")
            self.info(f"  scenes: {getattr(val_dataset, 'scene_folders', 'unknown')}")

    def log_step(self, epoch, step, total_steps, metrics):
        parts = [
            f"epoch={epoch}",
            f"step={step}/{total_steps}",
        ]

        for k, v in metrics.items():
            if isinstance(v, float):
                if "lr" in k:
                    parts.append(f"{k}={v:.8f}")
                else:
                    parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")

        self.info(" | ".join(parts))

    def log_epoch(self, metrics):
        if metrics is None:
            return

        metrics = dict(metrics)

        if not hasattr(self, "metrics_csv_path"):
            if hasattr(self, "metrics_path"):
                self.metrics_csv_path = self.metrics_path
            elif hasattr(self, "exp_dir"):
                self.metrics_csv_path = Path(self.exp_dir) / "metrics.csv"
            else:
                self.metrics_csv_path = Path("metrics.csv")

        csv_path = Path(self.metrics_csv_path)

        if not hasattr(self, "csv_fieldnames") or self.csv_fieldnames is None:
            self.csv_fieldnames = list(metrics.keys())

        new_keys = [k for k in metrics.keys() if k not in self.csv_fieldnames]

        if len(new_keys) > 0:
            old_fieldnames = list(self.csv_fieldnames)
            self.csv_fieldnames.extend(new_keys)

            old_rows = []

            if csv_path.exists() and csv_path.stat().st_size > 0:
                with open(csv_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    old_rows = list(reader)

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=self.csv_fieldnames,
                    extrasaction="ignore",
                )
                writer.writeheader()

                for row in old_rows:
                    fixed_row = {key: row.get(key, "") for key in self.csv_fieldnames}
                    writer.writerow(fixed_row)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.csv_fieldnames,
                extrasaction="ignore",
            )

            if csv_path.stat().st_size == 0:
                writer.writeheader()

            row = {key: metrics.get(key, "") for key in self.csv_fieldnames}
            writer.writerow(row)

        if hasattr(self, "info"):
            msg = " | ".join(
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
            )
            self.info(msg)

    def save_checkpoint(
        self,
        model,
        optimizer=None,
        scheduler=None,
        epoch=None,
        name="last.pt",
        **meta,
    ):
        path = os.path.join(self.ckpt_dir, name)

        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            **meta,
        }

        if optimizer is not None:
            if hasattr(optimizer, "state_dict"):
                ckpt["optimizer"] = optimizer.state_dict()
            else:
                ckpt["optimizer"] = optimizer

        if scheduler is not None:
            if hasattr(scheduler, "state_dict"):
                ckpt["scheduler"] = scheduler.state_dict()
            else:
                ckpt["scheduler"] = scheduler

        torch.save(ckpt, path)
        self.info(f"Saved checkpoint: {path}")

        return path

    def get_viz_dir(self):
        return self.viz_dir

    def get_run_dir(self):
        return self.run_dir

    def get_checkpoint_dir(self):
        return self.ckpt_dir

    def close(self):
        if self.closed:
            return

        self.info("Closing logger.")

        try:
            self.csv_file.close()
        except Exception:
            pass

        for handler in self.logger.handlers:
            handler.flush()
            handler.close()

        self.logger.handlers.clear()
        self.closed = True