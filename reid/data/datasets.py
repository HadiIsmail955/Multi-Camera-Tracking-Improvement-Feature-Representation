from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset


class ReIDDataset(Dataset):
    def __init__(
        self,
        records,
        transform,
        relabel: bool = False,
    ):
        self.records = records
        self.transform = transform

        all_pids = sorted({r.pid for r in records})

        self.pid2label = (
            {pid: idx for idx, pid in enumerate(all_pids)}
            if relabel
            else {}
        )

        self.num_classes = len(all_pids)
        self.pid_to_indices = defaultdict(list)

        for idx, record in enumerate(records):
            self.pid_to_indices[record.pid].append(idx)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        image = Image.open(record.filepath).convert("RGB")
        image = self.transform(image)

        label = self.pid2label.get(record.pid, record.pid)

        return image, label, record.camid