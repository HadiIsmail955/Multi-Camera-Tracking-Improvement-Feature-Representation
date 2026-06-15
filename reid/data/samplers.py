from torch.utils.data import Sampler
import random


class PKSampler(Sampler):

    def __init__(
        self,
        pid_to_indices: dict,
        P: int = 16,
        K: int = 4,
    ):
        self.pid_to_indices = {
            pid: list(indices)
            for pid, indices in pid_to_indices.items()
        }

        self.pids = list(self.pid_to_indices.keys())

        self.P = P
        self.K = K

        self.num_batches = len(self.pids) // P

    def __iter__(self):

        pids = self.pids.copy()
        random.shuffle(pids)

        for batch_idx in range(self.num_batches):

            batch_pids = pids[
                batch_idx * self.P:
                (batch_idx + 1) * self.P
            ]

            batch = []

            for pid in batch_pids:

                pool = self.pid_to_indices[pid]

                if len(pool) >= self.K:
                    chosen = random.sample(pool, self.K)
                else:
                    chosen = random.choices(pool, k=self.K)

                batch.extend(chosen)

            yield batch

    def __len__(self):
        return self.num_batches