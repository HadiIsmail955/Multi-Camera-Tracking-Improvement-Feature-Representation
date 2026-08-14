import csv
from dataclasses import dataclass


@dataclass(frozen=True)
class Record:
    filepath: str
    pid: int
    camid: int
    tracklet_id: str = ""


def load_records(csv_path: str) -> list[Record]:
    records: list[Record] = []

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            records.append(
                Record(
                    filepath=row["filepath"],
                    pid=int(row["pid"]),
                    camid=int(row["camid"]),
                    tracklet_id=row.get("tracklet_id", ""),
                )
            )

    return records