import csv


def make_log_row(
    epoch: int,
    loss: float,
    loss_ce: float,
    loss_triplet: float,
    loss_supcon: float,
    loss_arcface: float,
    lr: float,
    rank1: float = 0.0,
    mAP: float = 0.0,
) -> dict:
    return {
        "epoch": epoch,
        "loss": round(loss, 6),
        "loss_ce": round(loss_ce, 6),
        "loss_triplet": round(loss_triplet, 6),
        "loss_supcon": round(loss_supcon, 6),
        "loss_arcface": round(loss_arcface, 6),
        "lr": lr,
        "rank1": rank1,
        "mAP": mAP,
    }


def write_training_log(
    path: str,
    rows: list[dict],
):
    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)
