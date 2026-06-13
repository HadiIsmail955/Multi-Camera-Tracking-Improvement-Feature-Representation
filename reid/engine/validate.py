# src/engine/validate.py

from reid.engine.inference import extract_embeddings
from reid.evaluation import evaluate_reid


def validate(
    model,
    query_loader,
    gallery_loader,
    device,
) -> tuple[float, float]:
    q_embs, q_pids, q_camids = extract_embeddings(
        model=model,
        loader=query_loader,
        device=device,
        show_progress=False,
    )

    g_embs, g_pids, g_camids = extract_embeddings(
        model=model,
        loader=gallery_loader,
        device=device,
        show_progress=False,
    )

    return evaluate_reid(
        q_embs=q_embs,
        q_pids=q_pids,
        q_camids=q_camids,
        g_embs=g_embs,
        g_pids=g_pids,
        g_camids=g_camids,
    )
