from reid.engine.inference import extract_embeddings
from reid.evaluation import evaluate_reid


def validate(
    model,
    query_loader,
    gallery_loader,
    device,
) -> tuple[float, float, float, float]:
    q_embs, q_pids, q_camids = extract_embeddings(
        model=model,
        loader=query_loader,
        device=device,
        show_progress=True,
    )

    g_embs, g_pids, g_camids = extract_embeddings(
        model=model,
        loader=gallery_loader,
        device=device,
        show_progress=True,
    )

    return evaluate_reid(
        q_embs=q_embs,
        q_pids=q_pids,
        q_camids=q_camids,
        g_embs=g_embs,
        g_pids=g_pids,
        g_camids=g_camids,
    )
