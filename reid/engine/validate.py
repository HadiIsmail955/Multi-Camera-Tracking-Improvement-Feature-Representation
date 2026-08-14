from reid.engine.inference import extract_embeddings
from reid.evaluation import evaluate_reid


def validate(
    model,
    query_loader,
    gallery_loader,
    device,
    return_analysis: bool = False,
):
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

    eval_res = evaluate_reid(
        q_embs=q_embs,
        q_pids=q_pids,
        q_camids=q_camids,
        g_embs=g_embs,
        g_pids=g_pids,
        g_camids=g_camids,
        return_analysis=True,
    )

    if return_analysis:
        return eval_res

    rank1, rank5, rank10, mAP = eval_res[:4]
    return rank1, rank5, rank10, mAP

