import numpy as np
import torch


def eval_precisionk(
    scores: torch.Tensor, 
    labels: torch.Tensor, 
    K: int
):
    score_index = torch.topk(scores, K)[1]
    return torch.sum(labels[score_index]).cpu().item() / K


def eval_recallk(
    scores: torch.Tensor, 
    labels: torch.Tensor, 
    K: int
):
    score_index = torch.topk(scores, K)[1]
    return torch.sum(labels[score_index]).cpu().item() / sum(labels)


def dcg_at_k(relevances, K):
    relevances = np.asarray(relevances)[:K]
    n_relevances = len(relevances)
    if n_relevances == 0:
        return 0.

    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum(relevances / discounts)


def eval_ndcgk(
    pred_probs, 
    true_labels, 
    K: int
):
    ranked_by_pred = np.argsort(pred_probs)[::-1]
    true_relevances = np.take(true_labels, ranked_by_pred)
    
    dcg_max = dcg_at_k(sorted(true_labels, reverse=True), K)
    if dcg_max == 0:
        return 0.
    
    dcg = dcg_at_k(true_relevances, K)
    return dcg / dcg_max
