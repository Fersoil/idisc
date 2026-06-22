import torch


def temporal_smoothness_loss(
    depth: torch.Tensor, labels: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    if depth.dim() == 4:
        depth = depth[:, 0]
    T = depth.shape[0]
    if T < 3:
        return depth.new_zeros(())
    ids = labels.unique()
    ids = ids[ids != 0]
    terms = []
    for i in ids:
        present, logd = [], []
        for t in range(T):
            m = labels[t] == i
            present.append(bool(m.any()))
            logd.append(torch.log(depth[t][m].mean() + eps) if m.any()
                        else depth.new_zeros(()))
        for t in range(1, T - 1):
            if present[t - 1] and present[t] and present[t + 1]:
                terms.append((logd[t - 1] - 2 * logd[t] + logd[t + 1]) ** 2)
    if not terms:
        return depth.new_zeros(())
    return torch.stack(terms).mean()
