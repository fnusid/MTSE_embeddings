import numpy as np
import ot  # pip install POT
from configs import paper_config as config
import torch
import torch.nn as nn
# embeddings: preds [B,x,d], gts [B,y,d]; confidences p in [0,1]^x (optional)


class MetricsWrapper(nn.Module):
    def __init__(self, metrics):
        super(MetricsWrapper, self).__init__()
        self.metrics_names = metrics
            
    def forward(self, preds, gts, p=None):
        metric_values = {}
        for i, metric in enumerate(self.metrics_names):
            if metric == 'OT':
                costs = []
                for b in range(preds.shape[0]):
                    cost = ot_cost(preds[b], gts[b], None if p is None else p[b])
                    costs.append(cost)
                metric_values['OT'] = np.mean(costs).item()
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        return metric_values



def ot_cost(preds, gts, p=None):
    # weights
    eps = config.eps
    tau = config.tau
    x = preds.shape[0]
    y = gts.shape[0]
    a = np.ones((x))/x if p is None else (p/np.sum(p))
    b = np.ones((y))/y

    # cosine cost in [0,2]
    preds_n = preds / (np.linalg.norm(preds,axis=-1,keepdims=True)+1e-12)
    gts_n   = gts   / (np.linalg.norm(gts,axis=-1,keepdims=True)+1e-12)
    C = 1.0 - preds_n @ gts_n.T

    # unbalanced entropic OT
    # If you want balanced: ot.sinkhorn(a, b, C, reg=eps)
    P = ot.unbalanced.sinkhorn_unbalanced(a, b, C, reg=eps, reg_m=tau)
    cost = np.sum(P * C)
    return cost  # lower is better



if __name__ == "__main__":
    # test
    B, x, y, d = 2, 5, 7, 3
    preds = np.random.randn(B,x,d)
    gts   = np.random.randn(B,y,d)
    for i in range(B):
        cost = ot_cost(preds[i], gts[i])
        print(cost)
    