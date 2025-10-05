import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.optimize import linear_sum_assignment
import config

'''
Figure out the scales
'''




class LossWrapper(nn.Module):
    def __init__(self, loss_name, **kwargs):
        super().__init__()  
        config = kwargs.get("config")

        if self.loss_name == 'MagFace':
            self.loss_fn = MagFaceLoss(n_class = kwargs.get("num_class"), alpha=config.alpha, beta = config.beta, 
                                s=config.s, lmbda=config.lmbda, gamma=config.gamma, low=config.low, high=config.high,
                                feat_dim = config.emb_dim, eta=config.eta, xi=config.xi)

        else:
            raise ValueError(f"Unsupported loss function: {loss_fn}")

    def forward(self, x, p, labels):
        loss = self.loss_fn(x, p, labels)
        return loss
    







class MagFaceLoss(nn.Module):
    def __init__(self, n_class, alpha=0.01, beta=100, s=30.0,
                 lmbda=35.0, gamma=0.2, low=10.0, high=110.0, feat_dim=192, eta=100, xi=100, p_stop=0.5,
                c_miss=20.0, c_extra=10.0):
        """
        n_class: number of classes
        alpha, beta: parameters for adaptive margin function
        s: scale of logits
        lmbda: strength of regularization
        gamma: weight for regularization term
        low, high: [l, u] bounds for embedding norms
        feat_dim: feature dimension
        """
        super(MagFaceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.xi = xi
        self.p_stop=torch.tensor(p_stop),
        self.c_miss=c_miss
        self.c_extra=c_extra
        self.s = s
        self.lmbda = lmbda
        self.gamma = gamma
        self.low = low
        self.high = high
        self.device = 'cuda'

        # Class weights
        self.weight = nn.Parameter(torch.FloatTensor(n_class, feat_dim)).to(self.device)
        nn.init.xavier_normal_(self.weight, gain=1.0)

        self.ce = nn.CrossEntropyLoss()

    def magface_single_loss(self, x, label):
        """
        x: [B, D] input embeddings
        label: [B] ground truth class indices
        """
        # Norm of embedding, bounded
        norm_x = torch.norm(x, p=2, dim=1, keepdim=True).clamp(self.low, self.high)  # [B, 1]

        # Adaptive margin: m_i = α * ||x|| + β / ||x||
        m_i = self.alpha * norm_x + self.beta / norm_x  # [B, 1]

        # Cosine similarity with normalized weights
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))  # [B, C]
        sine = torch.sqrt((1.0 - cosine ** 2).clamp(0, 1))

        # Apply margin only to true class
        cos_m = torch.cos(m_i)
        sin_m = torch.sin(m_i)

        # phi = cos(θ + m)
        phi = cosine * cos_m - sine * sin_m

        th = torch.cos(math.pi - m_i)
        mm = torch.sin(math.pi - m_i) * m_i
        phi = torch.where(cosine > th, phi, cosine - mm)

        # One-hot encode labels
        # one_hot = torch.zeros_like(cosine)
        # one_hot.scatter_(1, label.view(-1, 1), 1)
        one_hot = label.float()

        # Replace target class cosine with phi
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        # Cross-entropy loss
        ce_loss = self.ce(output, label)

        # Regularization: encourage norms inside [low, high]
        # Here we use midpoint of [l,u] as "target norm"
        target_r = (self.low + self.high) / 2.0
        reg_loss = ((norm_x - target_r) ** 2).mean()

        loss = ce_loss + self.gamma * self.lmbda *0.01* reg_loss

        return loss

    def forward(self, pred_embs, pred_ps, gt_labels):
        """
        pred_embs: [M, D]  predicted embeddings v^(n)
        pred_ps:   [M]     existence probabilities p_n
        p_stop:    []      stop probability (scalar tensor)
        gt_labels: [S]     ground-truth speaker ids
        pair_loss_fn(x, y): returns scalar loss for embedding x and label y
        """

        device = pred_embs.device
        M = pred_embs.size(0)
        S = gt_labels.size(0)
    
        # 1) Build C (S x M)
        C = torch.empty(S, M, device=device)
        for i in range(S):
            for n in range(M):
                C[i, n] = self.magface_single_loss(pred_embs[n].unsqueeze(0), gt_labels[i].unsqueeze(0))

        # 2) Pad to square with dummies
        if M < S:
            pad = torch.full((S, S-M), self.c_miss, device=device)
            C_pad = torch.cat([C, pad], dim=1)            # S x S
        elif M > S:
            pad = torch.full((M-S, M), self.c_extra, device=device)
            C_pad = torch.cat([C, pad], dim=0)            # M x M
        else:
            C_pad = C                                     # S x S

        # 3) Hungarian on CPU (scipy)
        row_ind, col_ind = linear_sum_assignment(C_pad.detach().cpu().numpy())

        # 4) Compute L_spk and collect existence targets t_n
        L_spk_real = []
        t_exist = torch.zeros(M, device=device, dtype=pred_ps.dtype)

        for r, c in zip(row_ind, col_ind):
            if M <= S:  # padded columns mean misses
                if c < M:           # real pred matched to real GT
                    L_spk_real.append(C[r, c])
                    t_exist[c] = 1  # that slot should exist
                else:
                    # this GT was matched to a dummy column => a miss
                    pass
            else:       # padded rows mean extras
                if r < S:           # real GT matched to real pred
                    L_spk_real.append(C[r, c])
                    t_exist[c] = 1
                else:
                    # dummy row matched to some pred => extra prediction
                    # t_exist stays 0 for that slot
                    pass

        if len(L_spk_real) > 0:
            L_spk = torch.stack(L_spk_real).mean()
        else:
            L_spk = torch.tensor(0.0, device=device)

        # Add explicit miss/extra penalties via assignment size difference
        N_miss = max(0, S - M)
        N_extra = max(0, M - S)
        L_spk = L_spk + self.c_miss * N_miss + self.c_extra * N_extra

        # 5) Existence loss for each predicted slot
        L_exist = F.binary_cross_entropy(pred_ps.clamp(1e-6, 1-1e-6), t_exist)

        # 6) Stop loss: want to stop right after S speakers
        t_stop = torch.tensor(1.0 if M >= S else 0.0, device=device)
        L_stop = F.binary_cross_entropy(self.p_stop.clamp(1e-6, 1-1e-6).to(device), t_stop)

        # 7) Total
        L_total = L_spk + self.eta * L_exist + self.xi * L_stop
        return L_total


if __name__ == "__main__":
    # Test MagFaceLoss
    # breakpoint()
    loss_fn = MagFaceLoss(n_class=1000)
    device='cuda'
    x = torch.randn(4, 192).to(device)
    p = torch.Tensor([0.9, 0.8, 0.7, 0.5]).to(device)
    labels = torch.randint(0, 1000, (4,)).to(device)
    loss = loss_fn.permutation_free_loss_dynamic(x, p, labels)
    print(loss.item())