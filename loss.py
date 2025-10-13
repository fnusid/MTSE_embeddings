import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.optimize import linear_sum_assignment
import config
import numpy as np
'''
Figure out the scales
'''




class LossWrapper(nn.Module):
    def __init__(self, num_class, **kwargs):
        super().__init__()  
        config = kwargs.get("config")
        self.loss_name = kwargs.get("loss_name")
        if self.loss_name == 'MagFace':
            self.loss_fn = MagFaceLoss(n_class = num_class, alpha=kwargs.get("alpha"), beta = kwargs.get("beta"), 
                                s=kwargs.get("s"), lmbda=kwargs.get("lmbda"), gamma=kwargs.get("gamma"), low=kwargs.get("low"), high=kwargs.get("high"),
                                feat_dim = kwargs.get("feat_dim"), eta=kwargs.get("eta"), xi=kwargs.get("xi"))

        elif self.loss_name == "ArcFace":
            self.loss_fn = ArcFaceLoss(n_class = num_class, s = kwargs.get("s"), m = kwargs.get("m"),
                                        alpha = kwargs.get("alpha"), emb_dim = kwargs.get("emb_dim"))
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn}")

    def forward(self, x, p, labels):
        loss = self.loss_fn(x, p, labels)
        return loss
    



class ArcFaceLoss(nn.Module):
    """
    Implementation of the permutation-free ArcFace loss (Eq.15–20).
    Reference: Recursive Attention Pooling / Multi-speaker embedding papers.
    """

    def __init__(self, n_class, s=30.0, m=0.2, alpha=0.1, emb_dim=192):
        """
        n_class: number of speaker classes
        s: ArcFace scale factor
        m: additive angular margin
        alpha: weight for L_cnt
        """
        super().__init__()
        self.s = s
        self.m = m
        self.alpha = alpha

        # Learnable classifier weights
        self.weight = nn.Parameter(torch.FloatTensor(n_class, emb_dim))
        nn.init.xavier_normal_(self.weight)

        # precompute constants
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.m_min, self.m_max, self.m_warmup = 0.2, 0.5, 55     # margin warmup
        self.s_max, self.s_min, self.s_decay_start = 30.0, 20.0, 40  # scale decay

    # ------------------------------------------------------------
    # ArcFace CE loss for a single embedding vs label (Eq.17)
    # ------------------------------------------------------------

    def update_schedules(self, epoch: int):
        """Dynamically update margin and scale."""
        # Margin: linear warmup
        if epoch < self.m_warmup:
            self.m = self.m_min + (self.m_max - self.m_min) * (epoch / self.m_warmup)
        else:
            self.m = self.m_max

        # Scale: cosine decay
        if epoch > self.s_decay_start:
            progress = (epoch - self.s_decay_start) / 50  # spread over 50 epochs
            self.s = self.s_max - (self.s_max - self.s_min) * (
                0.5 * (1 - math.cos(math.pi * min(progress, 1.0)))
            )
        else:
            self.s = self.s_max

    def arcface_ce(self, x, label):
        x = F.normalize(x, dim=-1, eps=1e-6)
        W = F.normalize(self.weight, dim=-1, eps=1e-6)


        cosine = F.linear(x, W).clamp(-1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt(torch.clamp(1.0 - cosine**2, min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        logits = cosine.clone()
        logits[0, label] = phi[0, label]
        logits = logits * self.s

        if not torch.isfinite(logits).all():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e3, neginf=-1e3)

        return F.cross_entropy(logits, torch.tensor([label], device=x.device))

    # ------------------------------------------------------------
    # Main forward (Eq.15–20)
    # ------------------------------------------------------------
    def forward(self, pred_embs, pred_ps, gt_labels):
        """
        pred_embs: [B, N, D] predicted speaker embeddings
        pred_ps:   [B, N+1] existence probabilities (including stop p_{N+1})
        gt_labels: list of one-hot speaker indicators per batch item
        """
        device = pred_embs.device
        B, N, D = pred_embs.shape
        gt_labels = [np.argwhere(item.cpu() == 1).ravel().tolist() for item in gt_labels]

        L_spk_list, L_cnt_list = [], []

        for b in range(B):
            spk_ids = gt_labels[b]
            S = len(spk_ids)
            if S == 0:
                continue

            # ------------------------------------------------
            # (1) Permutation-free L_spk (Eq.16)
            # ------------------------------------------------
            C = torch.empty(S, N, device=device)
            for i in range(S):
                for n in range(N):
                    C[i, n] = self.arcface_ce(pred_embs[b][n].unsqueeze(0), spk_ids[i])


            if not torch.isfinite(C).all():
                C = torch.nan_to_num(C, nan=1e6, posinf=1e6, neginf=1e6)

            # Hungarian min over permutations
            row_ind, col_ind = linear_sum_assignment(C.detach().cpu().numpy())

            L_spk_real = []
            for r, c in zip(row_ind, col_ind):
                if r < S and c < N:
                    L_spk_real.append(C[r, c])

            if len(L_spk_real) > 0:
                L_spk = torch.stack(L_spk_real).mean()
            else:
                L_spk = torch.tensor(0.0, device=device)

            # ------------------------------------------------
            # (2) Counting loss L_cnt (Eq.19–20)
            # ------------------------------------------------
            p = pred_ps[b].clamp(1e-6, 1 - 1e-6)
            if S <= N:  # general Eq.(19)
                # first S → should be 1; next (S+1) → should be 0 (stop)
                valid = p[:S]
                stop = p[S] if S < p.numel() else p[-1]
                L_cnt = -(torch.log(1 - valid).sum() + torch.log(stop)) / (S + 1)
            else:
                # if S > N (shouldn't happen normally)
                L_cnt = torch.tensor(0.0, device=device)

            # simplified alternative for only 1–2 speakers (Eq.20)
            if S == 1 and p.size(0) > 1:
                L_cnt = -torch.log(p[1])
            elif S == 2 and p.size(0) > 1:
                L_cnt = -torch.log(1 - p[1])

            # ------------------------------------------------
            # (3) Total for this batch item (Eq.15)
            # ------------------------------------------------
            L_total = L_spk + self.alpha * L_cnt

            L_spk_list.append(L_spk)
            L_cnt_list.append(L_cnt)

        # ------------------------------------------------
        # Batch means
        # ------------------------------------------------
        L_spk_total = torch.stack(L_spk_list).mean()
        L_cnt_total = torch.stack(L_cnt_list).mean()
        L_total_total = L_spk_total + self.alpha * L_cnt_total

        return L_total_total, {
            "L_spk": L_spk_total,
            "L_cnt": L_cnt_total,
            "L_total": L_total_total,
        }




# if __name__ == "__main__":
#     # Test MagFaceLoss
#     # breakpoint()
#     loss_fn = MagFaceLoss(n_class=1000)
#     device='cuda'
#     x = torch.randn(4, 192).to(device)
#     p = torch.Tensor([0.9, 0.8, 0.7, 0.5]).to(device)
#     labels = torch.randint(0, 1000, (4,)).to(device)
#     loss = loss_fn.permutation_free_loss_dynamic(x, p, labels)
#     print(loss.item())