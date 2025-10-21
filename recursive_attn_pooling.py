import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from ECAPA_TDNN import ECAPA_TDNN
from configs import paper_config as config
from dataset import SpeakerIdentification

class RecursiveAttnPooling(nn.Module):
    def __init__(self, encoder: nn.Module, config: Dict, device: Optional[torch.device] = None):
        """
        encoder: any encoder returning [B, T, D]
        config: dict with keys
            d_model: encoder output dimension D
            dprime_model: hidden size for attention scoring
            emb_dim: output embedding dimension
            threshold_stop: stopping threshold
        """
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.encoder = ECAPA_TDNN(C=config.C).to(self.device)

        D = config.d_model
        Dp = config.dprime_model
        E = config.emb_dim

        # attention layers
        self.W1 = nn.Linear(3 * D, Dp, bias=False)
        self.Wc = nn.Linear(D, Dp, bias=False)
        self.W2 = nn.Linear(Dp, D)


        # embedding projection
        self.w0 = nn.Linear(2 * D, E, bias=True)

        # coverage init
        self.register_buffer("C0", torch.zeros(D))
        

    # ------------------------------
    # helper: weighted mean/std
    # ------------------------------
    def weighted_stats(self, h: torch.Tensor, a: torch.Tensor):
        """
        h: [B, T, D]
        a: [B, T]
        """
        mu = torch.sum(a * h, dim=1)  # [B, D]
        second_moment = torch.sum(a * (h ** 2), dim=1)  # [B, D]
        var = torch.clamp(second_moment - mu**2, min=1e-8)
        sigma = torch.sqrt(var)
        return mu, sigma

    # ------------------------------
    # helper: attention with coverage
    # ------------------------------
    def calculate_attn(self, h: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, C: torch.Tensor):
        """
        h: [B, T, D]
        mu, sigma: [B, D]
        C: [B, D] coverage
        Returns: A [B, T] attention weights
        """
        B, T, D = h.shape
        mu_exp = mu.unsqueeze(1).expand(-1, T, -1)  # [B, T, D]
        sigma_exp = sigma.unsqueeze(1).expand(-1, T, -1)

        e = torch.cat([h, mu_exp, sigma_exp], dim=-1)  # [B, T, 3D]

        out1 = self.W1(e) + self.Wc(C)  # [B, T, Dp]
        out2 = self.W2(F.relu(out1))              # [B, T, D]
        # breakpoint()

        # Final attention weights
        A = torch.softmax(out2, dim=-1)             # [B, T]
        # A = A / (A.sum(dim=1, keepdim=True) + 1e-8)

        return A, out2 #[B, T, D], [B, T, D]

        # return A, out


    # ------------------------------
    # forward
    # ------------------------------
    def forward(self, x: torch.Tensor, n_sp):
        """
        x: input to encoder
        Returns: [B, N, E] embeddings (N speakers found)
        """
        if x.ndim ==3:
            x = x.mean(axis = 1)
        
        h = self.encoder(x, aug=False)  # [B, D, T]
        h = h.transpose(1, 2)      # [B, T, D]
        B, T, D = h.shape

        C = self.C0.unsqueeze(0).unsqueeze(0).expand(B, T, -1)  # [B, T, D]

        embeddings = []
        count = 0
        #DO INDIVIDUAL STOP FOR EACH OF THE ITEM IN BATCH
        '''
        KEEP A MASK VARIABLE AND APPLY THE MASK AT THE END
        '''

        while count < n_sp:
            # uniform init attention to compute initial mu/sigma
            a_init = torch.ones(B, T, 1, device=h.device) / T
            mu, sigma = self.weighted_stats(h, a_init)

            # attention with coverage
            A, a = self.calculate_attn(h, mu, sigma, C)  # [B, T]

            # speaker-specific stats
            mu_post, sigma_post = self.weighted_stats(h, a)  # [B, D]

            # embedding
            emb = self.w0(torch.cat([mu_post, sigma_post], dim=-1))  # [B, E]
            #normalize the embeddings because ArcFace assumes embeddings lie on a unit hypersphere
            emb = F.normalize(emb, dim = -1)
            embeddings.append(emb)

            # update coverage
 
            C = C + a
            count +=1

        #multiply self.probabilities with mask and embeddings with mask

        embeddings = torch.stack(embeddings, dim=1)  # [B, N, E]


        return embeddings


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    import ast
    speeches_txt = "/nfs/turbo/coe-profdj/txts/voxceleb_test.txt"
    noise_txt = "/scratch/profdj_root/profdj0/sidcs/utils/freesound.txt"
    rir_txt = "/nfs/turbo/coe-profdj/txts/rirs_test.txt"

    with open(f"{speeches_txt}", "r") as f:
        content = f.read()

    speech_files = ast.literal_eval(content)
    with open(noise_txt) as f:
        noise_files = [ln.strip() for ln in f if ln.strip()]
    with open(rir_txt) as f:
        rir_files = [ln.strip() for ln in f if ln.strip()]
    
    dataset = SpeakerIdentification(
        speech_files,
        noise_files,
        rir_files,
        N_max_speakers=4,
        overlap_ratio=0.2,
        desired_duration=8.0,
        sr=16000,
        segment_length=8.0,
        add_noise_prob=0.5,
        overlap_prob=0.5,
        rir_probability=0.5,
        global_snr=(-5, 40),
        peak_normalization=True,
    )
    #get a sample
    sample = dataset[0]
    # breakpoint()
    model = RecursiveAttnPooling(encoder=None, config=config)
    noisy = sample["noisy"].mean(dim = 0).unsqueeze(0)  # [1, wav]
    embeddings, p = model(noisy)

    print(embeddings.shape)