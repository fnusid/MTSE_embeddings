import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
import tqdm
from pathlib import Path
import pandas as pd
import csv
from recursive_attn_pooling import RecursiveAttnPooling
from configs import paper_config as config
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SpeakerVerificationDataset(Dataset):
    def __init__(self, trials_txt, base_dir, target_sr=16000):
        """
        Args:
            trials_txt (str): path to trials file (e.g., trials.txt)
            base_dir (str): directory containing all wavs
            target_sr (int): resample target sample rate
        """
        super().__init__()
        self.base_dir = base_dir
        self.target_sr = target_sr

        # Read file lines
        self.samples = []
        with open(trials_txt, "r", newline='') as f:
            reader = csv.DictReader(f)
            for parts in reader:
                if not parts or len(parts) < 3:
                    continue  # skip empty or malformed rows
              
                label = int(parts['label'])
                path1 = f"{base_dir}/{parts['utt1'].split('/')[-1]}"
                path2 = f"{base_dir}/{parts['utt2'].split('/')[-1]}"
                self.samples.append((label, path1, path2))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, path1, path2 = self.samples[idx]

        # Load both audios
        wav1, sr1 = torchaudio.load(path1)
        wav2, sr2 = torchaudio.load(path2)

        # Convert to mono
        wav1 = wav1.mean(dim=0, keepdim=True) if wav1.size(0) > 1 else wav1
        wav2 = wav2.mean(dim=0, keepdim=True) if wav2.size(0) > 1 else wav2

        # Resample if needed
        if sr1 != self.target_sr:
            wav1 = torchaudio.transforms.Resample(sr1, self.target_sr)(wav1)
        if sr2 != self.target_sr:
            wav2 = torchaudio.transforms.Resample(sr2, self.target_sr)(wav2)

        return wav1, wav2, torch.tensor(label, dtype=torch.float32), path1, path2


def collate_fn(batch, max_len_sec=3.0, sr=16000):
    """
    Pads or truncates all audio pairs to max_len_sec seconds.
    Returns tensors ready for batching.
    """
    max_len = int(max_len_sec * sr)
    wav1_list, wav2_list, paths1, paths2, labels = [], [], [], [], []

    for wav1, wav2, label, p1, p2 in batch:
        # breakpoint()
        # Pad/truncate wav1
        if wav1.size(-1) < max_len:
            pad = max_len - wav1.size(-1)
            wav1 = torch.nn.functional.pad(wav1, (0, pad))
        else:
            wav1 = wav1[:, :max_len]

        # Pad/truncate wav2
        if wav2.size(-1) < max_len:
            pad = max_len - wav2.size(-1)
            wav2 = torch.nn.functional.pad(wav2, (0, pad))
        else:
            wav2 = wav2[:, :max_len]
        if wav1.ndim == 2:
            wav1 = wav1.squeeze(0)
        if wav2.ndim == 2:
            wav2 = wav2.squeeze(0)
        wav1_list.append(wav1)
        wav2_list.append(wav2)
        labels.append(label)

    wav1_batch = torch.stack(wav1_list)
    wav2_batch = torch.stack(wav2_list)
    labels = torch.stack(labels)
    paths1.append(p1)
    paths2.append(p2)

    return {"wav1": wav1_batch, "wav2": wav2_batch, "label": labels, "p1": paths1, "p2": paths2}


def load_model():

    model = RecursiveAttnPooling(encoder=None, config=config).to(device)
    
    ckpt = torch.load("/home/sidharth./codebase/speaker_embedding_codebase/ckpts/paper_oracle_speakers/best-checkpoint-epoch=196-val/loss=8.83.ckpt", weights_only=True, map_location='cuda')
    new_state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("model."):
            new_k = k.replace("model.", "", 1)  # remove only the first 'model.'
        else:
            new_k = k
        new_state_dict[new_k] = v

    ckpt["state_dict"] = new_state_dict
    #load the model checkpoint

    model.load_state_dict(ckpt['state_dict'], strict=False)
    print("Model loaded from checkpoint.")
    model.eval()
    return model




def calculate_eer(scores, gt, num_thresholds=1000):
    """
    Vectorized EER computation.

    Args:
        scores (torch.Tensor): shape (N, M)
        gt (torch.Tensor): shape (N,)
        num_thresholds (int): number of thresholds for interpolation

    Returns:
        eer (float): Equal Error Rate
        threshold (float): corresponding threshold
    """
    # 1️⃣ Collapse multi-candidate scores -> max per trial
    max_scores = scores

    # 2️⃣ Separate genuine and impostor scores
    genuine_scores = max_scores[gt == 1]
    imposter_scores = max_scores[gt == 0]

    # 3️⃣ Compute thresholds
    thresholds = torch.linspace(max_scores.min(), max_scores.max(), num_thresholds).to('cuda')

    # 4️⃣ Vectorized comparison: shape (num_thresholds, )
    # Broadcasting: each threshold is compared against all scores simultaneously
    far = (imposter_scores[:, None] > thresholds[None, :]).float().mean(dim=0)
    frr = (genuine_scores[:, None] < thresholds[None, :]).float().mean(dim=0)

    # 5️⃣ Find index where |FAR - FRR| is minimum
    idx = torch.argmin(torch.abs(far - frr))
    eer = ((far[idx] + frr[idx]) / 2).item()
    threshold = thresholds[idx].item()

    return eer, threshold

def calc_cosine_similarities(embs1, embs2):

    '''
    embs1: [B,D]
    embs2: [B,D]
    '''
    embs1_norm = F.normalize(embs1, dim=-1)
    embs2_norm = F.normalize(embs2, dim=-1)

    sim = torch.bmm(embs1_norm, embs2_norm.transpose(1, 2))

    return sim




if __name__ == '__main__':

    txt_path = "/mnt/disks/data/datasets/Datasets/Libri2Mix/sp_ver_clean.csv"

    # labels, wavs1, wavs2 = get_audio_and_labels(txt_file=txt_path)
    dataset = SpeakerVerificationDataset(trials_txt=txt_path, base_dir="/mnt/disks/data/datasets/Datasets/Libri2Mix/clean_2sp")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False,
                        collate_fn=lambda x: collate_fn(x, max_len_sec=3.0, sr=16000))
    model = load_model()

    all_scores, all_labels, all_rows = [], [], []
    all_embs, all_roles = [], []

    EERs, all_scores, all_labels = [], [], []
    sp2 = 0
    sp6 = 0
    sp_other = 0
    for batch in tqdm.tqdm(dataloader, desc="Iterating batches"):
        wav1 = batch["wav1"].to(device)
        wav2 = batch["wav2"].to(device)
        labels = batch["label"].to(device)
        with torch.no_grad():

            emb1 = model(wav1, 2) #get only emb and not p # [B, n_sp, emb_dim]
            emb2 = model(wav2, 2) #[B, n_sp, emb_dim]
            if emb1.size(1) == 2 and emb2.size(1) ==2:
                sp2 += 1
            elif emb1.size(1) == 6 and emb2.size(1) ==6:
                sp6 += 1
            else:
                sp_other += 1
    
            preds = calc_cosine_similarities(emb1, emb2)

        all_scores.append(preds.cpu())
        all_labels.append(labels.cpu())
    print(f"sp2: {sp2}, sp6: {sp6}, sp_other: {sp_other}")
    breakpoint()

    all_scores_compressed = torch.cat([torch.amax(scores, dim=(1, 2)) for scores in all_scores], dim=0)
    all_scores = all_scores_compressed
    all_labels = torch.cat(all_labels)
    eer, th = calculate_eer(all_scores_compressed.to(device), all_labels.to(device))
    print(f"\nFinal EER: {eer*100:.2f}%  (threshold = {th:.4f})")
   
    # embs = torch.cat(all_embs).numpy()
    # np.savez("/Users/sidharth./Downloads/codebase/speaker_verification/eval_meta/vox1_test_embs.npz", emb=embs, role=np.array(all_roles))
    # print("Saved embeddings → vox1_test_embs.npz")
    

    

    


'''
import os
import json
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import tqdm
from torch.utils.data import Dataset, DataLoader

from recursive_attn_pooling import RecursiveAttnPooling
from configs import paper_config as config

# -----------------------------
# Global setup
# -----------------------------
torch.backends.cudnn.benchmark = True  # enable autotuner
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Utilities
# -----------------------------
def db_to_ratio(db: float) -> float:
    """Convert dB to linear amplitude ratio."""
    return 10.0 ** (db / 20.0)

@torch.no_grad()
def set_signal_energy(x: torch.Tensor, target_energy: float) -> torch.Tensor:
    """Scale tensor so that its RMS energy equals target_energy."""
    # x: [C, T]
    eps = 1e-8
    rms = torch.sqrt(torch.mean(x ** 2) + eps)
    return x * (target_energy / (rms + eps))

# -----------------------------
# Dataset
# -----------------------------
class SpeakerVerificationDataset(Dataset):
    """
    Expects a JSON file containing entries:
        [label(int 0/1), path1(str), [path2a(str), path2b(str)]]
    base_dir is prepended to each relative path.
    """

    def __init__(self, trials_txt: str, base_dir: str, target_sr: int = 16000):
        super().__init__()
        self.base_dir = base_dir.rstrip("/")

        # We allow variable original SRs, so cache per-orig_SR resamplers.
        self.target_sr = target_sr
        self._resampler_cache = {}  # {orig_sr: torchaudio.transforms.Resample}

        # Read JSON list
        with open(trials_txt, "r") as f:
            reader = json.load(f)

        self.samples = []
        for parts in reader:
            label = int(parts[0])
            path1 = f"{self.base_dir}/{parts[1]}"
            path2 = [f"{self.base_dir}/{p}" for p in parts[2]]
            self.samples.append((label, path1, path2))

    def __len__(self):
        return len(self.samples)

    def _get_resampler(self, orig_sr: int):
        if orig_sr == self.target_sr:
            return None
        if orig_sr not in self._resampler_cache:
            self._resampler_cache[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=self.target_sr
            )
        return self._resampler_cache[orig_sr]

    def _to_mono(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [C, T] -> [1, T]
        return wav.mean(dim=0, keepdim=True) if wav.size(0) > 1 else wav

    def _load_mono_resampled(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)  # [C,T], sr
        wav = self._to_mono(wav)
        resampler = self._get_resampler(sr)
        if resampler is not None:
            wav = resampler(wav)
        return wav  # [1, T] @ target_sr

    def __getitem__(self, idx):
        label, path1, path2 = self.samples[idx]

        # Load reference and two candidates (to be mixed at random SIR)
        wav1 = self._load_mono_resampled(path1)                  # [1, T1]
        wav2_1 = self._load_mono_resampled(path2[0])             # [1, T2a]
        wav2_2 = self._load_mono_resampled(path2[1])             # [1, T2b]

        # Normalize energies and mix with random SIR in [-5, 5] dB
        sir_db = float(np.random.uniform(-5.0, 5.0))
        sir_ratio = db_to_ratio(sir_db)

        wav2_1 = set_signal_energy(wav2_1, 1.0)
        wav2_2 = set_signal_energy(wav2_2, 1.0 / (sir_ratio + 1e-8))

        min_len = min(wav2_1.size(1), wav2_2.size(1))
        wav2 = wav2_1[:, :min_len] + wav2_2[:, :min_len]         # [1, Tmin]

        return wav1.squeeze(0), wav2.squeeze(0), torch.tensor(label, dtype=torch.float32), path1, path2


# -----------------------------
# Collate
# -----------------------------
def collate_fn(batch, max_len_sec: float = 3.0, sr: int = 16000):
    """
    Pads or truncates all audio pairs to max_len_sec seconds.
    Returns dict of batched tensors.
    """
    max_len = int(max_len_sec * sr)
    wav1_list, wav2_list, labels, paths1, paths2 = [], [], [], [], []

    for wav1, wav2, label, p1, p2 in batch:
        # Ensure shape [T] -> pad/truncate to max_len
        T1, T2 = wav1.size(-1), wav2.size(-1)

        if T1 < max_len:
            wav1 = F.pad(wav1, (0, max_len - T1))
        else:
            wav1 = wav1[:max_len]

        if T2 < max_len:
            wav2 = F.pad(wav2, (0, max_len - T2))
        else:
            wav2 = wav2[:max_len]

        wav1_list.append(wav1)
        wav2_list.append(wav2)
        labels.append(label)
        paths1.append(p1)
        paths2.append(p2)

    wav1_batch = torch.stack(wav1_list)   # [B, T]
    wav2_batch = torch.stack(wav2_list)   # [B, T]
    labels = torch.stack(labels)          # [B]

    return {"wav1": wav1_batch, "wav2": wav2_batch, "label": labels, "p1": paths1, "p2": paths2}


# -----------------------------
# Model loading
# -----------------------------
def load_model(ckpt_path: str) -> nn.Module:
    model = RecursiveAttnPooling(encoder=None, config=config).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    new_state_dict = {}
    for k, v in ckpt["state_dict"].items():
        new_k = k.replace("model.", "", 1) if k.startswith("model.") else k
        new_state_dict[new_k] = v
    model.load_state_dict(new_state_dict, strict=False)

    print(f"Model loaded from checkpoint: {ckpt_path}")
    model.eval()
    return model


# -----------------------------
# Scoring / Metrics
# -----------------------------
@torch.no_grad()
def calc_cosine_similarities(embs1: torch.Tensor, embs2: torch.Tensor) -> torch.Tensor:
    """
    Pairwise cosine similarities per sample.
    embs1: [B, N1, D], embs2: [B, N2, D]  ->  sim: [B, N1, N2]
    """
    e1 = F.normalize(embs1, dim=-1)
    e2 = F.normalize(embs2, dim=-1)
    return torch.bmm(e1, e2.transpose(1, 2))


@torch.no_grad()
def calculate_eer(max_scores: torch.Tensor, gt: torch.Tensor, num_thresholds: int = 200):
    """
    EER using vectorized sweep.
    max_scores: [N] (one score per trial)
    gt: [N] (0/1)
    """
    assert max_scores.ndim == 1 and gt.ndim == 1
    thresholds = torch.linspace(max_scores.min(), max_scores.max(), num_thresholds, device=max_scores.device)

    genuine = max_scores[gt == 1]  # [Ng]
    impost  = max_scores[gt == 0]  # [Ni]

    # FAR(th) = P(impost > th); FRR(th) = P(genuine < th)
    far = (impost[:, None] > thresholds[None, :]).float().mean(dim=0)  # [T]
    frr = (genuine[:, None] < thresholds[None, :]).float().mean(dim=0) # [T]

    idx = torch.argmin(torch.abs(far - frr))
    eer = ((far[idx] + frr[idx]) / 2).item()
    th  = thresholds[idx].item()
    return eer, th


# -----------------------------
# Fast evaluation
# -----------------------------
@torch.no_grad()
def fast_eval(model: nn.Module, dataloader: DataLoader, use_amp: bool = True):
    from torch.cuda.amp import autocast

    all_scores, all_labels = [], []
    sp2 = sp6 = sp_other = 0

    if use_amp and device.type == "cuda":
        try:
            # Newer PyTorch (>=1.10)
            from torch.cuda.amp import autocast
            amp_ctx = autocast(device_type="cuda")
        except TypeError:
            # Older PyTorch (no device_type argument)
            from torch.cuda.amp import autocast
            amp_ctx = autocast()
    else:
        from contextlib import nullcontext
        amp_ctx = nullcontext()

    with amp_ctx:
        for batch in tqdm.tqdm(dataloader, desc="Fast Eval", dynamic_ncols=True):
            wav1 = batch["wav1"].to(device, non_blocking=True)  # [B, T]
            wav2 = batch["wav2"].to(device, non_blocking=True)  # [B, T]
            labels = batch["label"].to(device, non_blocking=True)

            # Model is expected to return embeddings per sample: [B, n_sp, D]
            breakpoint()
            emb1 = model(wav1, 1)
            emb2 = model(wav2, 2)

            n1, n2 = emb1.size(1), emb2.size(1)
            if n1 == 2 and n2 == 2:
                sp2 += 1
            elif n1 == 6 and n2 == 6:
                sp6 += 1
            else:
                sp_other += 1

            sims = calc_cosine_similarities(emb1, emb2)   # [B, n1, n2]
            # Collapse to one score per trial (max over all pairs):
            scores = torch.amax(sims, dim=(1, 2))          # [B]

            all_scores.append(scores.detach().cpu())
            all_labels.append(labels.detach().cpu())

    print(f"Count by n_sp: sp2={sp2}, sp6={sp6}, other={sp_other}")

    all_scores = torch.cat(all_scores, dim=0)  # [N]
    all_labels = torch.cat(all_labels, dim=0)  # [N]
    eer, th = calculate_eer(all_scores.to(device), all_labels.to(device), num_thresholds=200)
    print(f"\nFinal EER: {eer * 100:.2f}%  (threshold = {th:.4f})")
    return eer, th


# -----------------------------
# Main
# -----------------------------
def main():
    # ---- Paths (edit as needed)
    txt_path = "/mnt/disks/data/datasets/Datasets/Vox1_sp_ver/paper_svm.json"
    base_dir = "/mnt/disks/data/datasets/Datasets/voxceleb/vox1/eval/wav/"
    ckpt_path = "/home/sidharth./codebase/speaker_embedding_codebase/ckpts/paper_oracle_speakers/best-checkpoint-epoch=248-val/loss=8.37.ckpt"

    # ---- Dataset / DataLoader
    dataset = SpeakerVerificationDataset(
        trials_txt=txt_path,
        base_dir=base_dir,
        target_sr=16000,
    )

    # Tune num_workers to your storage throughput; start with #CPU cores / 2
    # num_workers = min(8, os.cpu_count() or 4)
    num_workers = 0
    loader = DataLoader(
        dataset,
        batch_size=64,            # ↑ larger batch helps GPU; adjust to VRAM
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, max_len_sec=3.0, sr=16000),
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )

    # ---- Model
    model = load_model(ckpt_path)

    # ---- Eval
    fast_eval(model, loader, use_amp=True)


if __name__ == "__main__":
    main()

'''