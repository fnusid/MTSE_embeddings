import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
import tqdm
from pathlib import Path
import pandas as pd
import csv
import json
from recursive_attn_pooling import RecursiveAttnPooling
from configs import paper_config as config
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
def db_to_ratio(db):
    """Convert dB value to linear amplitude ratio."""
    return 10 ** (db / 20)

def set_signal_energy(x, target_energy):
    """Scale tensor so that its RMS energy equals target_energy."""
    rms = torch.sqrt(torch.mean(x ** 2) + 1e-8)
    return x * (target_energy / (rms + 1e-8))

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
        with open(trials_txt, "r") as f:
            reader = json.load(f)
            for parts in reader:
              
                label = int(parts[0])
                path1 = f"{base_dir}/{parts[1]}"
                path2 = []
                for item in parts[2]:
                    path2.append(f"{base_dir}/{item}")
                self.samples.append((label, path1, path2))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, path1, path2 = self.samples[idx]

        # Load both audios
        wav1, sr1 = torchaudio.load(path1)
        wav2_1, sr2 = torchaudio.load(path2[0])
        wav2_2, sr2 = torchaudio.load(path2[1])

        sir_db = np.random.uniform(-5,5)
        sir_ratio = db_to_ratio(sir_db)
        wav2_1 = set_signal_energy(wav2_1, 1.0)
        wav2_2 = set_signal_energy(wav2_2, 1.0 / (sir_ratio + 1e-8))  # adjust relative loudness

        min_len = min(wav2_1.size(1), wav2_2.size(1))
        wav2_1 = wav2_1[:, :min_len]
        wav2_2 = wav2_2[:, :min_len]
        wav2 = wav2_1 + wav2_2

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
    
    ckpt = torch.load("/home/sidharth./codebase/speaker_embedding_codebase/ckpts/paper_oracle_speakers/best-checkpoint-epoch=248-val/loss=8.37.ckpt", weights_only=True, map_location='cuda')
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

    txt_path = "/mnt/disks/data/datasets/Datasets/Vox1_sp_ver/paper_svm.json"

    # labels, wavs1, wavs2 = get_audio_and_labels(txt_file=txt_path)
    dataset = SpeakerVerificationDataset(trials_txt=txt_path, base_dir="/mnt/disks/data/datasets/Datasets/voxceleb/vox1/eval/wav/")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False,
                        collate_fn=lambda x: collate_fn(x, max_len_sec=3.0, sr=16000), num_workers=20)
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
            # breakpoint()
            emb1 = model(wav1, 1) #get only emb and not p # [B, n_sp, emb_dim]
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
    

    

    
