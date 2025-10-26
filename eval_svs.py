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
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue  # skip invalid lines
                label = int(parts[0])
                path1 = f"{base_dir}/{parts[1]}"
                path2 = f"{base_dir}/{parts[2]}"
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
    
    ckpt = torch.load("/home/sidharth./codebase/speaker_embedding_codebase/ckpts/paper_oracle_speakers_1nsp_nandebug/best-checkpoint-epoch=986-val/loss=2.81.ckpt", weights_only=True, map_location='cuda')
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
    Vectorized Equal Error Rate computation.

    Args:
        scores (torch.Tensor): (N,) or (N, M)
        gt (torch.Tensor): (N,)
    """
    # Ensure both tensors are 1D and on same device
    if scores.ndim > 1:
        scores = scores.max(dim=1)[0]  # collapse multi-candidate scores

    device = scores.device
    genuine = scores[gt == 1]
    impostor = scores[gt == 0]

    thresholds = torch.linspace(scores.min(), scores.max(), num_thresholds, device=device)
    far = (impostor[:, None] > thresholds[None, :]).float().mean(dim=0)
    frr = (genuine[:, None] < thresholds[None, :]).float().mean(dim=0)

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

    sim = F.cosine_similarity(embs1, embs2, dim=-1)

    return sim




if __name__ == '__main__':

    txt_path = "/mnt/disks/data/datasets/Datasets/Vox1_sp_ver/svs.txt"

    # labels, wavs1, wavs2 = get_audio_and_labels(txt_file=txt_path)
    dataset = SpeakerVerificationDataset(trials_txt=txt_path, base_dir="/mnt/disks/data/datasets/Datasets/voxceleb/vox1/eval/wav/")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False,
                        collate_fn=lambda x: collate_fn(x, max_len_sec=3.0, sr=16000), num_workers=20)
    model = load_model()

    all_scores, all_labels, all_rows = [], [], []
    all_embs, all_roles = [], []

    EERs, all_scores, all_labels = [], [], []
    for batch in tqdm.tqdm(dataloader, desc="Iterating batches"):
        wav1 = batch["wav1"].to(device)
        wav2 = batch["wav2"].to(device)
        labels = batch["label"].to(device)
        with torch.no_grad():
            # breakpoint()
            emb1 = model(wav1, 1) #get only emb and not p # [B, n_sp, emb_dim]
            emb2 = model(wav2, 1) #[B, n_sp, emb_dim]
    
            preds = calc_cosine_similarities(emb1, emb2)

        all_scores.append(preds.cpu())
        all_labels.append(labels.cpu())
    breakpoint()

    all_scores_compressed = torch.cat([torch.amax(scores, dim=(1, 2)) for scores in all_scores], dim=0)
    all_scores = all_scores_compressed
    all_labels = torch.cat(all_labels)
    eer, th = calculate_eer(all_scores_compressed.to(device), all_labels.to(device))
    print(f"\nFinal EER: {eer*100:.2f}%  (threshold = {th:.4f})")
   
    # embs = torch.cat(all_embs).numpy()
    # np.savez("/Users/sidharth./Downloads/codebase/speaker_verification/eval_meta/vox1_test_embs.npz", emb=embs, role=np.array(all_roles))
    # print("Saved embeddings → vox1_test_embs.npz")
    

    

    
