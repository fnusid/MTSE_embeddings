import os
import random
from typing import Optional, Any, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
import pytorch_lightning as pl

import ast
import config

'''
Speaker files
    - IDXXXX
        - wav1.wav
        - wav2.wav
Training : Voxceleb1 + Voxceleb2 (devs)
Validation : Voxceleb2 (test)
noise_probability : 0.5
rir_probability : 0.5

'''


# ---------------------------
# DataLoader helpers
# ---------------------------

def _worker_init_fn(worker_id: int):
    # keep worker processes from oversubscribing CPU threads
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    # deterministic-ish seeds per worker
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def collate_pair(batch):
    """
    Pads variable-length waveforms in the batch to the maximum length.
    Returns:
        noisy:  (B, 2, T_max)
        labels: (B, num_classes)
    """
    # Extract tensors
    # print("⚡ CUSTOM COLLATE CALLED with batch size:", len(batch))
    wavs = [b["noisy"] for b in batch]
    labels = [torch.as_tensor(b["labels"], dtype=torch.float32) for b in batch]

    # Find max length in this batch
    max_len = max(wav.shape[-1] for wav in wavs)

    # Pad all to same length
    padded_wavs = []
    for w in wavs:
        pad_len = max_len - w.shape[-1]
        if pad_len > 0:
            # Pad at the end (right side)
            w = F.pad(w, (0, pad_len))
        padded_wavs.append(w)

    noisy = torch.stack(padded_wavs, dim=0)   # [B, 2, T_max]
    labels = torch.stack(labels, dim=0)       # [B, num_classes]

    return noisy, labels



# ---------------------------
# Dataset
# ---------------------------

class SpeakerIdentification(Dataset):
    """
    Expects lists of filepaths for speech/noise/rir. Produces dicts:
        {"noisy": (B,T), "labels": (B,)}
    
    speeches: {"ID":[list of wav files]}
    """

    def __init__(
        self,
        speeches: Dict[str, List[str]],
        noise: List[str],
        rir: List[str],
        N_max_speakers: int = 4,
        overlap_ratio: float = 0.2,
        desired_duration: float = 8.0,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.speeches = speeches
        self.speech_ids = list(speeches.keys())
        self.num_classes = len(self.speech_ids)
        self.noise = noise
        self.rir = rir
        self.overlap_ratio = overlap_ratio
        self.N_max_speakers = N_max_speakers
        self.desired_duration = desired_duration
        self.total_desired_duration = desired_duration * N_max_speakers #max possible length 

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # config
        self.sr = int(kwargs.get("sr", 16000))
        self.segment_length = float(kwargs.get("segment_length", 8.0))
        self.add_noise_prob = float(kwargs.get("add_noise_prob", 0.5))
        self.overlap_prob = float(kwargs.get("overlap_prob", 0.5))
        self.rir_probability = float(kwargs.get("rir_probability", 0.5))
        self.early_reverb_sec = float(kwargs.get("early_reverb_sec", 0.05))
        self.global_snr_range = tuple(kwargs.get("global_snr", (-5, 40)))  # dB
        # optional cap on RIR length for speed (seconds). None = no cap
        self.max_rir_seconds = kwargs.get("max_rir_seconds", 0.5)
        self.peak_normalize = kwargs.get("peak_normalization", True)

        # per-worker lightweight caches
        self._rir_cache: Dict[str, torch.Tensor] = {}
        self._resampler_cache: Dict[tuple, torchaudio.transforms.Resample] = {}

    # -------- I/O utils --------

    def _get_resampler(self, src_sr: int) -> Optional[torchaudio.transforms.Resample]:
        if src_sr == self.sr:
            return None
        key = (src_sr, self.sr)
        rs = self._resampler_cache.get(key)
        if rs is None:
            # torchscript-friendly, fast resampler
            rs = torchaudio.transforms.Resample(src_sr, self.sr)
            self._resampler_cache[key] = rs
        return rs

    def _load_wav_mono(self, paths: List[str], length=None, noise=None, n_sp=None) -> torch.Tensor:
        wavs = []
        for idx, path in enumerate(paths):
            if n_sp is not None and idx >= n_sp:
                break
            # breakpoint()
            wav, sr = torchaudio.load(path)  # (C, T)
            if wav.dim() == 2 and wav.size(0) > 1:
                wav = wav[0:1]  # take first channel
            wav = wav.squeeze(0).to(torch.float32)
            rs = self._get_resampler(sr)
            if rs is not None:
                wav = rs(wav.unsqueeze(0)).squeeze(0)
            if noise is None and length is None:
                if len(wav) < int(self.desired_duration * self.sr): #for speech
                    rem = int(self.desired_duration * self.sr) - len(wav)
                    wav = F.pad(wav, (rem//2, rem - rem//2))
                elif len(wav) > int(self.desired_duration * self.sr):
                    wav = wav[:int(self.desired_duration * self.sr)]
                
            if length is not None and noise is not None:
                # breakpoint()
                len_wav = sum(len(item) for item in wavs) if len(wavs)>0 else 0
                if len_wav > length:
                    wavs = torch.cat(wavs, dim=0)
                    return wavs[:length]
            
            wavs.append(wav)
        
        if n_sp is not None and n_sp < self.N_max_speakers:
            #padd with zeros in between each wav in a randomized manner
            wavs_padded = []
            chunk_indices = torch.arange(int(self.total_desired_duration/self.desired_duration))

            chosen_indices = random.sample(list(chunk_indices), n_sp)
            chosen_indices = [it.item() for it in chosen_indices]
            chosen_indices.sort()
            rest_indices = list(set(chunk_indices.tolist()) - set(chosen_indices))
            for i, idx in enumerate(chunk_indices):
                if idx in chosen_indices:
                    wavs_padded.append(wavs[chosen_indices.index(idx)])
                else:
                    wavs_padded.append(torch.zeros(int(self.desired_duration * self.sr), device=wavs[0].device))
            wavs = wavs_padded

    
        wavs = torch.stack(wavs, dim=0)

        return wavs

    def _load_rir(self, path: str, n_sp: int) -> torch.Tensor:
        if path in self._rir_cache:
            return self._rir_cache[path]
        rir, sr = torchaudio.load(path)  # (C, T)
        rir = rir.reshape(2, -1, rir.size(-1))  # (mics=2, srcs, T)
        # breakpoint()
        #randomly sample random n_sp samples from the rir if more than n_sp
        if rir.dim() == 3 and rir.size(1) > n_sp:
            indices = random.sample(range(rir.size(1)), 4)
            rir = rir[:, indices, :]
    
        rir = rir.to(torch.float32)
        rs = self._get_resampler(sr)
        if rs is not None:
            rir = rs(rir.unsqueeze(0)).squeeze(0)
        # optional cap for speed (e.g., 0.5 s @ 16k -> 8000 taps)
        if self.max_rir_seconds and self.max_rir_seconds > 0:
            max_taps = int(self.max_rir_seconds * self.sr)
            rir = rir[:, :, :max_taps]
        self._rir_cache[path] = rir
        return rir

    # -------- DSP helpers --------

    @staticmethod
    def _fft_convolve_same_len(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Fast convolution via FFT, returning first len(x) samples.
        x, h: (T,)
        """
        T = x.numel()
        n = T + h.numel() - 1
        nfft = 1 << (n - 1).bit_length()  # next power of two

        X = torch.fft.rfft(x, n=nfft)
        H = torch.fft.rfft(h, n=nfft)
        y = torch.fft.irfft(X * H, n=nfft)[:T]
        return y


    @staticmethod
    def _add_noise_at_snr(speech: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
        sp = speech.pow(2).mean().clamp_min(1e-10)
        npow = noise.pow(2).mean().clamp_min(1e-10)
        snr_lin = 10.0 ** (snr_db / 10.0)
        req_np = sp / snr_lin
        scale = (req_np / npow).sqrt()
        return speech + noise * scale

    @staticmethod
    def _mix_overlap_add(signals, overlap_ratio):
        '''
        signals: list of 1D tensors
        overlap_ratio: float between 0 and 1
        '''

        if len(signals)==0:
            return torch.tensor([])
        elif len(signals)==1:
            return signals[0]
        else:
            overlap_area = int(signals[0].shape[-1] * overlap_ratio)
            overlapped_signals = []
            for i in range(len(signals)):
                if i==0:
                    overlapped_signals.append(signals[i])
                else:
                    previous = overlapped_signals[i-1]
                    current = signals[i]
                    if overlap_area > 0:
                        new_signal = torch.zeros((2,previous.shape[-1] + current.shape[-1] - overlap_area,), device=previous.device)
                        new_signal[:, :previous.shape[-1]-overlap_area] = previous[:, :previous.shape[-1]-overlap_area]
                        new_signal[:, previous.shape[-1]-overlap_area:previous.shape[-1]] = previous[:, previous.shape[-1]-overlap_area:] + current[:, :overlap_area]
                        new_signal[:, previous.shape[-1]:] = current[:, overlap_area:]
                        overlapped_signals.append(new_signal)
                    else:
                        new_signal = torch.zeros((previous.shape[-1] + current.shape[-1],), device=previous.device)
                        new_signal[:, :previous.shape[-1]] = previous
                        new_signal[:, previous.shape[-1]:] = current
                        overlapped_signals.append(new_signal)
            return overlapped_signals[-1]

    # -------- Dataset API --------

    def __len__(self):
        return len(self.speech_ids)

    def __getitem__(self, idx: int):
        '''
        Modify here to return different speaker labels
        numpy
        ---
        IDXX:[bunch of wav files]
        IDXY:[]
        .
        .
        

        '''
        n_sp = random.randint(1, self.N_max_speakers)
        # n_sp = 1
        chosen_ids = random.choices(self.speech_ids, k=n_sp)

        chosen_wavs = [] #chosen wav dim: [n_sp, wav_path]
        for sp_id in chosen_ids:
            wav_file = random.choice(self.speeches[sp_id])
            chosen_wavs.append(wav_file)
        # breakpoint()
        sp_path = chosen_wavs
        sp_labels = [self.speech_ids.index(sp_id) for sp_id in chosen_ids]
        vec = np.zeros(self.num_classes, dtype=np.float32)
        vec[sp_labels] = 1.0
        nz_path = self.noise #[list of noise files]
        rr_path = self.rir[idx % len(self.rir)]
        speech = self._load_wav_mono(sp_path, n_sp=n_sp) #[n_sp, wav]
        if speech.shape[0] == 1:
            speech = speech.unsqueeze(0) #[1, wav]
        #add rir with a probability
        if np.random.rand() > self.rir_probability:
        # if 0 < self.rir_probability: #always add rir
            rir = self._load_rir(rr_path, n_sp = n_sp) #[n_mics, n_sources, T]
            convolved_speeches = []
            # breakpoint()
            for i in range(len(speech)):
                speech_mic1 = self._fft_convolve_same_len(speech[i], rir[0][i])
                speech_mic2 = self._fft_convolve_same_len(speech[i], rir[1][i])

                #cap the lrngth to min of both mics
                min_len = min(speech_mic1.numel(), speech_mic2.numel())
                speech_mic1 = speech_mic1[:min_len]
                speech_mic2 = speech_mic2[:min_len]
                #use both binaural mics
                convolved_speeches.append(torch.cat([speech_mic1.unsqueeze(0), speech_mic2.unsqueeze(0)], dim=0)) #dim: [2, T]
            speech = torch.stack(convolved_speeches, dim=0) #dim: [n_sp, 2, T]
        else:
            #duplicate the single channel to make it binaural
            speech = torch.stack([speech, speech], dim=1) #dim: [n_sp, 2, T]
                
        #combine speeches with the given overlap ratio
        if np.random.rand() > self.overlap_prob:
        # if 0 < self.overlap_prob: #always overlap
            overlap_ratio = np.random.uniform(0, self.overlap_ratio)

            speech = self._mix_overlap_add([speech[i] for i in range(len(speech))], self.overlap_ratio)

        else:
            speech = speech.transpose(0, 1).reshape(2, -1) 
        # breakpoint()

        noise = self._load_wav_mono(nz_path, length=speech.shape[-1], noise=True) #gets noise as long as speech

        if random.random() < self.add_noise_prob:
            snr = random.uniform(self.global_snr_range[0], self.global_snr_range[1])
            noisy = self._add_noise_at_snr(speech, noise, snr)
        else:
            noisy = speech
        if self.peak_normalize:
            noisy = noisy / noisy.max(axis = -1)[0].unsqueeze(-1) #get only the max values and not indices
        
        '''
        noisy will be max of length 8*4 = 32 seconds
        vec will be [0,0,0,,....,1,..,1...]
        vec will be a multi-hot vector of length num_classes
        '''
        # breakpoint()
        # print(">>> returning noisy", noisy.shape, "labels", vec.shape)
        return {"noisy": noisy, "labels": torch.from_numpy(vec)} #noisy: [2, T], vec: [num_classes]


# ---------------------------
# DataModule
# ---------------------------

class SpeakerIdentificationDM(pl.LightningDataModule):
    def __init__(
        self,
        speeches_list: str,
        noise_list: str,
        rir_list: str,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.speeches_list_path = speeches_list
        self.noise_list_path = noise_list
        self.rir_list_path = rir_list

        self.batch_size = batch_size
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.dataset_kwargs = kwargs

        # read filelists
        with open(f"{self.speeches_list_path}", "r") as f:
            content = f.read()

        self.speech_files = ast.literal_eval(content)
        with open(self.noise_list_path) as f:
            self.noise_files = [ln.strip() for ln in f if ln.strip()]
        with open(self.rir_list_path) as f:
            self.rir_files = [ln.strip() for ln in f if ln.strip()]

        # shuffle once before split
        spk_ids = list(self.speech_files.keys())
        random.shuffle(spk_ids)

        n_train = int(0.8 * len(spk_ids))
        train_ids = spk_ids[:n_train]
        val_ids = spk_ids[n_train:]
        #print the number of class in train and test
        print(f"Number of classes in train is {len(train_ids)}")
        print(f"Number of classes in test is {len(test_ids)}")

        self.train_num_class = len(train_ids)
        self.test_num_class = len(test_ids)

        self.train_speech = {spk: self.speech_files[spk] for spk in train_ids}
        self.val_speech = {spk: self.speech_files[spk] for spk in val_ids}

        random.shuffle(self.noise_files)
        random.shuffle(self.rir_files)

        # split 80/20
        nn, nr = map(len, (self.noise_files, self.rir_files))
        n_cut, r_cut =  int(0.8 * nn), int(0.8 * nr)

        self.train_noise = self.noise_files[:n_cut]
        self.val_noise = self.noise_files[n_cut:]

        self.train_rir = self.rir_files[:r_cut]
        self.val_rir = self.rir_files[r_cut:]

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = SpeakerIdentification(
            self.train_speech, self.train_noise, self.train_rir, **self.dataset_kwargs
        )
        self.val_dataset = SpeakerIdentification(
            self.val_speech, self.val_noise, self.val_rir, **self.dataset_kwargs
        )

    def train_dataloader(self):
        use_workers = self.num_workers > 0
        # prefetch_factor must be omitted when num_workers == 0
        kwargs = dict(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=use_workers,
            drop_last=True,
            collate_fn=lambda b: collate_pair(b),
            timeout=0,
            worker_init_fn=_worker_init_fn,
        )
        if use_workers:
            kwargs["prefetch_factor"] = 2  # good general default
        return DataLoader(**kwargs)

    def val_dataloader(self):
        # keep workers at 0 during sanity check to avoid pickling issues in some envs
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.pin_memory,
            persistent_workers=False,
            drop_last=False,
            collate_fn=lambda b: collate_pair(b),
            timeout=0,
            worker_init_fn=_worker_init_fn,
        )


# ---------------------------
# Quick local smoke test
# ---------------------------
if __name__ == "__main__":
    # expects lists of file paths; change these if you want to run locally
    # breakpoint()
    import ast
    speeches_txt = "/nfs/turbo/coe-profdj/txts/voxceleb_test.txt"
    noise_txt = "/scratch/profdj_root/profdj0/sidcs/utils/freesound.txt"
    rir_txt = "/nfs/turbo/coe-profdj/txts/rirs_test.txt"
    # breakpoint()
    dataset = SpeakerIdentificationDM(
        speeches_list = speeches_txt,
        noise_list = noise_txt,
        rir_list = rir_txt,
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
    dataset.setup()
    dl = dataset.train_dataloader()
    # breakpoint()
    for batch in dl:
        x, y = batch
        print("noisy:", x.shape, "labels:", y.shape)
        break