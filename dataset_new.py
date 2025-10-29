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
from configs import paper_config as config
from torch.utils.data import Sampler
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
##SIR

def mix_at_sir(sp1: torch.Tensor, sp2: torch.Tensor, sir_db: float):
    """
    Mix sp2 into sp1 at a target SIR (in dB).

    sp1, sp2: [T] or [C, T] waveforms (same shape)
    sir_db: desired SIR in dB (sp1 vs sp2)

    Returns:
        mixture: sp1 + scaled_sp2
        scaled_sp2: scaled interference
    """
    eps = 1e-10

    # power of signals (mean of square)
    P1 = torch.mean(sp1**2).clamp_min(eps)
    P2 = torch.mean(sp2**2).clamp_min(eps)

    # linear SIR ratio
    sir_lin = 10 ** (sir_db / 10.0)

    # required power for sp2 after scaling
    P2_target = P1 / sir_lin

    # scaling factor (sqrt because power → amplitude)
    scale = torch.sqrt(P2_target / (P2 + eps))

    sp2_scaled = sp2 * scale

    mix = sp1 + sp2_scaled
    return mix

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

from torch.utils.data import RandomSampler, BatchSampler

class NSPBatchSampler(BatchSampler):
    """Yields batches of (idx, n_sp) tuples, so all samples in a batch share n_sp."""
    def __init__(self, dataset, batch_size, drop_last, n_sp_max, trainer=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.n_sp_max = n_sp_max
        self.trainer = trainer
        self.base = BatchSampler(RandomSampler(dataset), batch_size, drop_last)


    def _get_curr_nsp(self):
        if self.trainer is None:
            return random.randint(1, self.n_sp_max)
        else:
            epoch = self.trainer.current_epoch

            # if epoch < 5:
            #     return 1
            # elif epoch < 15:
            #     return 2
            # else:
            #     return random.randint(1, self.n_sp_max)
            # if epoch < 5:
            #     prob_2 = 0
            # elif epoch < 15:
            #     prob_2 = 0.5
            # else:
            #     prob_2 = 0.33
            
            # if random.random() < prob_2:
            #     return 2
            # else:
            #     return 1
            return random.randint(1, self.n_sp_max)

    def __iter__(self):
        for batch_indices in self.base:

            n_sp = self._get_curr_nsp()
            yield [(idx, n_sp) for idx in batch_indices]

    def __len__(self):
        return len(self.base)



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
        if w.ndim == 3: #You can remove this and the below line later on
            w = w.squeeze(1)
        padded_wavs.append(w)

    noisy = torch.stack(padded_wavs, dim=0)   # [B, 2, T_max]
    labels = torch.stack(labels, dim=0)       # [B, num_classes]

    return noisy, labels

# def db_to_ratio(db):
#     """Convert dB value to linear amplitude ratio."""
#     return 10 ** (db / 20)

# def set_signal_energy(x, target_energy):
#     """Scale tensor so that its RMS energy equals target_energy."""
#     rms = torch.sqrt(torch.mean(x ** 2) + 1e-8)
#     return x * (target_energy / (rms + 1e-8))


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
        self.current_n_sp = 1
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

    # def _get_resampler(self, src_sr: int) -> Optional[torchaudio.transforms.Resample]:
    #     if src_sr == self.sr:
    #         return None
    #     key = (src_sr, self.sr)
    #     rs = self._resampler_cache.get(key)
    #     if rs is None:
    #         # torchscript-friendly, fast resampler
    #         rs = torchaudio.transforms.Resample(src_sr, self.sr)
    #         self._resampler_cache[key] = rs
    #     return rs


    ## simple function to load wav


    def load_audio(self, path: List[str], length=None, n_sp=None):
        wavs = []
        for idx, p in enumerate(path):
            if n_sp is not None and idx >= n_sp:
                break
            
            wav, sr = torchaudio.load(p)  # (C, T)
            if wav.dim() == 2 and wav.size(0) > 1:
                wav = wav[0:1]  # take first channel
            
            wav = wav.squeeze(0).to(torch.float32)

            if sr != self.sr:
                wav = torchaudio.transforms.Resample(sr, self.sr)(wav.unsqueeze(0)).squeeze(0)
                sr = self.sr
            
            if wav.size(-1) != int(self.desired_duration * self.sr):
                if len(wav) < int(self.desired_duration * self.sr): #for speech
                    rem = int(self.desired_duration * self.sr) - len(wav)
                    wav = F.pad(wav, (rem//2, rem - rem//2))
                elif len(wav) > int(self.desired_duration * self.sr):
                    wav = wav[:int(self.desired_duration * self.sr)]
            
            wavs.append(wav)

        if len(wavs) > 1:
            sp1 = wavs[0]
            sp2 = wavs[1]
            mix = mix_at_sir(sp1, sp2, random.uniform(-5,5))
            return mix
        else:
            return torch.sum(torch.stack(wavs, dim=0), dim=0)
            



    def load_noise(self, paths: List[str], length: int) -> torch.Tensor:
        """
        Load and concatenate noise waveforms until reaching 'length' samples.

        Args:
            paths (List[str]): list of noise file paths
            length (int): total target length in samples (usually len(speech))

        Returns:
            torch.Tensor: mono noise waveform of shape [length]
        """
        wavs = []
        total_len = 0

        for path in paths:
            if total_len >= length:
                break

            try:
                wav, sr = torchaudio.load(path)  # (C, T)
            except Exception as e:
                print(f"Skipping invalid noise file: {path} ({e})")
                continue

            # Use first channel only
            if wav.dim() == 2 and wav.size(0) > 1:
                wav = wav[0:1]
            wav = wav.squeeze(0).to(torch.float32)

            # Resample if needed
            if sr != self.sr:
                wav = torchaudio.transforms.Resample(sr, self.sr)(wav.unsqueeze(0)).squeeze(0)
                sr = self.sr
            # Normalize to [-1, 1] (avoid NaNs or overly loud noise)
            wav = wav / (wav.abs().max() + 1e-8)

            # Append until reaching target length
            wavs.append(wav)
            total_len += len(wav)

        # Concatenate all chunks
        if len(wavs) == 0:
            # In case all noise files failed
            return torch.zeros(length, dtype=torch.float32)

        noise_cat = torch.cat(wavs, dim=0)

        # Trim or pad to match desired length
        if len(noise_cat) < length:
            pad = length - len(noise_cat)
            noise_cat = F.pad(noise_cat, (0, pad))
        elif len(noise_cat) > length:
            noise_cat = noise_cat[:length]
        assert torch.isnan(noise_cat).sum() == 0, "NaN in loaded noise"
        return noise_cat


    def _load_rir(self, path: str, n_sp: int) -> torch.Tensor:
        if path in self._rir_cache:
            return self._rir_cache[path]

        rir, sr = torchaudio.load(path)  # (C, T)
        assert torch.isnan(rir).sum() == 0, f"NaN in RIR file {path}"
        C = rir.size(0)
        if C % 2 != 0:
            raise ValueError(f"Unexpected RIR channel count ({C}) in {path}")
        rir = rir.reshape(2, C // 2, rir.size(-1))  # (mics=2, srcs, T)

        #only mono rir
        rir = rir[0]  # (srcs, T)

        # sample n_sp sources
        if rir.size(0) > n_sp:
            indices = random.sample(range(rir.size(0)), n_sp)
            rir = rir[indices, :]

        rir = rir.to(torch.float32)
        if sr != self.sr:
            rir = torchaudio.transforms.Resample(sr, self.sr)(rir)
            sr = self.sr

        if self.max_rir_seconds and self.max_rir_seconds > 0:
            max_taps = int(self.max_rir_seconds * self.sr)
            rir = rir[:, :max_taps]

        # normalize energy
        rir = rir / (rir.abs().max() + 1e-8)
        assert torch.isnan(rir).sum() == 0, f"NaN after processing RIR file {path}"

        self._rir_cache[path] = rir
        return rir #[n_sp, T]

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

        assert torch.isnan(y).sum() == 0, "NaN in convolution output"
        return y


    @staticmethod
    def _add_noise_at_snr(speech: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:

        sp = speech.pow(2).mean().clamp_min(1e-10)
        npow = noise.pow(2).mean().clamp_min(1e-10)
        if sp < 1e-8 or npow < 1e-8:
            return speech  # skip adding noise to avoid instability
        snr_lin = 10.0 ** (snr_db / 10.0)
        req_np = sp / (snr_lin + 1e-8)
        scale = (req_np / npow + 1e-8).sqrt().clamp(max=10.0)
        mixed = speech + noise * scale
        mixed = torch.clamp(mixed, min=-1.0, max=1.0)
        assert torch.isnan(mixed).sum() == 0, "NaN in noisy mixture"
        return mixed

    # @staticmethod
    # def _mix_overlap_add(signals, overlap_ratio):
    #     '''
    #     signals: list of 1D tensors
    #     overlap_ratio: float between 0 and 1
    #     '''

    #     if len(signals)==0:
    #         return torch.tensor([])
    #     elif len(signals)==1:
    #         return signals[0]
    #     else:
    #         overlap_area = int(signals[0].shape[-1] * overlap_ratio)
    #         overlapped_signals = []
    #         for i in range(len(signals)):
    #             if i==0:
    #                 overlapped_signals.append(signals[i])
    #             else:
    #                 previous = overlapped_signals[i-1]
    #                 current = signals[i]
    #                 if overlap_area > 0:
    #                     new_signal = torch.zeros((2,previous.shape[-1] + current.shape[-1] - overlap_area,), device=previous.device)
    #                     new_signal[:, :previous.shape[-1]-overlap_area] = previous[:, :previous.shape[-1]-overlap_area]
    #                     new_signal[:, previous.shape[-1]-overlap_area:previous.shape[-1]] = previous[:, previous.shape[-1]-overlap_area:] + current[ :overlap_area]
    #                     new_signal[:, previous.shape[-1]:] = current[:, overlap_area:]
    #                     overlapped_signals.append(new_signal)
    #                 else:
    #                     new_signal = torch.zeros((previous.shape[-1] + current.shape[-1],), device=previous.device)
    #                     new_signal[:, :previous.shape[-1]] = previous
    #                     new_signal[:, previous.shape[-1]:] = current
    #                     overlapped_signals.append(new_signal)
    #         return overlapped_signals[-1]

    # -------- Dataset API --------

    def __len__(self):
        # return len(self.speech_ids)
        # return sum(len(v) for v in self.speeches.values())
        return 10_000

    def __getitem__(self, item):
        '''
        Modify here to return different speaker labels
        numpy
        ---
        IDXX:[bunch of wav files]
        IDXY:[]
        .
        .
        

        '''
        if isinstance(item, tuple):
            idx, n_sp = item
            # print("idx, n_sp:", idx, n_sp)
        else:
            idx = item
        # idx, n_sp = batch_info["indices"], batch_info["n_sp"]
        # n_sp = random.randint(1, self.N_max_speakers)
        # n_sp = self.N_max_speakers
        # n_sp = 2
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
        # speech = self._load_wav_mono(sp_path, n_sp=n_sp) #[n_sp, wav]
        ## load the speeches
        speech = self.load_audio(sp_path, n_sp=n_sp) #[wav] #summed speech

        ## add reverb with probability
        if np.random.rand() < self.rir_probability:
            rir = self._load_rir(rr_path, n_sp = n_sp) #[n_sp, T]
            convolved_speeches = []
            # breakpoint()
            for i in range(n_sp):
                speech_mic1 = self._fft_convolve_same_len(speech, rir[i])
                # speech_mic2 = self._fft_convolve_same_len(speech[i], rir[1][i])

                #cap the lrngth to min of both mics
                # min_len = min(speech_mic1.numel(), speech_mic2.numel())
                # speech_mic1 = speech_mic1[:min_len]
                # speech_mic2 = speech_mic2[:min_len]
                #use both binaural mics
                convolved_speeches.append(speech_mic1) #dim: [T]
            speech = torch.stack(convolved_speeches, dim=0).sum(dim=0) #dim: [T]

        ## add noise with probability
        snr = random.uniform(self.global_snr_range[0], self.global_snr_range[1])
        noise = self.load_noise(nz_path, length=speech.shape[-1]) #gets noise as long as speech
        if random.random() < self.add_noise_prob:
            noisy = self._add_noise_at_snr(speech, noise, snr)
        else:
            noisy = speech

    
        # if speech.shape[0] == 1:
        #     speech = speech.unsqueeze(0) #[1, wav]
        
        # if config.config_mode == "paper" and n_sp ==2:
        #     sir_range = config.dataset_params.get("sir_range", (-5,5))  # dB
        #     sir_db = random.uniform(*sir_range)
        #     sir_ratio = db_to_ratio(sir_db)

        #     speech[0] = set_signal_energy(speech[0], 1.0)
        #     speech[1] = set_signal_energy(speech[1], 1.0 / (sir_ratio + 1e-8))  # adjust relative loudness



        # #add rir with a probability
        # if np.random.rand() < self.rir_probability:
        # # if 0 < self.rir_probability: #always add rir
        #     rir = self._load_rir(rr_path, n_sp = n_sp) #[n_mics, n_sources, T]
        #     convolved_speeches = []
        #     # breakpoint()
        #     for i in range(len(speech)):
        #         speech_mic1 = self._fft_convolve_same_len(speech[i], rir[0][i])
        #         speech_mic2 = self._fft_convolve_same_len(speech[i], rir[1][i])

        #         #cap the lrngth to min of both mics
        #         min_len = min(speech_mic1.numel(), speech_mic2.numel())
        #         speech_mic1 = speech_mic1[:min_len]
        #         speech_mic2 = speech_mic2[:min_len]
        #         #use both binaural mics
        #         convolved_speeches.append(torch.cat([speech_mic1.unsqueeze(0), speech_mic2.unsqueeze(0)], dim=0)) #dim: [2, T]
        #     speech = torch.stack(convolved_speeches, dim=0) #dim: [n_sp, 2, T]
        # else:
        #     #duplicate the single channel to make it binaural
        #     speech = torch.stack([speech, speech], dim=1) #dim: [n_sp, 2, T]
                

        # #combine speeches with the given overlap ratio
        # if np.random.rand() < self.overlap_prob:
        # # if 0 < self.overlap_prob: #always overlap
        #     overlap_ratio = np.random.uniform(0, self.overlap_ratio)

        #     speech = self._mix_overlap_add([speech[i] for i in range(len(speech))], self.overlap_ratio)  #[T]

        # else:
        #     speech = speech.transpose(0, 1).reshape(2, -1) 


        # # breakpoint()

        # noise = self._load_wav_mono(nz_path, length=speech.shape[-1], noise=True) #gets noise as long as speech

        # if random.random() < self.add_noise_prob:
        #     snr = random.uniform(self.global_snr_range[0], self.global_snr_range[1])
        #     noisy = self._add_noise_at_snr(speech, noise, snr)
        # else:
        #     noisy = speech
        # if self.peak_normalize:
        #     noisy = noisy / noisy.max(axis = -1)[0].unsqueeze(-1) #get only the max values and not indices
        
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



    ##############################

        # spk_ids = spk_ids[5:7]
        train_ids = []
        val_ids = []
        train_speech, val_speech = {}, {}

        for spk in spk_ids:
            utterances = self.speech_files[spk]
            if len(utterances) < 2:
                # skip speakers with only one utterance to avoid leakage
                continue

            random.shuffle(utterances)
            n_train_utts = int(0.8 * len(utterances))
            if n_train_utts == 0 or n_train_utts == len(utterances):
                # skip if split degenerates (all train or all val)
                continue

            train_speech[spk] = utterances[:n_train_utts]
            val_speech[spk] = utterances[n_train_utts:]

            train_ids.append(spk)
            val_ids.append(spk)
        
        self.train_speech = train_speech
        self.val_speech = val_speech
        self.train_num_class = len(train_ids)
        self.val_num_class = len(val_ids)



    ###############################################################################
        # # n_train = int(0.8 * len(spk_ids))

        # # train_ids = spk_ids[:n_train]
        # # val_ids = spk_ids[n_train:]

        # train_ids = spk_ids[5:7]
        # val_ids = spk_ids[20:22]
        # #print the number of class in train and test
        # print(f"Number of classes in train is {len(train_ids)}")
        # print(f"Number of classes in val is {len(val_ids)}")

        # self.train_num_class = len(train_ids)
        # self.val_num_class = len(val_ids)

        # self.train_speech = {spk: self.speech_files[spk] for spk in train_ids}
        # self.val_speech = {spk: self.speech_files[spk] for spk in val_ids}

    #################################################################################
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
        batch_sampler = NSPBatchSampler(
            self.train_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            n_sp_max=self.dataset_kwargs.get("N_max_speakers", 2),
            trainer=self.trainer
        )
        kwargs = dict(
            dataset=self.train_dataset,
            batch_sampler=batch_sampler,
            # batch_size=self.batch_size,
            # shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=use_workers,
            # drop_last=True,
            collate_fn=lambda b: collate_pair(b),
            timeout=0,
            worker_init_fn=_worker_init_fn,
        )
        if use_workers:
            kwargs["prefetch_factor"] = 2  # good general default
        
        # self.train_dataset.current_n_sp = random.randint(1, self.dataset_kwargs.get("N_max_speakers"))
        # print(f"[DataLoader] Using n_sp = {self.train_dataset.current_n_sp} for this batch/epoch")

        return DataLoader(**kwargs)

    def val_dataloader(self):
        batch_sampler = NSPBatchSampler(
            self.val_dataset,
            batch_size=self.batch_size,
            drop_last=False,
            n_sp_max=self.dataset_kwargs.get("N_max_speakers", 2),
            trainer=self.trainer
        )

        return DataLoader(
            self.val_dataset,
            batch_sampler=batch_sampler,
            # batch_size=self.batch_size,
            # shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False,
            # drop_last=False,
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
    speeches_list = "/mnt/disks/data/datasets/txts/paper_config/voxceleb_train.txt"
    noise_list = "/mnt/disks/data/datasets/txts/noise.txt"
    rir_list = "/mnt/disks/data/datasets/txts/rirs_dev.txt"
    # breakpoint()
    dataset = SpeakerIdentificationDM(
        speeches_list = speeches_list,
        noise_list = noise_list,
        rir_list = rir_list,
        N_max_speakers=2,
        overlap_ratio=0.2,
        desired_duration=8.0,
        sr=16000,
        segment_length=8.0,
        add_noise_prob=0.0,
        overlap_prob=0.5,
        rir_probability=0.0,
        global_snr=(0, 40),
        peak_normalization=True,
    )
    #get a sample
    # dataset.setup()
    # dl = dataset.train_dataloader()
    # breakpoint()
    # for batch in dl:
    #     x, y = batch
    #     print("noisy:", x.shape, "labels:", y.shape)
    #     break

    save_dir = "./dataset_samples"
    os.makedirs(save_dir, exist_ok=True)

    # Get one batch from the dataloader
    dataset.setup()
    dl = dataset.train_dataloader()
    # breakpoint()
    for i, (noisy, labels) in enumerate(dl):
        print("noisy:", noisy.shape, "labels:", labels.shape)
        # noisy shape: [B, 2, T] (binaural)
        # labels shape: [B, num_classes]
        for j in range(min(10, noisy.size(0))):
            wav = noisy[j]  # [T]
            # Peak normalize again for safe saving
            # breakpoint()
            wav = wav / wav.abs().max()
            torchaudio.save(
                os.path.join(save_dir, f"sample_{i:02d}_{j:02d}.wav"),
                wav.unsqueeze(0).cpu(),
                sample_rate=dataset.train_dataset.sr
            )
            print(f"Saved {save_dir}/sample_{i:02d}_{j:02d}.wav | active speakers: {labels[j].sum().item():.0f}")
        break  # just one batch