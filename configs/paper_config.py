config_mode="paper"
#model params
model_name="mtse_embeddings_paper_oracle_speakers"
C=1024
d_model=1536
dprime_model=1536
emb_dim=192

#dataest paramss
dataset_params= dict(
speeches_list = "/mnt/disks/data/datasets/txts/paper_config/voxceleb_train.txt",
noise_list = "/mnt/disks/data/datasets/txts/noise.txt",
rir_list = "/mnt/disks/data/datasets/txts/rirs_dev.txt",
N_max_speakers=2,
overlap_ratio=1.0,
desired_duration=3.0,
sr=16000,
segment_length=3.0,
add_noise_prob=0.5,
overlap_prob=0.5,
rir_probability=0.5,
global_snr=(0, 40),
sir_range = (-5, 5),
peak_normalization=True,
num_workers=20,
batch_size=32)


#metric params
metric_name = "OT"
eps = 0.1
tau = 1.0


#loss params

loss_params=dict(
    loss_name="ArcFace",
    s=30.0,
    m=0.2,
    emb_dim = emb_dim
)


#Trainer params
max_epochs=400
devices=[0]
check_val_every_n_epoch=2
log_every_n_steps=10
gradient_clip_val=0.8
enable_checkpointing=True
# ckpt_path=None
# ckpt_path="/scratch/profdj_root/profdj0/sidcs/codebase/speaker_embedding_codebase/model_clean_4sp/best-checkpoint-epoch=18-val/loss=14.87.ckpt"
# ckpt_path = "/home/sidharth./codebase/speaker_embedding_codebase/model_clean_2sp/best-checkpoint-epoch=98-val/loss=11.17.ckpt"
ckpt_path = "/home/sidharth./codebase/speaker_embedding_codebase/ckpts/paper_oracle_speakers/best-checkpoint-epoch=14-val/loss=14.27.ckpt"
# ckpt_path = "/home/sidharth./codebase/speaker_embedding_codebase/model_noisy_2sp/best-checkpoint-epoch=122-val/loss=12.45.ckpt"
# ckpt_path = "/home/sidharth./codebase/speaker_embedding_codebase/model_noisyl_4sp/best-checkpoint-epoch=22-val/loss=14.83.ckpt"
# ckpt_path = "/home/sidharth./codebase/speaker_embedding_codebase/model_clean_4sp/best-checkpoint-epoch=110-val/loss=12.21.ckpt"

lr=5e-4
weight_decay=1e-5
cycle_length = 20
warmup_steps = 1000
decay_factor = 0.75
base_lr_factor = 0.1


#wandb params
project="mtse_speech_embeedings"
model_name="paper_oracle_speakers"