
#model params
model_name="mtse_embeddings_model_clean_2sp"
C=512
d_model=512
dprime_model=512
emb_dim=192
threshold_stop=0.2

#dataest paramss
dataset_params= dict(
speeches_list = "/nfs/turbo/coe-profdj/txts/voxceleb_train.txt",
noise_list = "/nfs/turbo/coe-profdj/txts/noise.txt",
rir_list = "/nfs/turbo/coe-profdj/txts/rirs_dev.txt",
N_max_speakers=2,
overlap_ratio=0.0,
desired_duration=8.0,
sr=16000,
segment_length=8.0,
add_noise_prob=0.0,
overlap_prob=0.0,
rir_probability=0.0,
global_snr=(0, 40),
peak_normalization=True,
num_workers=8,
batch_size=64)


#metric params
metric_name = "OT"
eps = 0.1
tau = 1.0


#loss params

loss_params=dict(
    loss_name="ArcFace",
    s=30.0,
    m=0.2,
    alpha=0.1,
    emb_dim = emb_dim
)


#Trainer params
max_epochs=400
devices=[0]
check_val_every_n_epoch=2
log_every_n_steps=10
gradient_clip_val=1.0
enable_checkpointing=True
ckpt_path="/scratch/profdj_root/profdj0/sidcs/codebase/speaker_embedding_codebase/model_clean_2sp/best-checkpoint-epoch=23-val/loss=18.97.ckpt"
lr=1e-4
weight_decay=1e-5


#wandb params
project="mtse_speech_embeedings"
model_name="model_clean_2sp"