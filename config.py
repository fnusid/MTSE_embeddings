
#model params
C=512
d_model=512
dprime_model=512
emb_dim=192
threshold_stop=0.5

#dataest paramss
dataset_params= dict(
speeches_list = "/nfs/turbo/coe-profdj/txts/voxceleb_train.txt",
noise_list = "/nfs/turbo/coe-profdj/txts/noise.txt",
rir_list = "/nfs/turbo/coe-profdj/txts/rirs_dev.txt",
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
num_workers=0,
batch_size=8)


#metric params
metric_name = "OT"
eps = 0.1
tau = 1.0


#loss params
#define n_class after the train, valid split
loss_params=dict(
    loss_name="MagFace",
    alpha=0.01,
    beta=100,
    s=30.0,
    lmbda=35.0,
    gamma=0.2,
    low=10.0,
    high=110.0,
    feat_dim=emb_dim,
    eta=100,
    xi=100,
    p_stop=0.5,
    c_miss=20.0,
    c_extra=10.0
)


#Trainer params
max_epochs=400
devices=[0]
check_val_every_n_epoch=2
log_every_n_steps=10
gradient_clip_val=1.0
enable_checkpointing=True
ckpt_path=None
lr=1e-4
weight_decay=1e-5


#wandb params
project="mtse_speech_embeedings"
model_name="vanila"