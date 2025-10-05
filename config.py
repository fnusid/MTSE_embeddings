
#model params
C=512
d_model=512
dprime_model=512
emb_dim=192
threshold_stop=0.5

#dataest paramss
speech_dev_dir = '/nfs/turbo/coe-profdj/voxceleb_dev/'
speech_eval_dir = '/nfs/turbo/coe-profdj/voxceleb_eval/'
noise_dir = '/scratch/profdj_root/profdj0/shared_data/MUSAN'
rir_dir = 'PATH_TO_RIRS_NOISES'


#metric params
eps = 0.1
tau = 1.0