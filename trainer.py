import torch
import torch.nn 
import pytorch_lightning as pl
import torch.nn.functional as F
import wandb
from pytorch_lightning.loggers import WandbLogger
from metrics import MetricsWrapper
from recursive_attn_pooling import RecursiveAttnPooling
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from loss import LossWrapper
import config
import os
import ast
from dataset import SpeakerIdentificationDM
import warnings
warnings.filterwarnings("ignore", module="torchaudio")

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def debug_gradients_and_losses(model, loss_dict):
    """
    Prints per-term loss magnitudes and gradient norms of key modules.
    Call this once every few training steps or at the end of each epoch.
    """
    print("\n--- LOSS COMPONENTS ---")
    for name, val in loss_dict.items():
        if torch.is_tensor(val):
            print(f"{name:>15}: {val.detach().cpu().item():8.4f}")
        else:
            print(f"{name:>15}: {val:.4f}")
    
    print("\n--- GRADIENT NORMS ---")
    grad_info = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_info[name] = p.grad.data.abs().mean().item()
    # Sort by largest gradients
    for k,v in sorted(grad_info.items(), key=lambda x: -x[1])[:10]:
        print(f"{k:>40}: {v:.5f}")
    print("-----------------------\n")


class SpeakerEmbeddingModule(pl.LightningModule):
    def __init__(self, config, num_class):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])
        self.model = RecursiveAttnPooling(encoder=None, config=config)
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        #define num_class after dataset
        self.num_class = num_class
        emb_dim = config.emb_dim
        self.loss = LossWrapper(num_class=num_class,**config.loss_params)

        self.metric = MetricsWrapper([config.metric_name])



    def forward(self, x):
        emb, p = self.model(x)
        # emb = emb.mean(dim = 1)
        return emb, p

    def on_train_epoch_start(self):
        if hasattr(self.loss, "update_schedules"):
            self.loss.update_schedules(self.current_epoch)

    def on_validation_epoch_start(self):
        if hasattr(self.loss, "update_schedules"):
            self.loss.update_schedules(self.current_epoch)
    
    def training_step(self, batch, batch_idx):
        if hasattr(self.loss.loss_fn, "update_schedules"):
            self.loss.loss_fn.update_schedules(self.current_epoch)

        noisy, labels = batch
        # x = noisy.mean(dim=1) #why?
        emb, p = self(noisy)
        total_loss, loss_dict = self.loss(emb, p, labels)
        # if batch_idx % 500 == 0:
        #     # 1️⃣ Backward pass first to populate gradients
        #     self._debug_loss_dict = loss_dict

        # self.log("train/loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, prog_bar=False, on_step=True, on_epoch=True)
 

        return total_loss
    
    def on_after_backward(self):
        """Lightning automatically calls this after every backward()."""
        if hasattr(self, "_debug_loss_dict") and self.global_step % 500 == 0:
            
            debug_gradients_and_losses(self, self._debug_loss_dict)

    def validation_step(self, batch, batch_idx):
        # import time
        # start = time.time()
        # breakpoint()
        noisy, labels = batch
        emb, p = self(noisy)
        # mid = time.time()
        total_loss, loss_dict = self.loss(emb, p, labels)
        # end = time.time()
        # print(f"[Val step {batch_idx}] forward: {mid-start:.2f}s | loss: {end-mid:.2f}s")
        self.log("val/loss", total_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=self.lr / 10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

if __name__ == "__main__":


    #Initialize W&B logger
    wandb_logger = WandbLogger(
        project=config.project,
        name=config.model_name,
        log_model=True,
        save_dir="./wandb_logs"
    )

    NUM_WORKERS = min(len(os.sched_getaffinity(0)), config.dataset_params['num_workers'])
    config.dataset_params['num_workers'] = NUM_WORKERS
    print(f"USING {NUM_WORKERS} NUM_WORKERS ")
    dm = SpeakerIdentificationDM(**config.dataset_params)
    dm.setup()

    #Model
    sample_batch = next(iter(dm.train_dataloader()))
    num_classes = sample_batch[1].shape[1]
    model = SpeakerEmbeddingModule(config, num_class=num_classes)

    #Trainer with WandB logger
    
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config.devices,
        logger=wandb_logger,
        log_every_n_steps=config.log_every_n_steps,
        gradient_clip_val=config.gradient_clip_val,
        enable_checkpointing=True,
        callbacks=[
            EarlyStopping(monitor='val/loss', patience=20, mode='min'),
            ModelCheckpoint(dirpath=f'/scratch/profdj_root/profdj0/sidcs/codebase/speaker_embedding_codebase/{config.model_name}', monitor='val/loss', mode='min', save_top_k=3, filename='best-checkpoint-{epoch:02d}-{val/loss:.2f}')
        ],
    )
    # trainer = pl.Trainer(
    #     accelerator="gpu",
    #     devices=[0],
    #     max_epochs=100,
    #     overfit_batches=1,   # ← magic line
    #     log_every_n_steps=1,
    #     enable_checkpointing=False
    # )

    trainer.fit(model, dm, ckpt_path= config.ckpt_path)

    wandb.finish()