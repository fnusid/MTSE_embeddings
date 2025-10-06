import torch
import torch.nn 
import pytorch_lightning as pl
import torch.nn.functional as F
import wandb
from pytorch_lightning.loggers import WandbLogger
from metrics import MetricsWrapper
from recursive_attn_pooling import RecursiveAttnPooling
from loss import LossWrapper
import config
import ast
from dataset import SpeakerIdentificationDM

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
    
    def training_step(self, batch, batch_idx):
        noisy, labels = batch
        # x = noisy.mean(dim=1) #why?
        emb, p = self(noisy)
        loss = self.loss(emb, p, labels)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        import time
        start = time.time()
        breakpoint()
        noisy, labels = batch
        emb, p = self(noisy)
        mid = time.time()
        loss = self.loss(emb, p, labels)
        end = time.time()
        print(f"[Val step {batch_idx}] forward: {mid-start:.2f}s | loss: {end-mid:.2f}s")
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss
    
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
        logger=None,
        log_every_n_steps=config.log_every_n_steps,
        gradient_clip_val=config.gradient_clip_val,
        enable_checkpointing=True,
    )

    trainer.fit(model, dm)

    wandb.finish()