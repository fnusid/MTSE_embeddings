import torch
from dataset import SpeakerIdentificationDM
from trainer import SpeakerEmbeddingModule
from configs import paper_config as config




if __name__ == '__main__':
    dm = SpeakerIdentificationDM(**config.dataset_params)
    dm.setup()

    sample_batch = next(iter(dm.train_dataloader()))
    num_classes = sample_batch[1].shape[1]
    ckpt_path = "/home/sidharth./codebase/speaker_embedding_codebase/ckpts/model_noisypaper_2sp/best-checkpoint-epoch=347-val/loss=10.32.ckpt"  # Replace with actual checkpoint path
    model = SpeakerEmbeddingModule.load_from_checkpoint(ckpt_path, config=config, num_class=num_classes)
    model.eval()

    batch = next(iter(dm.val_dataloader()))
    noisy, labels = batch
    noisy, labels = noisy.to(model.device), labels.to(model.device)

    model.zero_grad()
    emb, p = model(noisy)
    loss, loss_dict = model.loss(emb, p, labels)
    loss.backward()

    grad_norms = {name: param.grad.norm().item() for name, param in model.named_parameters() if param.grad is not None}

    for name, val in sorted(grad_norms.items(), key=lambda x: -x[1])[:10]:
        print(f"{name:40s}: {val:.6f}")

    print(f"\nTotal parameters: {len(grad_norms)}")
    print(f"Max grad norm: {max(grad_norms.values()):.4f}")
    print(f"Mean grad norm: {sum(grad_norms.values()) / len(grad_norms):.4f}")

    import matplotlib.pyplot as plt
    plt.hist(list(grad_norms.values()), bins=50)
    plt.xlabel("Gradient Norm")
    plt.ylabel("Frequency")
    plt.title("Distribution of Gradient Magnitudes")
    plt.show()
    plt.savefig("gradient_magnitude_distribution.png")