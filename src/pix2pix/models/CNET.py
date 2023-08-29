from .DDPM import create_DDPM
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet, ControlNet
from generative.networks.schedulers import DDPMScheduler
from monai.utils import first
import torch
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

def create_cnet():

    device = torch.device("cuda")

    model = create_DDPM()
    model.to(device)
    model.load_state_dict(torch.load('../../models/DDPM.pt'))

    # Create control net
    controlnet = ControlNet(
        spatial_dims=2,
        in_channels=3,
        num_channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256,
        conditioning_embedding_num_channels=(16,),
    )
    # Copy weights from the DM to the controlnet
    controlnet.load_state_dict(model.state_dict(), strict=False)
    controlnet = controlnet.to(device)
    # Now, we freeze the parameters of the diffusion model.
    for p in model.parameters():
        p.requires_grad = False

    return model, controlnet

def train_cnet(train_loader, test_loader):

    model, controlnet = create_cnet()

    device = torch.device("cuda")

    if not os.path.exists('../../reports/CNET'):
        os.makedirs('../../reports/CNET')

    n_epochs = 50
    val_interval = 10
    epoch_loss_list = []
    val_epoch_loss_list = []
    max_batch = 100

    optimizer = torch.optim.Adam(params=controlnet.parameters(), lr=2.5e-5)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    inferer = DiffusionInferer(scheduler)

    scaler = GradScaler()
    total_start = time.time()
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=max_batch, ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:

            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=True):

                # Generate random noise
                noise = torch.randn_like(images).to(device)

                # Create timesteps
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                ).long()

                images_noised = scheduler.add_noise(images, noise=noise, timesteps=timesteps)

                # Get controlnet output
                down_block_res_samples, mid_block_res_sample = controlnet(
                    x=images_noised, timesteps=timesteps, controlnet_cond=masks
                )
                # Get model prediction
                noise_pred = model(
                    x=images_noised,
                    timesteps=timesteps,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )

                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
            if step == max_batch:
                break
        epoch_loss_list.append(epoch_loss / (step + 1))

        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_epoch_loss = 0
            for step, batch in enumerate(test_loader):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                with torch.no_grad():
                    with autocast(enabled=True):
                        noise = torch.randn_like(images).to(device)
                        timesteps = torch.randint(
                            0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                        ).long()
                        noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
                        val_loss = F.mse_loss(noise_pred.float(), noise.float())

                val_epoch_loss += val_loss.item()
                progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
                break
            val_epoch_loss_list.append(val_epoch_loss / (step + 1))

            # Sampling image during training with controlnet conditioning
            progress_bar_sampling = tqdm(scheduler.timesteps, total=len(scheduler.timesteps), ncols=110)
            progress_bar_sampling.set_description("sampling...")
            sample = torch.randn((1, 3, 64, 64)).to(device)
            for t in progress_bar_sampling:
                with torch.no_grad():
                    with autocast(enabled=True):
                        down_block_res_samples, mid_block_res_sample = controlnet(
                            x=sample, timesteps=torch.Tensor((t,)).to(device).long(), controlnet_cond=masks[0, None, ...]
                        )
                        noise_pred = model(
                            sample,
                            timesteps=torch.Tensor((t,)).to(device),
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                        )
                        sample, _ = scheduler.step(model_output=noise_pred, timestep=t, sample=sample)

            plt.subplots(1, 2, figsize=(4, 2))
            plt.subplot(1, 2, 1)
            plt.imshow(masks[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
            plt.axis("off")
            plt.title("Conditioning mask")
            plt.subplot(1, 2, 2)
            plt.imshow(sample[0].cpu().detach().numpy().transpose(1, 2, 0), vmin=0, vmax=1, cmap="gray")
            plt.axis("off")
            plt.title("Sample image")
            plt.tight_layout()
            plt.axis("off")
            plt.savefig(f"../../reports/CNET/cnet_sample_{epoch}.png", dpi=300)
            plt.close()
            torch.save(controlnet.state_dict(), '../../models/CNET.pt')

    total_time = time.time() - total_start
    print(f"train completed, total time: {total_time}.")



    progress_bar_sampling = tqdm(scheduler.timesteps, total=len(scheduler.timesteps), ncols=110, position=0, leave=True)
    progress_bar_sampling.set_description("sampling...")
    num_samples = 8
    sample = torch.randn((num_samples, 3, 64, 64)).to(device)

    val_batch = first(train_loader)
    val_images = val_batch["image"].to(device)
    val_masks = val_batch["mask"].to(device)
    for t in progress_bar_sampling:
        with torch.no_grad():
            with autocast(enabled=True):
                down_block_res_samples, mid_block_res_sample = controlnet(
                    x=sample, timesteps=torch.Tensor((t,)).to(device).long(), controlnet_cond=val_masks[:num_samples, ...]
                )
                noise_pred = model(
                    sample,
                    timesteps=torch.Tensor((t,)).to(device),
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )
                sample, _ = scheduler.step(model_output=noise_pred, timestep=t, sample=sample)

    plt.subplots(num_samples, 3, figsize=(6, 8))
    for k in range(num_samples):
        plt.subplot(num_samples, 3, k * 3 + 1)
        plt.imshow(val_masks[k, 0, ...].cpu(), vmin=0, vmax=1, cmap="gray")
        plt.axis("off")
        if k == 0:
            plt.title("Conditioning mask")
        plt.subplot(num_samples, 3, k * 3 + 2)
        plt.imshow(val_images[k].cpu().detach().numpy().transpose(1, 2, 0), vmin=0, vmax=1)
        plt.axis("off")
        if k == 0:
            plt.title("Actual val image")
        plt.subplot(num_samples, 3, k * 3 + 3)
        plt.imshow(sample[k].cpu().detach().numpy().transpose(1, 2, 0), vmin=0, vmax=1)
        plt.axis("off")
        if k == 0:
            plt.title("Sampled image")
    plt.tight_layout()
    plt.savefig(f"../../reports/CNET/cnet_final.png", dpi=300)
    plt.close()
