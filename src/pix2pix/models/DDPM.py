import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet, ControlNet
from generative.networks.schedulers import DDPMScheduler
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

def create_DDPM():
    device = torch.device("cuda")

    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        num_channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256,
    )
    model.to(device)
    return model

def train_DDPM(train_loader, test_loader):

    model = create_DDPM()

    device = torch.device("cuda")

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)

    inferer = DiffusionInferer(scheduler)

    if not os.path.exists('../../reports/DDPM'):
        os.makedirs('../../reports/DDPM')

    n_epochs = 70
    val_interval = 10
    epoch_loss_list = []
    val_epoch_loss_list = []
    max_batch = 100
    scaler = GradScaler()
    total_start = time.time()
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=max_batch, ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch["image"].to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                # Generate random noise
                noise = torch.randn_like(images).to(device)

                # Create timesteps
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                ).long()

                # Get model prediction
                noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)

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
            val_epoch_loss_list.append(val_epoch_loss / (step + 1))

            # Sampling image during training
            noise = torch.randn((1, 3, 64, 64))
            noise = noise.to(device)
            scheduler.set_timesteps(num_inference_steps=1000)
            with autocast(enabled=True):
                image = inferer.sample(input_noise=noise, diffusion_model=model, scheduler=scheduler)
            
            plt.figure(figsize=(2, 2))
            # the image is rgb, so we need to transpose it to be in the correct format
            plt.imshow(image[0].cpu().detach().numpy().transpose(1, 2, 0), vmin=0, vmax=1)
            plt.tight_layout()
            plt.axis("off")
            plt.savefig(f"../../reports/DDPM/DDPM_{epoch}.png")
            plt.close()

            torch.save(model.state_dict(), '../../models/DDPM.pt')

    total_time = time.time() - total_start
    print(f"train completed, total time: {total_time}.")
    #save model