# _*_coding:utf-8_*_
import os
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
from accelerate import Accelerator
import matplotlib.pyplot as plt
from cp_diffusion_model import ConditionalDDPMPipeline

losses = []

def plot_loss(losses, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, condition_embedding):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process and config.output_dir:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, lr_scheduler, condition_embedding = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, condition_embedding
    )

    global_step = 0

    for epoch in range(config.num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            true_data, conditions = batch
            embedded_conditions = condition_embedding(conditions.float())
            noise = torch.randn(true_data.shape, device=true_data.device)
            bs = true_data.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=true_data.device, dtype=torch.int64)
            noisy_data = noise_scheduler.add_noise(true_data, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_data, timesteps, encoder_hidden_states=embedded_conditions, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_dataloader)
        losses.append(avg_loss)

        if accelerator.is_main_process and (epoch + 1) % config.save_model_epochs == 0:
            pipeline = ConditionalDDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            pipeline.save_pretrained(config.output_dir)

    plot_loss(losses,save_path=config.output_dir+'/training_loss.png')

