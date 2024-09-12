# _*_coding:utf-8_*_
import torch
from diffusers import UNet2DConditionModel, DDPMScheduler, DDPMPipeline

class ConditionEmbedding(torch.nn.Module):
    def __init__(self, condition_dim, embed_dim):
        super().__init__()
        self.embed = torch.nn.Linear(condition_dim, embed_dim)

    def forward(self, conditions):
        return self.embed(conditions)

class ConditionalDDPMPipeline(DDPMPipeline):
    def __init__(self, unet, scheduler):
        super().__init__(unet=unet, scheduler=scheduler)

    def __call__(self, batch_size, condition_embedding, num_inference_steps=1000, generator=None):
        device = self.unet.device
        noise = torch.randn(batch_size, 1, *self.unet.sample_size).to(device)

        for t in range(num_inference_steps - 1, -1, -1):
            timesteps = torch.full((batch_size,), t, dtype=torch.long, device=device)
            with torch.no_grad():
                model_output = self.unet(noise, timesteps, encoder_hidden_states=condition_embedding).sample
            noise = self.scheduler.step(model_output, t, noise).prev_sample

        return noise

def get_model(data_size):
    return UNet2DConditionModel(
        sample_size=data_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        cross_attention_dim=128
    )

def get_scheduler():
    return DDPMScheduler(num_train_timesteps=1000)
