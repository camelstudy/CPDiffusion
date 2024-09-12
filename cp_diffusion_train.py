# _*_coding:utf-8_*_
from cp_diffusion.cp_diffusion_model import get_model,get_scheduler,ConditionEmbedding
from diffusers.optimization import get_cosine_schedule_with_warmup
from cp_diffusion.deal_data import load_data
from cp_diffusion.train import train_loop
from config_file import TrainingConfig
import torch

config = TrainingConfig()

train_dataloader = load_data(config.data_path, config.train_batch_size)
model = get_model(config.data_size)
noise_scheduler = get_scheduler()
condition_embedding = ConditionEmbedding(4, 128)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

# Start training

train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, condition_embedding)