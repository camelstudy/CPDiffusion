# _*_coding:utf-8_*_
from cp_diffusion_model import get_model,get_scheduler,ConditionEmbedding
from diffusers.optimization import get_cosine_schedule_with_warmup
from deal_data import load_data
from train import train_loop

import torch
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    data_size = (200, 4)  # the generated image resolution
    train_batch_size = 32
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-128"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
config = TrainingConfig()

train_dataloader = load_data('datas.pkl', config.train_batch_size)
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