# _*_coding:utf-8_*_
from cp_diffusion.cp_diffusion_model import ConditionalDDPMPipeline
from dataclasses import dataclass
from cp_diffusion.generate import generate_labels

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

# 从保存的模型中加载
loaded_pipeline = ConditionalDDPMPipeline.from_pretrained(config.output_dir)

# Labels for generation

labels = [1,1,1,35]
generate_condition = generate_labels(labels, config)

# 使用加载的 pipeline 进行生成
generated_images = loaded_pipeline(
    batch_size=config.eval_batch_size,
    condition_embedding=generate_condition,
    num_inference_steps=1000
)
