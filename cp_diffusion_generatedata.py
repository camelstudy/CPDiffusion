# _*_coding:utf-8_*_
from cp_diffusion.cp_diffusion_model import ConditionalDDPMPipeline,ConditionEmbedding
from cp_diffusion.generate import generate_labels,decoder2seq
from config_file import TrainingConfig

config = TrainingConfig()

# 从保存的模型中加载
loaded_pipeline = ConditionalDDPMPipeline.from_pretrained(config.output_dir)

# Labels for generation
condition_embedding = ConditionEmbedding(4, 128)

labels = [1,1,1,35]
generate_condition = generate_labels(labels, config)

# 使用加载的 pipeline 进行生成
generate = loaded_pipeline(
    batch_size=config.eval_batch_size,
    condition_embedding=condition_embedding(generate_condition),
    num_inference_steps=1000
)
for i, data in enumerate(generate):
    # print(data.shape)
    rd = decoder2seq(data)
    print(f"Generated sample {i}: {rd}")