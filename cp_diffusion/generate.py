# _*_coding:utf-8_*_
import torch

def generate_labels(label,config):
    labels = torch.tensor(label, dtype=torch.float32)
    batch_labels = labels.unsqueeze(0).repeat(config.eval_batch_size, 1)
    return batch_labels.unsqueeze(1)


def decoder2seq(x):
    seq = ''
    for i in range(x.shape[0]):
        bases = x[i]
        max_idx = torch.argmax(bases).item()
        seq += 'GCTA'[max_idx]
    return seq

def evaluate(config, epoch, pipeline, embedded_conditions):
    datas = pipeline(
        batch_size=config.eval_batch_size,
        condition_embedding=embedded_conditions,
        generator=torch.Generator(device='cpu').manual_seed(config.seed)
    )

    for i, data in enumerate(datas):
        rd = decoder2seq(data)
        print(f"Generated sample {i} at epoch {epoch}: {rd}")
