1.cp_diffusion_model.py 为模型信息，包括条件嵌入函数，unit参数定义等。
2.deal_data.py 包括读取、处理以及生成pytorch的Dataset。
3.generate.py 包含生成数据时所需要的函数
4.train.py 为模型训练代码，采用了accelerate库来优化模型的训练