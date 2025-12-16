#训练 CViViT 模型，以将视频压缩为离散的视觉令牌（Token）
#     folder = '',
#     batch_size = 16,
#     num_frames = 11,
#     grad_accum_every = 4,
#     train_on_images = False,
#     use_ema = True,
#     num_train_steps = 500_000,
#     save_model_every = 5000,
#     accelerate_kwargs = accelerate_kwargs

import torch
# 从核心引擎文件夹引入 CViViT（压缩模型）和 CViViTTrainer（训练管家）
from phenaki_pytorch import CViViT, CViViTTrainer
# 引入加速工具，用于多显卡训练
from accelerate import Accelerator

# --- 配置加速器Accelerator参数 ---
#使用 fp16 混合精度（省显存，跑得快）？？？？？？细说
accelerate_kwargs = {
    'mixed_precision': 'fp16',  # use mixed precision training
    'split_batches': True
}


# --- 初始化CViViT模型 ---
# 这里定义了视频压缩模型的“形状”
cvivit = CViViT(
    dim = 512,# 隐藏层特征维度，越大模型越聪明但越慢
    codebook_size = 8192,# 密码本大小。视频会被压缩成 0-8191 之间的数字
    image_size = 128,# 输入视频的画面大小 128x128 像素
    patch_size = 32,# 把图像切成 32x32 的小块来处理
    temporal_patch_size = 2,# 在时间上每 2 帧切一块
    spatial_depth = 4,# 空间处理层有 4 层
    temporal_depth = 4,# 时间处理层有 4 层
    dim_head = 64,# 注意力机制的头大小
    heads = 8# 注意力机制的头数量
).cuda()

# Use Accelerator for model and data preparation
# accelerator = Accelerator(**accelerate_kwargs)
# cvivit, = accelerator.prepare(cvivit)

# Initialize the trainer
# --- 初始化训练管家 ---
# Trainer 负责具体怎么喂数据、怎么更新模型参数
trainer = CViViTTrainer(
    vae=cvivit,  # Pass the unwrapped model 把上面定义的模型交给管家
    folder='', # 【关键！】这里需要填入视频数据的文件夹路径，现在是空的，不填会报错！
    batch_size=128, # 一次训练看 128 个视频片段
    num_frames=11, # 每个视频片段取 11 帧
    grad_accum_every=4, # 梯度累积，相当于变相扩大 batch_size 为 128*4=512
    #？？？？细说？？？？
    train_on_images=False, # 我们训练的是视频，不是单张图片
    use_ema=True, # 一种让模型训练更稳定的技巧
    num_train_steps=1000000,# 总共训练 100万步
    save_model_every=5000, # 每 5000 步保存一次模型
    accelerate_kwargs=accelerate_kwargs # 传入加速配置
)

# --- 开始训练 ---
# 启动训练循环。结果会保存在 ./results 文件夹
trainer.train()
