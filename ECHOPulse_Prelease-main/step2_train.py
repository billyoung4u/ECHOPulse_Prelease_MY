import sys
import os

# 此部分训练生成模型 (Step 2) 这是核心部分，利用 ECG 生成视频

# 设置使用哪张显卡。'0' 代表第一张卡。如果是单卡，不用动；多卡可能需要改。
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # 仅启用GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 把上一级目录加入系统路径，这样才能导入同级文件夹里的模块
#这两句具体什么意思？？？？？

#为了快速复现注释掉了下面两行
# parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
# sys.path.append(parent_dir)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# import torch
# from phenaki_pytorch.cvivit import CViViT # 引入 Step 1 里的压缩模型结构
# from phenaki_pytorch.cvivit_trainer import CViViTTrainer
# #
# from phenaki_pytorch.phenaki_pytorch_ekg import MaskGit, Phenaki
# from phenaki_pytorch.phenaki_trainer_ekg import PhenakiTrainer

import torch
# 这里的 CViViT 相关的可以用原来库里的，也可以用本地的，建议统一改为本地路径以防版本冲突
from EchoPulse_pytorch.cvivit import CViViT
from EchoPulse_pytorch.cvivit_trainer import CViViTTrainer

# 关键修改：将 phenaki_pytorch 改为 EchoPulse_pytorch
# 引入 Step 2 的主角：MaskGit (生成模型) 和 Phenaki (把压缩模型和生成模型包在一起)
from EchoPulse_pytorch.phenaki_pytorch_ekg import MaskGit, Phenaki
from EchoPulse_pytorch.phenaki_trainer_ekg import PhenakiTrainer
# 用于多卡并行（这里其实主要靠 Accelerator）
from torch.nn import DataParallel

# --- 第一步：定义压缩模型 (Tokenizer) ---
# 这些参数必须和 step1_train.py 里完全一样，否则权重加载会失败
#？？？？为何？？？？
cvivit = CViViT(
    dim=512,  # embedding后的隐藏层特征维度，越大模型越聪明但越慢
    codebook_size=8192,  # 密码本大小。视频会被压缩成 0-8191 之间的数字
    image_size=128,  # H,W
    patch_size=8,  # 把图像切成 32x32 的小块来处理
    local_vgg=True,
    wandb_mode='disabled',
    temporal_patch_size=2,  # 在时间上每 2 帧切一块
    spatial_depth=4,  # 空间处理层有 4 层
    temporal_depth=4,  # 时间处理层有 4 层
    dim_head=64,  # 注意力机制的头大小
    heads=8,  # 注意力机制的头数量
    ff_mult=4,  # 32 * 64 = 2048 MLP size in transfo out
    commit_loss_w=1.,  # commit loss weight
    gen_loss_w=1.,  # generator loss weight
    perceptual_loss_w=1.,  # vgg loss weight
    i3d_loss_w=1.,  # i3d loss weight
    recon_loss_w=10.,  # reconstruction loss weight
    use_discr=0,  # whether to use a stylegan loss or not
    gp_weight=10



)

# --- 第二步：加载 Step 1 训练好的权重 ---
# 【注意！】这是一个绝对路径，是你电脑上没有的路径。
# 你必须把它改成你自己下载的权重文件路径（例如 './Model_weights/pytorch_model.bin'）
cvivit.load('./Model_weights/CVIVIT_pytorch_model_finetune.bin')
# cvivit.load('/raid/home/CAMCA/yl463/Video/results/ckpt_accelerate_20/pytorch_model.bin')

# --- 第三步：定义生成模型 (MaskGit) ---
# 这个模型负责根据 ECG 预测 Token
maskgit = MaskGit(
    num_tokens = 8192,  # 必须等于 codebook_size，代表密码本里有多少种密码
    max_seq_len = 2048,# 生成的视频序列最大长度
    dim = 512,# 模型内部维度
    dim_context = 768,# ECG 信号的特征维度 (来自 ST-MEM 模型)
    depth = 6,# 这是一个 6 层的 Transformer
)

# --- 第四步：合体 ---
# Phenaki 是一个大容器，里面装着 cvivit (负责看和画) 和 maskgit (负责想)
phenaki = Phenaki(
    cvivit = cvivit,
    maskgit = maskgit
).cuda()

# --- 第五步：初始化训练器 ---
trainer = PhenakiTrainer(
    phenaki = phenaki,
    folder =  'MyToyDataset', # 【关键！】同样需要填入你的视频数据路径
    train_on_images = False,# 我们训练的是视频，不是单张图片
    #为了快速复现改了，下方原值是50
    batch_size = 1,
    #为了快速复现改了，下方原值是1
    grad_accum_every = 1, # 梯度累积，相当于没有变相扩大 batch_size
    num_frames = 11, # 每个视频片段取 11 帧
    sample_num_frames = None,# 在采样时生成的视频帧数，None代表和训练时一样
    train_lr = 1e-4,# 训练学习率
    #为了快速复现改了，下方原值是1000_002
    train_num_steps = 1000,# 总共训练 100万步
    max_grad_norm = None,## 最大梯度裁剪，None代表不裁剪
    ema_update_every = 10,# 每 10 步更新一次 EMA 模型
    ema_decay = 0.995,# EMA 衰减率
    adam_betas = (0.9, 0.99),# Adam 优化器的 beta 参数
    wd = 0,## 权重衰减
    #为了快速复现改了，下方原值是10000
    save_and_sample_every = 50,# 每 10000 步保存一次模型和采样一次
    num_samples = 25, # 每次采样生成 25 个视频
    results_folder = '',# 采样结果保存路径
    amp = True,## 是否使用自动混合精度
    fp16 = True,## 是否使用半精度浮点数
    split_batches = True,## 是否拆分批次以节省内存
    # 下面这两个参数如果不跑文本生成，可以忽略
    convert_image_to = None,## 是否转换图像格式，None代表不转换
    sample_texts_file_path = './test_2c4c.txt', # 采样时使用的文本文件路径
    losses_file_folder = './results_step2',# 损失文件保存路径


)

# --- 开始 Step 2 训练 ---
trainer.train()

