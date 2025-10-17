import os
import numpy as np
import torch
from tifffile import imread, imsave
from div2k import normalize, PercentileNormalizer
# 假设你已将原始代码中的 utility 和 model 模块放在同级目录
import utility
import model

# ----------------------------
# 配置参数（可按需修改）
# ----------------------------
TASK = "SR_Microtubules"  # 可选: "SR_Microtubules", "SR_CCPs", "SR_F-actin", "SR_ER"
INPUT_TIF = "001.tif"   # 输入图像路径（必须是 2D .tif）
OUTPUT_TIF = "output_sr.tif"  # 输出路径

# ----------------------------
# 模型配置类（简化版 Args）
# ----------------------------
class Args:
    model = 'SwinIR'
    test_only = True
    resume = 0
    modelpath = None
    save = None
    task = None
    dir_data = None
    dir_demo = None
    data_test = None

    epoch = 1000
    batch_size = 16
    patch_size = None
    rgb_range = 1
    n_colors = 1
    inch = None
    datamin = 0
    datamax = 100
    
    cpu = False
    print_every = 1000
    test_every = 2000
    load = ''
    lr = 0.00005
    n_GPUs = 1
    n_resblocks = 8
    n_feats = 32
    save_models = True
    save_results = True
    save_gt = False

    debug = False
    scale = None
    chunk_size = 144
    n_hashes = 4
    chop = False
    self_ensemble = False
    no_augment = False
    inputchannel = None

    act = 'relu'
    extend = '.'
    res_scale = 0.1
    shift_mean = True
    dilation = False
    precision = 'single'

    seed = 1
    local_rank = 0
    n_threads = 0
    reset = False
    split_batch = 1
    gan_k = 1
# ----------------------------
# 设置模型路径
# ----------------------------
args = Args()
args.task = 1
args.patch_size = 128
args.scale = '2'
args.inch = 1
      
if TASK == "SR_Microtubules":
    args.save = "SwinIRMicrotubules"
    args.modelpath = "./experiment/SwinIRMicrotubules/model_best.pt"
elif TASK == "SR_CCPs":
    args.save = "SwinIRCCPs"
    args.modelpath = "./experiment/SwinIRCCPs/model_best.pt"
elif TASK == "SR_F-actin":
    args.save = "SwinIRF-actin"
    args.modelpath = "./experiment/SwinIRF-actin/model_best181.pt"
elif TASK == "SR_ER":
    args.save = "SwinIRER"
    args.modelpath = "./experiment/SwinIRER/model_best147.pt"
else:
    raise ValueError(f"Unsupported task: {TASK}")

# 检查模型文件是否存在
if not os.path.exists(args.modelpath):
    raise FileNotFoundError(f"Model not found at {args.modelpath}. Please download it first.")

# ----------------------------
# 加载模型
# ----------------------------
print("Loading model...")
checkpoint = utility.checkpoint(args)
model_sr = model.Model(args, checkpoint)
model_sr.eval()
device = torch.device('cpu' if args.cpu else 'cuda')
model_sr = model_sr.to(device)
print(f"Model loaded on {device}.")

# ----------------------------
# 读取输入图像
# ----------------------------
# ----------------------------
# 读取输入图像
# ----------------------------
print(f"Reading input image: {INPUT_TIF}")
if not os.path.exists(INPUT_TIF):
    raise FileNotFoundError(f"Input file not found: {INPUT_TIF}")

image = imread(INPUT_TIF)

# 处理 (H, W, 1) 格式的灰度图
# print(image.shape)
if image.ndim == 3 and image.shape[-1] == 1:
    image = image[:, :, 0]

# print(image.ndim)
if image.ndim != 2:
    raise ValueError("Input image must be 2D (grayscale) for super-resolution.")

print(f"Input shape: {image.shape}")

# ----------------------------
# 预处理：归一化到 [0, 1] * rgb_range
# ----------------------------
lr = normalize(image, args.datamin, args.datamax, clip=True) * args.rgb_range
lr_tensor = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).float().to(device)  # [1, 1, H, W]

# ----------------------------
# 推理
# ----------------------------
print("Running super-resolution...")
with torch.no_grad():
    sr_tensor = model_sr(lr_tensor, 0)

# 反归一化 & 转为 numpy
sr = utility.quantize(sr_tensor, args.rgb_range)
sr = sr.squeeze().cpu().numpy().astype(np.float32)

print(f"Output shape: {sr.shape}")

# ----------------------------
# 保存结果
# ----------------------------
imsave(OUTPUT_TIF, sr)
print(f"Super-resolved image saved to: {OUTPUT_TIF}")