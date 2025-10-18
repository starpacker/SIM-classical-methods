from sparse_recon.sparse_deconv import sparse_deconv
import mrcfile
import numpy as np

# 配置参数
PSF_FACTOR = 5          
SPARSE_ITER = 1000    

# 输入和输出文件路径（请根据实际情况修改）
input_path = "C://sim//data//CCP//RawSIMData_level_01.mrc"   # 修改为 MRC 文件
output_path = "C://sim//data//CCP//output.mrc"  # 输出也为 MRC

# input_path = "C:/sim/data/ER/RawSIMData_level_01.mrc"
# output_path = "C:/sim/data/ER/output.mrc"

input_path = "C:/sim/data/f-actin/RawSIMData_level_01.mrc"
output_path = "C:/sim/data/f-actin/output.mrc"

# 读取 MRC 文件
with mrcfile.open(input_path, mode='r', permissive=True) as mrc:  # permissive=True 处理非标准文件
    # 访问头信息
    header = mrc.header
    print(f"维度: {header.nx} x {header.ny} x {header.nz}")
    print(f"数据类型 (mode): {header.mode}")
    
    # 读取数据到 NumPy 数组（形状为 (nz, ny, nx)，Z-Y-X 顺序）
    data = mrc.data.astype(np.float32)  # 转换为 float32 以便处理
    print(f"数据形状: {data.shape}")
    print(f"数据类型: {data.dtype}")

    im = mrc.data
    print(f"📊 输入图像形状: {im.shape}, 数据类型: {im.dtype}")

# 执行稀疏反卷积
img_recon = sparse_deconv(im, PSF_FACTOR, sparse_iter=SPARSE_ITER)

# 保存重建结果到 MRC 文件
with mrcfile.new(output_path, overwrite=True) as mrc_out:
    mrc_out.set_data(img_recon.astype(np.float32))  # 确保数据类型一致
    # 可选：复制原始头信息（如 cell dimensions, pixel size 等）
    with mrcfile.open(input_path, permissive=True) as mrc_in:
        mrc_out.header.nx = mrc_in.header.nx
        mrc_out.header.ny = mrc_in.header.ny
        mrc_out.header.nz = mrc_in.header.nz
        mrc_out.header.mode = 2  # float32 对应 mode=2
        mrc_out.header.cella = mrc_in.header.cella
        mrc_out.header.mapc = mrc_in.header.mapc
        mrc_out.header.mapr = mrc_in.header.mapr
        mrc_out.header.maps = mrc_in.header.maps
        # 如果你知道像素尺寸存储在 header 中（如 mrc_in.voxel_size），也可以设置：
        mrc_out.voxel_size = mrc_in.voxel_size
