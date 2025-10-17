from sparse_recon.sparse_deconv import sparse_deconv
from skimage import io

# 配置参数
PSF_FACTOR = 5          
SPARSE_ITER = 1000    

# 输入和输出文件路径（请根据实际情况修改）
input_path = "001.tif"
output_path = "output.tif"

# 读取图像
im = io.imread(input_path)

# 执行稀疏反卷积
img_recon = sparse_deconv(im, PSF_FACTOR, sparse_iter=SPARSE_ITER)

# 保存结果，保持原始数据类型
io.imsave(output_path, img_recon.astype(im.dtype))

print(f"✅ Processing complete! Result saved to: {output_path}")