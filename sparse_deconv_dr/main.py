import os
from sparse_recon.sparse_deconv import sparse_deconv
from skimage import io
import glob

# ✅ 直接使用字符串绝对路径（请根据你的实际路径修改！）
DATASET_ROOT = "C:\\sim\\biosr_dataset\\BioSR\\Microtubules"  # ←←← Windows 示例
 
TEST_WF_DIR = os.path.join(DATASET_ROOT, "test_wf")
OUTPUT_ROOT = os.path.join(DATASET_ROOT, "test_recon")

PSF_FACTOR =  5  # present best 5
# PSF_FACTOR = 2
def process_image(input_path, output_path):
    im = io.imread(input_path)
    img_recon = sparse_deconv(im, PSF_FACTOR,sparse_iter=40000)
    io.imsave(output_path, img_recon.astype(im.dtype))

def main():
    print("Starting sparse deconvolution...")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    level_dirs = [d for d in os.listdir(TEST_WF_DIR) if d.startswith("level_")]
    # 👇 关键修改：按数字倒序
    level_dirs.sort(key=lambda x: int(x.split('_')[1]), reverse=True)

    for level in level_dirs:
        level_input_dir = os.path.join(TEST_WF_DIR, level)
        level_output_dir = os.path.join(OUTPUT_ROOT, level)
        os.makedirs(level_output_dir, exist_ok=True)

        print(f"Processing {level}...")

        tif_files = glob.glob(os.path.join(level_input_dir, "*.tif"))
        tif_files.sort()

        for tif_file in tif_files:
            filename = os.path.basename(tif_file)
            output_file = os.path.join(level_output_dir, filename)
            process_image(tif_file, output_file)

    print("✅ All done! Results saved to:", OUTPUT_ROOT)

if __name__ == '__main__':
    main()