import os
from PIL import Image
import glob

category = '10'

# 创建必要的目录
dirs = ['reference_images', 'reference_masks', 'target_images', 'target_masks']
for dir_name in dirs:
    os.makedirs(dir_name, exist_ok=True)

# 获取所有图像文件
image_files = sorted(glob.glob('images/*.jpg'))
mask_files = sorted(glob.glob('masks/*.jpg'))

print(f"共有{len(image_files)}张图像和{len(mask_files)}张掩膜。")

# 处理第一张图片作为参考图像
ref_img = Image.open(image_files[0])
ref_mask = Image.open(mask_files[0])

# 保存参考图像
ref_img.convert('RGB').save(os.path.join('reference_images', 'Kvasir_0000001.jpg'))
ref_mask.convert('RGB').save(os.path.join('reference_masks', 'Kvasir_0000001.jpg'))

# 处理接下来的10张图片作为目标图像
for i, (img_path, mask_path) in enumerate(zip(image_files[1:11], mask_files[1:11]), start=2):
    # 读取图像
    img = Image.open(img_path)
    mask = Image.open(mask_path)
    
    # 生成新的文件名
    new_name = f'Kvasir_{category}_{i:07d}.jpg'
    
    # 保存图像
    img.convert('RGB').save(os.path.join('target_images', new_name))
    mask.convert('RGB').save(os.path.join('target_masks', new_name))

print("ok！")
