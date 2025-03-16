import os
from PIL import Image
import glob

category = '100'

# 创建必要的目录
dirs = ['reference_images', 'reference_masks', 'target_images', 'target_masks']
for dir_name in dirs:
    os.makedirs(dir_name, exist_ok=True)

# 获取所有图像文件
image_files = sorted(glob.glob('image/*.jpg'))
mask_files = sorted(glob.glob('gt/*.png'))

# 处理第一张图片作为参考图像
ref_img = Image.open(image_files[0])
ref_mask = Image.open(mask_files[0])

# 保存参考图像
ref_img.convert('RGB').save(os.path.join('reference_images', 'ISIC_0000001.jpg'))
ref_mask.convert('RGB').save(os.path.join('reference_masks', 'ISIC_0000001.jpg'))

# 处理接下来的100张图片作为目标图像
for i, (img_path, mask_path) in enumerate(zip(image_files[1:101], mask_files[1:101]), start=2):
    # 读取图像
    img = Image.open(img_path)
    mask = Image.open(mask_path)
    
    # 生成新的文件名
    new_name = f'ISIC_{category}_{i:07d}.jpg'
    
    # 保存图像
    img.convert('RGB').save(os.path.join('target_images', new_name))
    mask.convert('RGB').save(os.path.join('target_masks', new_name))

print("ok！")
