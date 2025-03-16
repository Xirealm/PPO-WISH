import os
import shutil
from PIL import Image

def to_camel_case(snake_str):
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

# 创建所需文件夹
folders = ['reference_images', 'reference_masks', 'target_images', 'target_masks']
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# 获取当前目录名称和文件
current_dir = os.path.basename(os.getcwd())
current_dir = to_camel_case(current_dir)  # 转换为小驼峰命名
jpg_files = [f for f in os.listdir() if f.endswith('.jpg')]
png_files = [f for f in os.listdir() if f.endswith('.png')]

# 处理第一张图片作为reference
if jpg_files:
    # 复制第一张jpg到reference_images
    first_jpg = jpg_files[0]
    reference_jpg_name = f'FSS-1000_{current_dir}_{first_jpg}'
    shutil.copy(first_jpg, os.path.join('reference_images', reference_jpg_name))
    
    # 找到对应的png文件并处理为reference mask
    first_png = first_jpg.replace('.jpg', '.png')
    if first_png in png_files:
        # 将PNG转换为JPG并保存到reference_masks
        img = Image.open(first_png)
        reference_mask_name = f'FSS-1000_{current_dir}_{first_png.replace(".png", ".jpg")}'
        img.convert('RGB').save(os.path.join('reference_masks', reference_mask_name))

# 处理所有图片作为target
for jpg_file in jpg_files:
    target_name = f'FSS-1000_{current_dir}_{jpg_file}'
    shutil.copy(jpg_file, os.path.join('target_images', target_name))

# 处理所有PNG掩码
for png_file in png_files:
    img = Image.open(png_file)
    target_mask_name = f'FSS-1000_{current_dir}_{png_file.replace(".png", ".jpg")}'
    img.convert('RGB').save(os.path.join('target_masks', target_mask_name))

print("OK！")
