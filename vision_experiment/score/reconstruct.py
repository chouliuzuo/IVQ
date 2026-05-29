"""批量图片重建脚本 - 使用 TiTok-S-128 模型

参考 TiTok 实现，加载模型、批量重建图片并保存结果。
"""

import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from modeling.titok import TiTok
import argparse
from pathlib import Path

# ========== 配置参数 ==========
# 默认权重路径

# original weight path
# DEFAULT_WEIGHT_PATH = "/workspace/lfj/zhd_ivq/1d-tokenizer/weight" 

# trained weight path
DEFAULT_WEIGHT_PATH = "/models/zhd/titok_b64_stage2_run1_50w/checkpoint-350000/"
# =============================

# 支持的图像格式
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

def get_image_files(input_dir):
    """获取输入目录中的所有图像文件，按文件名排序"""
    image_files = []
    for file_path in Path(input_dir).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
            image_files.append(file_path)
    # 按文件名排序，确保顺序一致
    image_files.sort(key=lambda x: x.name)
    return image_files

def reconstruct_image(model, image_path, transform, device):
    """重建单张图像"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 编码图片
    z_quantized, result_dict = model.encode(image_tensor)
    
    # 解码重建图片
    with torch.no_grad():
        reconstructed_image = model.decode(z_quantized)
        reconstructed_image = torch.clamp(reconstructed_image, 0.0, 1.0)
        # 转换为 numpy array: [1, C, H, W] -> [H, W, C]
        reconstructed_image = (reconstructed_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
        reconstructed_image = Image.fromarray(reconstructed_image)
    
    return reconstructed_image

def main():
    parser = argparse.ArgumentParser(description='批量重建图像 - 使用 TiTok-S-128 模型')
    parser.add_argument('--input_dir', type=str, required=True, help='输入图像文件夹路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出图像文件夹路径')
    parser.add_argument('--weight', type=str, default=DEFAULT_WEIGHT_PATH, help='模型权重路径')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载 TiTok-S-128 tokenizer
    print(f"加载模型权重: {args.weight}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    titok_tokenizer = TiTok.from_pretrained(args.weight)
    titok_tokenizer.eval()
    titok_tokenizer.requires_grad_(False)
    titok_tokenizer = titok_tokenizer.to(device)
    print(f"使用设备: {device}")
    
    # 准备图像变换（与训练时保持一致）
    # 使用与训练时评估阶段相同的预处理流程：
    # 1. Resize 到 256x256（直接resize到固定尺寸，使用BICUBIC插值）
    # 2. CenterCrop 到 256x256
    # 3. ToTensor（自动归一化到[0,1]）
    # 4. Normalize（默认mean=[0,0,0], std=[1,1,1]，实际不改变值）
    crop_size = 256
    transform = transforms.Compose([
        transforms.Resize(crop_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])  # 实际不改变值，保持[0,1]
    ])
    
    # 获取所有图像文件
    input_dir = Path(args.input_dir)
    image_files = get_image_files(input_dir)
    
    if len(image_files) == 0:
        print(f"错误: 在 {args.input_dir} 中没有找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 张图像，开始处理...")
    
    # 批量处理图像
    for idx, image_path in enumerate(image_files, 1):
        try:
            print(f"处理 [{idx}/{len(image_files)}]: {image_path.name}")
            
            # 重建图像
            reconstructed_image = reconstruct_image(titok_tokenizer, image_path, transform, device)
            
            # 保存结果，保持原文件名
            output_path = output_dir / image_path.name
            reconstructed_image.save(output_path)
            
        except Exception as e:
            print(f"处理 {image_path.name} 时出错: {str(e)}")
            continue
    
    print(f"\n重建完成！所有结果已保存至: {output_dir}")
    print(f"共处理 {len(image_files)} 张图像")

if __name__ == "__main__":
    main()