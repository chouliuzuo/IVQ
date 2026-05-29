#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1D-Tokenizer (TiTok) 图像评分脚本
对原始图像和重建图像进行评分，计算FID、IS、PSNR、SSIM四个指标
"""

import os
import sys
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import argparse
from pathlib import Path

from score_functions import calculate_all_scores

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


def preprocess_image_for_scoring(image_path, image_size=256):
    """
    对原始图像进行预处理（与reconstruct.py中一致）
    返回预处理后的PIL Image（用于评分）
    
    注意：与reconstruct.py中的预处理保持一致：
    - Resize 到 256x256（使用BICUBIC插值，antialias=True）
    - CenterCrop 到 256x256
    但不进行ToTensor和Normalize，因为评分函数需要原始像素值
    """
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 预处理变换（与reconstruct.py中保持一致，但不做ToTensor和Normalize）
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(image_size),
    ])
    
    # 应用变换（只做resize和crop，不进行归一化，因为评分函数需要原始像素值）
    preprocessed_image = transform(image)
    
    return preprocessed_image


def load_reconstructed_image(image_path):
    """
    加载重建图像（不需要预处理）
    """
    image = Image.open(image_path).convert('RGB')
    return image


def main():
    parser = argparse.ArgumentParser(description='计算原始图像和重建图像的评分指标')
    parser.add_argument('--original_dir', type=str, required=True, 
                        help='原始图像文件夹路径')
    parser.add_argument('--reconstruct_dir', type=str, required=True, 
                        help='重建图像文件夹路径')
    parser.add_argument('--image_size', type=int, default=256, 
                        help='图像尺寸（用于原始图像预处理）')
    parser.add_argument('--device', type=str, default=None,
                        help='计算设备（cuda/cpu），默认自动选择')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='批处理大小（用于FID和IS计算）')
    
    args = parser.parse_args()
    
    # 确定设备
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("1D-Tokenizer (TiTok) 图像评分")
    print("=" * 60)
    print(f"原始图像目录: {args.original_dir}")
    print(f"重建图像目录: {args.reconstruct_dir}")
    print(f"图像尺寸: {args.image_size}")
    print(f"计算设备: {device}")
    print(f"批处理大小: {args.batch_size}")
    print("=" * 60)
    
    # 1. 获取所有图像文件
    original_dir = Path(args.original_dir)
    reconstruct_dir = Path(args.reconstruct_dir)
    
    original_files = get_image_files(original_dir)
    reconstructed_files = get_image_files(reconstruct_dir)
    
    if len(original_files) == 0:
        print(f"错误: 在 {args.original_dir} 中没有找到图像文件")
        return
    
    if len(reconstructed_files) == 0:
        print(f"错误: 在 {args.reconstruct_dir} 中没有找到图像文件")
        return
    
    if len(original_files) != len(reconstructed_files):
        print(f"警告: 原始图像数量 ({len(original_files)}) 与重建图像数量 ({len(reconstructed_files)}) 不一致")
        print("将只处理文件名匹配的图像对")
    
    # 2. 匹配图像文件（按文件名）
    original_dict = {f.name: f for f in original_files}
    reconstructed_dict = {f.name: f for f in reconstructed_files}
    
    # 找到共同的文件名
    common_names = set(original_dict.keys()) & set(reconstructed_dict.keys())
    if len(common_names) == 0:
        print("错误: 没有找到文件名匹配的图像对")
        return
    
    # 按文件名排序
    common_names = sorted(common_names)
    
    print(f"\n找到 {len(common_names)} 对匹配的图像")
    
    # 3. 加载并预处理图像
    print("\n加载原始图像并进行预处理...")
    original_images = []
    for name in common_names:
        try:
            original_path = original_dict[name]
            preprocessed_img = preprocess_image_for_scoring(original_path, args.image_size)
            original_images.append(preprocessed_img)
        except Exception as e:
            print(f"处理原始图像 {name} 时出错: {str(e)}")
            continue
    
    print(f"成功加载 {len(original_images)} 张原始图像")
    
    print("\n加载重建图像...")
    reconstructed_images = []
    for name in common_names:
        try:
            reconstructed_path = reconstructed_dict[name]
            recon_img = load_reconstructed_image(reconstructed_path)
            reconstructed_images.append(recon_img)
        except Exception as e:
            print(f"处理重建图像 {name} 时出错: {str(e)}")
            continue
    
    print(f"成功加载 {len(reconstructed_images)} 张重建图像")
    
    # 确保两个列表长度一致
    min_len = min(len(original_images), len(reconstructed_images))
    original_images = original_images[:min_len]
    reconstructed_images = reconstructed_images[:min_len]
    
    if len(original_images) == 0:
        print("错误: 没有成功加载任何图像对")
        return
    
    print(f"\n开始计算评分指标（共 {len(original_images)} 对图像）...")
    
    # 4. 计算评分指标
    try:
        results = calculate_all_scores(
            reconstruct_image_list=reconstructed_images,
            original_image_list=original_images,
            device=device,
            batch_size=args.batch_size
        )
        
        # 5. 输出结果
        print("\n" + "=" * 60)
        print("评分结果")
        print("=" * 60)
        print(f"PSNR: {results['PSNR']:.4f} dB")
        print(f"SSIM: {results['SSIM']:.4f}")
        print(f"FID:  {results['FID']:.4f}")
        print(f"IS:   {results['IS']:.4f}")
        print("=" * 60)
        
        # 保存结果到文件
        output_file = Path(args.reconstruct_dir) / "score_results.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("1D-Tokenizer (TiTok) 图像评分结果\n")
            f.write("=" * 60 + "\n")
            f.write(f"原始图像目录: {args.original_dir}\n")
            f.write(f"重建图像目录: {args.reconstruct_dir}\n")
            f.write(f"图像数量: {len(original_images)}\n")
            f.write(f"图像尺寸: {args.image_size}\n")
            f.write("=" * 60 + "\n")
            f.write(f"PSNR: {results['PSNR']:.4f} dB\n")
            f.write(f"SSIM: {results['SSIM']:.4f}\n")
            f.write(f"FID:  {results['FID']:.4f}\n")
            f.write(f"IS:   {results['IS']:.4f}\n")
            f.write("=" * 60 + "\n")
        
        print(f"\n结果已保存至: {output_file}")
        
    except Exception as e:
        print(f"\n计算评分时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
