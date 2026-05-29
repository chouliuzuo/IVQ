"""
评分函数模块
提供FID、IS、PSNR、SSIM等图像质量评估指标
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from scipy import linalg
from scipy import signal
from PIL import Image
from typing import List, Union
import warnings

def _get_inception_for_fid():
    m = inception_v3(pretrained=True, transform_input=False)
    m.fc = torch.nn.Identity()
    m.eval()
    return m

def _get_inception_for_is():
    m = inception_v3(pretrained=True, transform_input=False)
    m.eval()
    return m

def _to_numpy_array(img: Union[np.ndarray, Image.Image, torch.Tensor]) -> np.ndarray:
    """
    将各种格式的图像转换为numpy数组
    
    Args:
        img: 可以是numpy数组、PIL Image或torch Tensor
        
    Returns:
        numpy数组，形状为(H, W, C)，值范围0-255，dtype为uint8
    """
    if isinstance(img, Image.Image):
        img = np.array(img)
    elif isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        # 如果是(C, H, W)格式，转换为(H, W, C)
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        # 如果是归一化的[0,1]范围，转换为[0,255]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    elif isinstance(img, np.ndarray):
        # 确保是uint8格式
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
    else:
        raise TypeError(f"不支持的图像类型: {type(img)}")
    
    # 确保是(H, W, C)格式
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    if img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    
    return img


def _to_tensor(img: np.ndarray, device='cpu') -> torch.Tensor:
    """
    将numpy数组转换为torch tensor，用于Inception网络
    
    Args:
        img: numpy数组，形状(H, W, C)，值范围0-255
        device: 设备
        
    Returns:
        torch tensor，形状(1, C, H, W)，值范围0-1
    """
    # 转换为(C, H, W)格式
    if img.ndim == 3:
        img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float() / 255.0
    img = img.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
    
    # 调整大小到299x299（Inception v3的输入尺寸）
    img = F.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
    
    # 归一化到[-1, 1]（Inception v3的输入要求）
    img = (img - 0.5) * 2.0
    
    return img.to(device)


def calculate_fid(reconstruct_image_list: List[Union[np.ndarray, Image.Image, torch.Tensor]],
                  original_image_list: List[Union[np.ndarray, Image.Image, torch.Tensor]],
                  device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                  batch_size: int = 50) -> float:
    """
    计算Fréchet Inception Distance (FID)
    
    FID通过比较真实图像和生成图像在Inception网络特征空间中的分布来计算。
    值越小表示图像质量越好。
    
    Args:
        reconstruct_image_list: 重建图像列表
        original_image_list: 原始图像列表
        device: 计算设备
        batch_size: 批处理大小
        
    Returns:
        FID分数（float）
    """
    assert len(reconstruct_image_list) == len(original_image_list), \
        "两个图像列表必须等长"
    
    # 加载Inception模型
    model = _get_inception_for_fid().to(device)
    model.eval()
    
    def extract_features(image_list):
        """提取图像特征"""
        features = []
        model.eval()
        
        with torch.no_grad():
            for i in range(0, len(image_list), batch_size):
                batch_images = image_list[i:i+batch_size]
                batch_tensors = []
                
                for img in batch_images:
                    img_np = _to_numpy_array(img)
                    img_tensor = _to_tensor(img_np, device)
                    batch_tensors.append(img_tensor)
                
                if batch_tensors:
                    batch = torch.cat(batch_tensors, dim=0)
                    feat = model(batch)
                    features.append(feat.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    # 提取特征
    print("提取原始图像特征...")
    real_features = extract_features(original_image_list)
    print("提取重建图像特征...")
    fake_features = extract_features(reconstruct_image_list)
    
    # 计算FID
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # 计算Fréchet距离
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    # 处理数值误差
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    return float(fid)


def calculate_is(reconstruct_image_list: List[Union[np.ndarray, Image.Image, torch.Tensor]],
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 batch_size: int = 50,
                 splits: int = 10) -> float:
    """
    计算Inception Score (IS)
    
    IS评估生成图像的质量和多样性。值越大表示图像质量越好。
    
    Args:
        reconstruct_image_list: 重建图像列表
        device: 计算设备
        batch_size: 批处理大小
        splits: 用于计算IS的分割数量
        
    Returns:
        IS分数（float）
    """
    # 加载Inception模型（需要完整的分类器）
    model = _get_inception_for_is().to(device)
    model.eval()
    
    def get_predictions(image_list):
        """获取图像预测概率"""
        preds = []
        
        with torch.no_grad():
            for i in range(0, len(image_list), batch_size):
                batch_images = image_list[i:i+batch_size]
                batch_tensors = []
                
                for img in batch_images:
                    img_np = _to_numpy_array(img)
                    img_tensor = _to_tensor(img_np, device)
                    batch_tensors.append(img_tensor)
                
                if batch_tensors:
                    batch = torch.cat(batch_tensors, dim=0)
                    # 获取logits
                    logits = model(batch)
                    # 转换为概率
                    probs = F.softmax(logits, dim=1)
                    preds.append(probs.cpu().numpy())
        
        return np.concatenate(preds, axis=0)
    
    print("计算Inception Score...")
    preds = get_predictions(reconstruct_image_list)
    
    # 计算IS
    scores = []
    for i in range(splits):
        part = preds[i * (len(preds) // splits): (i + 1) * (len(preds) // splits)]
        py = np.mean(part, axis=0)
        scores.append(np.exp(np.mean([np.sum(p * np.log(p / py)) for p in part])))
    
    return float(np.mean(scores))


def calculate_psnr(reconstruct_image_list: List[Union[np.ndarray, Image.Image, torch.Tensor]],
                   original_image_list: List[Union[np.ndarray, Image.Image, torch.Tensor]]) -> float:
    """
    计算Peak Signal-to-Noise Ratio (PSNR)
    
    PSNR评估图像重建质量。值越大表示重建质量越好。
    通常PSNR > 30dB表示较好的重建质量。
    
    Args:
        reconstruct_image_list: 重建图像列表
        original_image_list: 原始图像列表
        
    Returns:
        平均PSNR分数（float，单位：dB）
    """
    assert len(reconstruct_image_list) == len(original_image_list), \
        "两个图像列表必须等长"
    
    psnr_values = []
    
    for recon_img, orig_img in zip(reconstruct_image_list, original_image_list):
        # 转换为numpy数组
        recon_np = _to_numpy_array(recon_img).astype(np.float64)
        orig_np = _to_numpy_array(orig_img).astype(np.float64)
        
        # 确保尺寸一致
        if recon_np.shape != orig_np.shape:
            # 调整重建图像尺寸以匹配原始图像
            from PIL import Image
            recon_pil = Image.fromarray(recon_np.astype(np.uint8))
            orig_pil = Image.fromarray(orig_np.astype(np.uint8))
            recon_pil = recon_pil.resize(orig_pil.size, Image.BICUBIC)
            recon_np = np.array(recon_pil).astype(np.float64)
        
        # 计算MSE
        mse = np.mean((recon_np - orig_np) ** 2)
        
        # 避免除零
        if mse == 0:
            psnr = 100.0  # 完美匹配
        else:
            # 假设像素值范围是0-255
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        psnr_values.append(psnr)
    
    return float(np.mean(psnr_values))


def calculate_ssim(reconstruct_image_list: List[Union[np.ndarray, Image.Image, torch.Tensor]],
                   original_image_list: List[Union[np.ndarray, Image.Image, torch.Tensor]],
                   window_size: int = 11) -> float:
    """
    计算Structural Similarity Index (SSIM)
    
    SSIM评估两幅图像的结构相似性。值范围在[-1, 1]之间，1表示完全相同。
    通常SSIM > 0.9表示较好的重建质量。
    
    Args:
        reconstruct_image_list: 重建图像列表
        original_image_list: 原始图像列表
        window_size: SSIM计算窗口大小
        
    Returns:
        平均SSIM分数（float）
    """
    assert len(reconstruct_image_list) == len(original_image_list), \
        "两个图像列表必须等长"
    
    def ssim_single(img1, img2, window_size=11):
        """计算单对图像的SSIM"""
        # 转换为灰度图像（如果是RGB）
        if img1.ndim == 3 and img1.shape[2] == 3:
            # 使用标准RGB到灰度的权重
            img1 = 0.299 * img1[:, :, 0] + 0.587 * img1[:, :, 1] + 0.114 * img1[:, :, 2]
        if img2.ndim == 3 and img2.shape[2] == 3:
            img2 = 0.299 * img2[:, :, 0] + 0.587 * img2[:, :, 1] + 0.114 * img2[:, :, 2]
        
        # 确保尺寸一致
        if img1.shape != img2.shape:
            from PIL import Image
            img1_pil = Image.fromarray(img1.astype(np.uint8))
            img2_pil = Image.fromarray(img2.astype(np.uint8))
            img1_pil = img1_pil.resize(img2_pil.size, Image.BICUBIC)
            img1 = np.array(img1_pil).astype(np.float64)
        
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        # SSIM参数
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # 创建高斯窗口
        def gaussian_window(size, sigma=1.5):
            gauss = np.array([np.exp(-(x - size//2)**2 / (2*sigma**2)) for x in range(size)])
            return gauss / gauss.sum()
        
        window = np.outer(gaussian_window(window_size), gaussian_window(window_size))
        window = window / window.sum()
        
        # 使用2D卷积计算局部统计
        mu1 = signal.convolve2d(img1, window, mode='valid')
        mu2 = signal.convolve2d(img2, window, mode='valid')
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # 计算方差和协方差
        sigma1_sq = signal.convolve2d(img1 ** 2, window, mode='valid') - mu1_sq
        sigma2_sq = signal.convolve2d(img2 ** 2, window, mode='valid') - mu2_sq
        sigma12 = signal.convolve2d(img1 * img2, window, mode='valid') - mu1_mu2
        
        # 计算SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return np.mean(ssim_map)
    
    ssim_values = []
    
    for recon_img, orig_img in zip(reconstruct_image_list, original_image_list):
        # 转换为numpy数组
        recon_np = _to_numpy_array(recon_img)
        orig_np = _to_numpy_array(orig_img)
        
        ssim = ssim_single(recon_np, orig_np, window_size)
        ssim_values.append(ssim)
    
    return float(np.mean(ssim_values))


# 便捷函数：计算所有指标
def calculate_all_scores(reconstruct_image_list: List[Union[np.ndarray, Image.Image, torch.Tensor]],
                         original_image_list: List[Union[np.ndarray, Image.Image, torch.Tensor]],
                         device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                         batch_size: int = 50) -> dict:
    """
    计算所有评分指标
    
    Args:
        reconstruct_image_list: 重建图像列表
        original_image_list: 原始图像列表
        device: 计算设备
        batch_size: 批处理大小
        
    Returns:
        包含所有评分的字典
    """
    results = {}
    
    print("计算PSNR...")
    results['PSNR'] = calculate_psnr(reconstruct_image_list, original_image_list)
    
    print("计算SSIM...")
    results['SSIM'] = calculate_ssim(reconstruct_image_list, original_image_list)
    
    print("计算FID...")
    results['FID'] = calculate_fid(reconstruct_image_list, original_image_list, device, batch_size)
    
    print("计算IS...")
    results['IS'] = calculate_is(reconstruct_image_list, device, batch_size)
    
    return results
