"""
OlmoEarth 最简推理脚本
====================
直接用代码定义模型结构，不依赖任何config解析

使用方法：
    python olmoearth_inference_jp2_simple.py --jp2_dir /path/to/sentinel2_safe

Tensor Shape说明:
    - sentinel2_l2a: [B, H, W, T, C]  其中T=1表示单时间点，C=12是Sentinel-2波段数
    - mask: [B, H, W, T, S]  S=3是band sets数量（与Inference-Quickstart.md一致）
    - 输出features: [B, H', W', T, S, D]  经过pooled=mean(dim=[3,4])变成[B, H', W', D]
"""

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from data.constants import Modality
from data.normalize import Normalizer, Strategy
from datatypes import MaskedOlmoEarthSample


def read_sentinel2_jp2_files(jp2_dir: str, target_size: int = 512) -> np.ndarray | None:
    """读取Sentinel-2 L2A的JP2文件
    
    Returns:
        image: np.ndarray with shape [H, W, num_bands] - Channel-Last格式
    """
    band_order = Modality.SENTINEL2_L2A.band_order
    print(f"需要的波段: {band_order}")
    print(f"波段数量: {len(band_order)}")
    
    fnames = []
    for band_name in band_order:
        pattern = f"{jp2_dir}/*.SAFE/GRANULE/*/IMG_DATA/*/*_{band_name}_*.jp2"
        matches = glob.glob(pattern)
        if matches:
            fnames.append(matches[0])
        else:
            print(f"警告: 波段 {band_name} 未找到")
            fnames.append(None)
    
    if None in fnames:
        print("错误: 部分波段文件缺失！")
        return None
    
    try:
        import rasterio
        from rasterio.vrt import WarpedVRT
        from rasterio.enums import Resampling
        
        with rasterio.open(fnames[0]) as src:
            crs = src.crs
            transform = src.transform
        
        images = []
        for fname in fnames:
            with rasterio.open(fname) as src:
                with rasterio.vrt.WarpedVRT(
                    src, crs=crs, transform=transform,
                    width=target_size, height=target_size,
                    resampling=Resampling.bilinear,
                ) as vrt:
                    images.append(vrt.read(1))
        
        # [H, W, num_bands] - Channel-Last，与datatypes.py一致
        image = np.stack(images, axis=-1)
        print(f"读取完成: {image.shape} (H, W, num_bands)")
        return image
        
    except ImportError:
        print("错误: 请安装 rasterio: pip install rasterio")
        return None


def normalize_image(image: np.ndarray) -> np.ndarray:
    """归一化图像
    
    Args:
        image: [H, W, num_bands] Channel-Last格式
    
    Returns:
        normalized: [H, W, num_bands] 归一化后的数据
    """
    normalizer = Normalizer(strategy=Strategy.COMPUTED)
    modality = Modality.SENTINEL2_L2A
    
    # 添加batch和time维度: [H, W, C] -> [1, H, W, 1, C]
    image_with_batch = image[np.newaxis, :, :, np.newaxis, :]
    normalized = normalizer.normalize(modality, image_with_batch.astype(np.float32))
    
    # 去掉batch和time维度: [1, H, W, 1, C] -> [H, W, C]
    normalized = normalized[0, :, :, 0, :]
    
    print(f"归一化完成: shape={normalized.shape}, min={normalized.min():.3f}, max={normalized.max():.3f}")
    return normalized


def prepare_input(normalized_image: np.ndarray, crop_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """准备模型输入
    
    Args:
        normalized_image: [H, W, num_bands] 归一化后的图像
        crop_size: 裁剪大小
    
    Returns:
        image_tensor: [1, H, W, 1, C] - 与MaskedOlmoEarthSample.sentinel2_l2a格式一致
        mask: [1, H, W, 1, S] - band sets维度是S=3
    """
    h, w = normalized_image.shape[:2]
    
    # 中心裁剪
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    cropped = normalized_image[start_h:start_h + crop_size, start_w:start_w + crop_size, :]
    
    # 转换为tensor并添加batch和time维度: [H, W, C] -> [1, H, W, 1, C]
    image_tensor = torch.tensor(cropped, dtype=torch.float32).unsqueeze(0).unsqueeze(3)
    
    print(f"输入准备完成: {image_tensor.shape} (B=1, H={crop_size}, W={crop_size}, T=1, C={cropped.shape[-1]})")
    
    # 创建mask: [B, H, W, T, S] - S=3是band sets数量
    from datatypes import MaskValue
    B, H, W, T, C = image_tensor.shape
    S = 3  # band sets数量（与Inference-Quickstart.md一致）
    mask = torch.full((B, H, W, T, S), MaskValue.ONLINE_ENCODER.value, dtype=torch.int64)
    
    return image_tensor, mask


def load_model_direct():
    """直接用代码构建模型，不依赖config.json
    
    模型参数来自OlmoEarth-v1-Base:
    - embedding_size: 768
    - depth: 12 (encoder)
    - num_heads: 12
    - mlp_ratio: 4.0
    
    注意: EncoderConfig 中 supported_modalities 是 @property，
    不是字段。它通过 supported_modality_names 自动计算得出。
    因此不要直接传递 supported_modalities 参数。
    """
    from nn.flexi_vit import EncoderConfig
    
    # 定义支持的模态（只需要传递名称列表，Config会自动计算 supported_modalities）
    modalities = [
        "sentinel2_l2a", "sentinel1", "landsat", "worldcover",
        "srtm", "openstreetmap_raster", "wri_canopy_height_map", "cdl", "worldcereal"
    ]
    
    # Encoder配置
    encoder_config = EncoderConfig(
        supported_modality_names=modalities,
        embedding_size=768,
        max_patch_size=8,
        min_patch_size=1,
        num_heads=12,
        mlp_ratio=4.0,
        depth=12,
        drop_path=0.1,
        max_sequence_length=12,
        num_register_tokens=0,
        learnable_channel_embeddings=True,
        random_channel_embeddings=False,
        num_projection_layers=1,
        aggregate_then_project=True,
        use_flash_attn=False,
        frozen_patch_embeddings=False,
        qk_norm=False,
        log_token_norm_stats=False,
        use_linear_patch_embed=False,  # 必须是False以匹配checkpoint（训练时使用Conv2d）
    )
    
    # 只构建 encoder
    encoder = encoder_config.build()
    
    print("模型构建成功（仅 Encoder）")
    return encoder


def visualize_and_save(features: torch.Tensor, input_tensor: torch.Tensor, output_dir: str = "."):
    """使用PCA可视化特征并保存
    
    Args:
        features: [B, H', W', T, S, D] 模型输出特征
        input_tensor: [B, H, W, T, C] 输入图像
        output_dir: 输出目录
    """
    try:
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
    except ImportError:
        print("提示: 需要安装 sklearn 和 matplotlib 来保存可视化结果")
        print("  pip install scikit-learn matplotlib")
        return
    
    # 特征形状: [B, H', W', T, S, D]
    B, H_prime, W_prime, T, S, D = features.shape
    # 输入形状: [B, H, W, T, C]
    _, H, W, _, C = input_tensor.shape
    
    # 对特征进行PCA降维到3维（RGB）
    # 先转到CPU，避免GPU无法运行sklearn
    features_2d = features[0].detach().cpu().numpy().reshape(-1, D)  # [H'*W'*T*S, D]
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features_2d)  # [H'*W'*T*S, 3]
    
    # 重塑为图像格式 [H', W', T, S, 3] -> [H', W', 3]
    features_img = features_pca.reshape(H_prime, W_prime, T, S, 3)
    # 去掉T和S维度，取第一个时间点和第一个band set
    features_img = features_img[:, :, 0, 0, :]  # [H', W', 3]
    
    # 归一化到[0,1]
    features_img = (features_img - features_img.min()) / (features_img.max() - features_img.min() + 1e-8)
    
    # 保存PCA特征图（已经是 [H', W', 3] 格式，直接显示）
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(features_img)  # [H', W', 3]
    plt.title(f"PCA Features (D={D} -> RGB)")
    plt.axis('off')
    
    # 将输入转为可视化格式（RGB: R=B04, G=B03, B=B02）
    input_np = input_tensor[0].cpu().numpy()  # [H, W, T, C]
    if T > 1 or C >= 3:
        # 取RGB波段 (B04=Red, B03=Green, B02=Blue)
        # band_order = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'B01']
        input_rgb = input_np[:, :, 0, 2:5]  # B04, B03, B02 -> R, G, B
        input_rgb = (input_rgb - input_rgb.min()) / (input_rgb.max() - input_rgb.min() + 1e-8)
    else:
        # 单波段可视化
        input_rgb = np.stack([input_np[:, :, 0, 0]] * 3, axis=-1)
        input_rgb = (input_rgb - input_rgb.min()) / (input_rgb.max() - input_rgb.min() + 1e-8)
    
    plt.subplot(1, 2, 2)
    plt.imshow(input_rgb)
    plt.title(f"Input RGB (H={H}, W={W}, C={C})")
    plt.axis('off')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "feature_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化结果已保存: {output_path}")
    
    # 单独保存原图
    input_path = Path(output_dir) / "input_rgb.png"
    plt.figure(figsize=(6, 6))
    plt.imshow(input_rgb)
    plt.title(f"Input RGB")
    plt.axis('off')
    plt.savefig(input_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"原图已保存: {input_path}")


def main():
    parser = argparse.ArgumentParser(description="OlmoEarth Inference")
    parser.add_argument("--jp2_dir", type=str, required=True, help="Sentinel-2 SAFE文件夹路径")
    parser.add_argument("--crop_size", type=int, default=128, help="裁剪大小（必须是patch_size的整数倍）")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch大小")
    parser.add_argument("--target_size", type=int, default=512, help="读取图像尺寸")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).parent), help="输出目录")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("OlmoEarth Inference (最简版)")
    print("=" * 60)
    
    # 1. 读取JP2
    print("\n[1/4] 读取JP2文件...")
    image = read_sentinel2_jp2_files(args.jp2_dir, args.target_size)
    if image is None:
        return
    
    # 2. 归一化
    print("\n[2/4] 归一化...")
    normalized = normalize_image(image)
    
    # 3. 准备输入
    print("\n[3/4] 准备输入...")
    image_tensor, mask = prepare_input(normalized, args.crop_size)
    
    # 4. 构建模型
    print("\n[4/4] 构建模型...")
    model = load_model_direct()
    model = model.to(args.device)
    model.eval()
    
    # 加载权重（如果有权重文件）
    weights_path = Path(__file__).parent / "params" / "weights.pth"
    if weights_path.exists() and weights_path.stat().st_size > 1000:
        print("加载权重文件...")
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # 处理权重前缀
        # 1. 清洗常见的分布式前缀
        # 2. 剥离 encoder. 前缀，因为我们现在 model 本身就是 encoder
        # 3. 丢弃 decoder 和 target_encoder 权重
        new_state_dict = {}
        decoder_count = 0
        for k, v in state_dict.items():
            new_key = k.replace('module.', '').replace('_forward_module.', '').replace('model.module.', '')
            
            # 丢弃 decoder 权重，节约内存
            if new_key.startswith('decoder.'):
                decoder_count += 1
                continue
            if new_key.startswith('target_encoder.'):
                decoder_count += 1
                continue
                
            # 剥离 encoder. 前缀，因为我们现在 model 本身就是 encoder
            if new_key.startswith('encoder.'):
                new_key = new_key[8:]
                
            new_state_dict[new_key] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print(f"权重加载完成 (跳过了 {decoder_count} 个decoder/target_encoder权重)")
    else:
        print("提示: 未找到权重文件，模型使用随机初始化权重")
    
    # 推理
    print("\n运行推理...")
    print(f"  patch_size: {args.patch_size}")
    print(f"  crop_size: {args.crop_size}")
    print(f"  预期patch数量: {args.crop_size // args.patch_size} x {args.crop_size // args.patch_size}")
    
    with torch.no_grad():
        # timestamps格式: [B, T, D] 其中D=[day, month, year]，月份是0-indexed
        sample = MaskedOlmoEarthSample(
            timestamps=torch.tensor([[[22, 7, 2025]]], dtype=torch.int64).to(args.device),
            sentinel2_l2a=image_tensor.to(args.device),
            sentinel2_l2a_mask=mask.to(args.device),
        )
        
        # 直接调用 encoder（不再是 model.encoder）
        output = model(sample, fast_pass=True, patch_size=args.patch_size)
        tokens_and_masks = output["tokens_and_masks"]
        # 输出: [B, H', W', T, S, D] 与 Inference-Quickstart.md 一致
        features = tokens_and_masks.sentinel2_l2a
        
        # pooled: mean(dim=[3, 4]) 去掉 T 和 S 维度 -> [B, H', W', D]
        pooled = features.mean(dim=[3, 4])
    
    print("\n" + "=" * 60)
    print("结果:")
    print(f"  Input shape: {image_tensor.shape} (B, H, W, T, C)")
    print(f"  Output feature shape: {features.shape} (B, H', W', T, S, D)")
    print(f"  Pooled shape: {pooled.shape} (B, H', W', D)")
    print(f"  Pooled (first 5): {pooled[0, 0, 0, :5].cpu().numpy()}")
    print("=" * 60)
    
    # 可视化特征
    visualize_and_save(features, image_tensor, args.output_dir)


if __name__ == "__main__":
    main()
