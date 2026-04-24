"""
OlmoEarth 最简推理脚本
====================
直接用代码定义模型结构，不依赖任何config解析

使用方法：
    python olmoearth_inference_jp2_simple.py --jp2_dir /path/to/sentinel2_safe

Tensor Shape说明:
    - sentinel2_l2a: [B, H, W, T, num_bands]  其中T=1表示单时间点（Channel-Last格式）
    - mask不是必须的，推理时可以传None
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


def prepare_input(normalized_image: np.ndarray, crop_size: int = 128) -> torch.Tensor:
    """准备模型输入
    
    Args:
        normalized_image: [H, W, num_bands] 归一化后的图像
        crop_size: 裁剪大小
    
    Returns:
        image_tensor: [1, H, W, 1, num_bands] - 与MaskedOlmoEarthSample.sentinel2_l2a格式一致
    """
    h, w = normalized_image.shape[:2]
    
    # 中心裁剪
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    cropped = normalized_image[start_h:start_h + crop_size, start_w:start_w + crop_size, :]
    
    # 转换为tensor并添加batch和time维度: [H, W, C] -> [1, H, W, 1, C]
    image_tensor = torch.tensor(cropped, dtype=torch.float32).unsqueeze(0).unsqueeze(3)
    
    print(f"输入准备完成: {image_tensor.shape} (B=1, H={crop_size}, W={crop_size}, T=1, C={cropped.shape[-1]})")
    return image_tensor


def load_model_direct():
    """直接用代码构建模型，不依赖config.json
    
    模型参数来自OlmoEarth-v1-Base:
    - embedding_size: 768
    - depth: 12 (encoder) / 4 (decoder)
    - num_heads: 12
    - mlp_ratio: 4.0
    
    注意: EncoderConfig 和 PredictorConfig 中 supported_modalities 是 @property，
    不是字段。它们通过 supported_modality_names 自动计算得出。
    因此不要直接传递 supported_modalities 参数。
    """
    from nn.galileo import Galileo
    from nn.flexi_vit import EncoderConfig, PredictorConfig
    
    # 定义支持的模态（只需要传递名称列表，Config会自动计算 supported_modalities）
    modalities = [
        "sentinel2_l2a", "sentinel1", "landsat", "worldcover",
        "srtm", "openstreetmap_raster", "wri_canopy_height_map", "cdl", "worldcereal"
    ]
    
    # Encoder配置
    # 注意: 只传递 supported_modality_names，不要传递 supported_modalities
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
    
    # Decoder配置
    # 注意: 只传递 supported_modality_names，不要传递 supported_modalities
    decoder_config = PredictorConfig(
        supported_modality_names=modalities,
        encoder_embedding_size=768,
        decoder_embedding_size=768,
        depth=4,
        mlp_ratio=4.0,
        num_heads=12,
        max_sequence_length=12,
        drop_path=0.0,
        learnable_channel_embeddings=True,
        random_channel_embeddings=False,
        use_flash_attn=False,
        qk_norm=False,
    )
    
    # 构建模型
    encoder = encoder_config.build()
    decoder = decoder_config.build()
    model = Galileo(encoder=encoder, decoder=decoder, reconstructor=None)
    
    print("模型构建成功")
    return model


def main():
    parser = argparse.ArgumentParser(description="OlmoEarth Inference")
    parser.add_argument("--jp2_dir", type=str, required=True, help="Sentinel-2 SAFE文件夹路径")
    parser.add_argument("--crop_size", type=int, default=128, help="裁剪大小（必须是patch_size的整数倍）")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch大小")
    parser.add_argument("--target_size", type=int, default=512, help="读取图像尺寸")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    
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
    
    # 3. 准备输入（不使用mask，只传数据）
    print("\n[3/4] 准备输入...")
    image_tensor = prepare_input(normalized, args.crop_size)
    
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
        
        # 处理分布式训练带来的前缀问题
        # 常见前缀：module., _forward_module., model.module. 等
        new_state_dict = {}
        prefix_count = 0
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_key = k[7:]  # 去掉 "module." 前缀
                prefix_count += 1
            elif k.startswith('_forward_module.'):
                new_key = k[16:]  # 去掉 "_forward_module." 前缀
                prefix_count += 1
            elif k.startswith('model.module.'):
                new_key = k[13:]  # 去掉 "model.module." 前缀
                prefix_count += 1
            else:
                new_key = k
            new_state_dict[new_key] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print(f"权重加载完成 (处理了 {prefix_count} 个带前缀的键)")
    else:
        print("提示: 未找到权重文件，模型使用随机初始化权重")
    
    # 推理（不使用mask）
    print("\n运行推理...")
    print(f"  patch_size: {args.patch_size}")
    print(f"  crop_size: {args.crop_size}")
    print(f"  预期patch数量: {args.crop_size // args.patch_size} x {args.crop_size // args.patch_size}")
    
    with torch.no_grad():
        # mask设为None，encoder内部会创建默认mask
        # timestamps格式: [B, T, D] 其中D=[day, month, year]，月份是0-indexed
        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=image_tensor.to(args.device),
            sentinel2_l2a_mask=None,  # 不使用mask
            timestamps=torch.tensor([[[22, 7, 2025]]], dtype=torch.int64).to(args.device),  # 整数类型，与官方一致
        )
        
        output = model.encoder(sample, fast_pass=True, patch_size=args.patch_size)
        tokens_and_masks = output["tokens_and_masks"]
        features = tokens_and_masks.sentinel2_l2a  # [B, P_H, P_W, T, D]
        pooled = features.mean(dim=[1, 2, 3])  # [B, D]
    
    print("\n" + "=" * 60)
    print("结果:")
    print(f"  Input shape: {image_tensor.shape}")
    print(f"  Output feature shape: {features.shape}")
    print(f"  Pooled shape: {pooled.shape}")
    print(f"  Pooled (first 5): {pooled[0, :5].cpu().numpy()}")
    print("=" * 60)


if __name__ == "__main__":
    main()