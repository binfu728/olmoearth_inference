"""
OlmoEarth JP2图像读取与推理脚本
================================
处理Sentinel-2 L2A的.jp2文件

目录结构：
    olmoearth_inference/
    ├── olmoearth_inference_jp2.py   # 本脚本（推理入口）
    ├── config.py                     # 配置类
    ├── datatypes.py                  # 数据结构定义
    ├── types.py                      # 类型别名
    ├── nn/                          # 神经网络模块
    │   ├── galileo.py               # Galileo模型
    │   ├── flexi_vit.py             # Encoder/Decoder
    │   ├── attention.py              # 注意力机制
    │   ├── flexi_patch_embed.py      # Patch嵌入
    │   ├── encodings.py              # 位置编码
    │   ├── pooling.py                # 池化
    │   ├── tokenization.py           # 分词配置
    │   └── utils.py                  # 工具函数
    ├── data/                        # 数据处理
    │   ├── constants.py              # 模态定义
    │   ├── normalize.py              # 归一化
    │   └── norm_configs/             # 归一化参数
    └── params/                       # 模型权重
        ├── config.json               # 模型配置
        └── weights.pth               # 权重文件

使用方法：
    python olmoearth_inference_jp2.py --jp2_dir /path/to/sentinel2_safe
    
参数说明：
    --jp2_dir: Sentinel-2 SAFE文件夹路径
    --image_size: 输入图像大小（默认64）
    --patch_size: Patch大小（默认4）
    --target_size: 读取图像尺寸（默认512）
    --device: 运行设备（默认cuda）
"""

import argparse
import glob
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from data.constants import Modality
from data.normalize import Normalizer, Strategy


def read_sentinel2_jp2_files(jp2_dir: str, target_size: int = 512) -> np.ndarray:
    """读取Sentinel-2 L2A的JP2文件"""
    band_order = Modality.SENTINEL2_L2A.band_order
    print(f"需要的波段: {band_order}")
    
    # 查找JP2文件
    fnames = []
    for band_name in band_order:
        pattern = f"{jp2_dir}/*.SAFE/GRANULE/*/IMG_DATA/*/*_{band_name}_*.jp2"
        matches = glob.glob(pattern)
        if matches:
            fnames.append(matches[0])
        else:
            fnames.append(None)
    
    if None in fnames:
        print("错误: 部分波段文件缺失！")
        return None
    
    # 使用rasterio读取
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
        
        image = np.stack(images, axis=0).transpose(1, 2, 0)
        print(f"读取完成: {image.shape}")
        return image
        
    except ImportError:
        print("错误: 请安装 rasterio")
        return None


def normalize_image(image: np.ndarray) -> np.ndarray:
    """归一化图像"""
    normalizer = Normalizer(strategy=Strategy.COMPUTED)
    modality = Modality.SENTINEL2_L2A
    
    image_with_batch = image[np.newaxis, :, :, np.newaxis, :]
    normalized = normalizer.normalize(modality, image_with_batch.astype(np.float32))
    normalized = normalized[0, :, :, 0, :]
    
    print(f"归一化完成: min={normalized.min():.3f}, max={normalized.max():.3f}")
    return normalized


def prepare_input(normalized_image: np.ndarray, image_size: int = 64) -> tuple:
    """准备模型输入"""
    from datatypes import MaskedOlmoEarthSample, MaskValue
    
    # 裁剪图像
    h, w = normalized_image.shape[:2]
    start_h = (h - image_size) // 2
    start_w = (w - image_size) // 2
    cropped = normalized_image[start_h:start_h+image_size, start_w:start_w+image_size, :]
    
    # 转为张量
    image_tensor = torch.tensor(cropped, dtype=torch.float32).unsqueeze(0).unsqueeze(3)
    
    # Mask
    mask = torch.ones(1, image_size, image_size, 1, 3) * MaskValue.ONLINE_ENCODER.value
    
    # 时间戳
    timestamps = torch.tensor([[[15, 7, 2025]]])
    
    print(f"输入准备完成: {image_tensor.shape}")
    return image_tensor, mask, timestamps


def load_model():
    """加载模型
    
    支持两种config.json格式：
    1. 简化版：直接包含encoder_config和decoder_config
    2. 原始版：嵌套在"model"键下（带_CLASS_字段）
    """
    import json
    
    script_dir = Path(__file__).parent
    config_path = script_dir / "params" / "config.json"
    weights_path = script_dir / "params" / "weights.pth"
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    from nn.galileo import Galileo, GalileoConfig
    
    # 自动检测config格式并提取正确的配置
    if "model" in config_dict and isinstance(config_dict["model"], dict):
        # 原始格式：配置嵌套在"model"键下
        model_config = GalileoConfig.from_dict(config_dict["model"])
    else:
        # 简化格式：配置直接在根层级
        model_config = GalileoConfig.from_dict(config_dict)
    
    model = model_config.build()
    
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    print("模型加载完成")
    return model


@torch.no_grad()
def run_inference(model, image_tensor, mask, timestamps, patch_size=4, device="cuda"):
    """运行推理"""
    from datatypes import MaskedOlmoEarthSample
    
    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=image_tensor.to(device),
        sentinel2_l2a_mask=mask.to(device),
        timestamps=timestamps.to(device),
    )
    
    output = model.encoder(sample, fast_pass=True, patch_size=patch_size)
    tokens_and_masks = output["tokens_and_masks"]
    features = tokens_and_masks.sentinel2_l2a
    
    # 全局池化
    pooled = features.mean(dim=[2, 3, 4])
    
    return features, pooled


def main():
    parser = argparse.ArgumentParser(description="OlmoEarth JP2 Inference")
    parser.add_argument("--jp2_dir", type=str, required=True, help="Sentinel-2 SAFE文件夹路径")
    parser.add_argument("--image_size", type=int, default=64, help="输入图像大小")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch大小")
    parser.add_argument("--target_size", type=int, default=512, help="读取图像尺寸")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("OlmoEarth JP2 Inference")
    print("=" * 60)
    
    # 1. 读取JP2
    print(f"\n[1/4] 读取JP2文件...")
    image = read_sentinel2_jp2_files(args.jp2_dir, args.target_size)
    if image is None:
        return
    
    # 2. 归一化
    print("\n[2/4] 归一化...")
    normalized = normalize_image(image)
    
    # 3. 准备输入
    print("\n[3/4] 准备输入...")
    image_tensor, mask, timestamps = prepare_input(normalized, args.image_size)
    
    # 4. 加载模型并推理
    print("\n[4/4] 加载模型并推理...")
    model = load_model()
    model = model.to(args.device)
    model.eval()
    
    features, pooled = run_inference(
        model, image_tensor, mask, timestamps,
        patch_size=args.patch_size, device=args.device
    )
    
    print("\n" + "=" * 60)
    print("结果:")
    print(f"  Feature shape: {features.shape}")
    print(f"  Pooled shape: {pooled.shape}")
    print(f"  Pooled (first 5): {pooled[0, :5].cpu().numpy()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
