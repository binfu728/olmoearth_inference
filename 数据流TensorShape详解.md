# OlmoEarth Inference 数据流 Tensor Shape 详解

本文档详细追踪从原始数据到最终推理输出的完整数据流，记录每个关键步骤的 Tensor Shape 变化。

---

## 概览

```
原始数据 (JP2) 
    ↓
NumPy数组 [H, W, C=12]
    ↓
归一化后 [H, W, C=12]
    ↓
PyTorch Tensor [B=1, H, W, T=1, C=12]
    ↓
MaskedOlmoEarthSample 组装
    ↓
MultiModalPatchEmbeddings (BandSet分离 + Patch化)
    ↓
CompositeEncodings (位置编码+通道编码)
    ↓
ViT Encoder (Transformer Blocks)
    ↓
TokensAndMasks 输出
    ↓
最终特征 [B, H', W', T, S=3, D=768]
```

---

## 1. 原始数据读取

### 1.1 JP2文件读取 → NumPy数组

**源代码：** `olmoearth_inference_jp2_simple.py` 第29-79行

```
输入: Sentinel-2 L2A JP2文件 (12个波段)
    ↓
输出: np.ndarray [H, W, C=12]  # Channel-Last格式
```

- H = 图像高度（默认512，经WarpedVRT重采样）
- W = 图像宽度（默认512）
- C = 12（波段数量）

### 1.2 归一化

**源代码：** `olmoearth_inference_jp2_simple.py` 第82-102行

```
输入: [H, W, C=12]
    ↓
输出: [H, W, C=12]  # 值域变为[0, 1]
```

---

## 2. PyTorch Tensor 准备

### 2.1 添加Batch和Time维度

**源代码：** `olmoearth_inference_jp2_simple.py` 第105-134行

```
输入: [H, W, C=12]  # NumPy数组
    ↓
torch.tensor(cropped)  # 添加维度
    ↓
unsqueeze(0)  # 添加Batch维度
    ↓
unsqueeze(3)  # 添加Time维度
    ↓
输出: [B=1, H, W, T=1, C=12]
```

### 2.2 Mask Tensor 创建

**源代码：** `olmoearth_inference_jp2_simple.py` 第128-132行

```python
S = 3  # BandSet数量（Sentinel-2 L2A有3个BandSet）
mask = torch.full((B, H, W, T, S), MaskValue.ONLINE_ENCODER.value)
```

```
输出: [B=1, H, W, T=1, S=3]
```

### 2.3 Timestamps 创建

**源代码：** `olmoearth_inference_jp2_simple.py` 第340-342行

```python
timestamps = torch.tensor([[[22, 7, 2025]]], dtype=torch.int64)
```

```
输出: [B=1, T=1, D=3]  # D=[day, month, year]
```

---

## 3. MaskedOlmoEarthSample 组装

**源代码：** `olmoearth_inference_jp2_simple.py` 第339-345行

```python
sample = MaskedOlmoEarthSample(
    timestamps=timestamps,                    # [B, T, D=3]
    sentinel2_l2a=image_tensor,               # [B, H, W, T, C]
    sentinel2_l2a_mask=mask,                  # [B, H, W, T, S]
)
```

**MaskedOlmoEarthSample 各字段 Shape：**

| 字段 | Shape | 说明 |
|------|-------|------|
| `timestamps` | `[B, T, D=3]` | 时间信息 |
| `sentinel2_l2a` | `[B, H, W, T, C=12]` | 原始12波段数据 |
| `sentinel2_l2a_mask` | `[B, H, W, T, S=3]` | BandSet掩码 |

---

## 4. MultiModalPatchEmbeddings（关键：BandSet分离 + Patch化）

**源代码：** `nn/flexi_vit.py` 第183-419行

这是理解Shape变化的核心模块。

### 4.1 BandSet索引选择

**源代码：** `nn/flexi_vit.py` 第309-326行

```python
# 通过buffer索引从原始数据中提取对应bandset的波段
inp_data = torch.index_select(modality_data, -1, getattr(self, buffer_name))
```

Sentinel-2 L2A的3个BandSet：
| BandSet | 波段 | 原始通道索引 | 提取后通道数 |
|---------|------|-------------|-------------|
| 0 | B02, B03, B04, B08 | [0, 1, 2, 7] | 4 |
| 1 | B05, B06, B07, B8A, B11, B12 | [3, 4, 5, 6, 9, 10] | 6 |
| 2 | B01, B09 | [8, 11] | 2 |

### 4.2 FlexiPatchEmbed（Patch化）

**源代码：** `nn/flexi_patch_embed.py`

每个BandSet独立通过 `FlexiPatchEmbed`：

```
输入: [B, H, W, T, C_bandset]  # C_bandset = 4, 6, 或 2
    ↓
FlexiPatchEmbed (patch_size=4)
    ↓
输出: [B, H/patch_size, W/patch_size, T, D]
```

以BandSet 0为例（C=4）：
```
输入: [B=1, H, W, T=1, C=4]
    ↓
输出: [B=1, H/4, W/4, T=1, D=768]
```

### 4.3 Stack所有BandSet

**源代码：** `nn/flexi_vit.py` 第351-353行

```python
# 收集每个bandset的token
modality_tokens.append(patchified_data)
modality_masks.append(token_mask)

# Stack: [B, H', W', T, S, D]
return torch.stack(modality_tokens, dim=-2), torch.stack(modality_masks, dim=-1)
```

**MultiModalPatchEmbeddings输出：**

```
输出: {
    'sentinel2_l2a': [B=1, H/4, W/4, T=1, S=3, D=768],
    'sentinel2_l2a_mask': [B=1, H/4, W/4, T=1, S=3]
}
```

其中 `H' = H / patch_size`, `W' = W / patch_size`

---

## 5. CompositeEncodings（编码添加）

**源代码：** `nn/flexi_vit.py` 第612-853行

### 5.1 编码组成

**源代码：** `nn/flexi_vit.py` 第648-820行

每个token embedding的768维被分为4部分，每部分占 768/4 = 192维：

| 编码类型 | 维度范围 | 位置 |
|---------|---------|------|
| Channel Embedding | [0:192] | 通道/模态标识 |
| Position Time Embedding | [192:384] | 时间位置 |
| Month Embedding | [384:576] | 月份信息 |
| Position Space Embedding | [576:768] | 空间位置 |

### 5.2 Shape变化

```
输入: [B, H', W', T, S=3, D=768]
    ↓
CompositeEncodings (各编码相加)
    ↓
输出: [B, H', W', T, S=3, D=768]  # Shape不变，内容变化
```

---

## 6. ViT Encoder（Transformer Blocks）

**源代码：** `nn/flexi_vit.py` 第856-1890行

### 6.1 Collapse空间维度

**源代码：** `nn/flexi_vit.py` 第944-962行

```python
# collapse_and_combine_hwtc: 将空间维度展平到序列维度
tokens, masks = [], []
for modality in modalities_to_process:
    x_modality = x[modality]
    tokens.append(rearrange(x_modality, "b ... d -> b (...) d"))
    masks.append(rearrange(x_modality_mask, "b ... -> b (...)"))

tokens = torch.cat(tokens, dim=1)
masks = torch.cat(masks, dim=1)
```

```
输入: [B, H', W', T, S=3, D=768]
    ↓
rearrange: "b h w t s d -> b (h w t s) d"
    ↓
输出: [B, H'*W'*T*S, D] = [B, H'*W'*3, 768]
```

### 6.2 Transformer Blocks

**源代码：** `nn/flexi_vit.py` 第890-905行

```python
self.blocks = nn.ModuleList([
    Block(embedding_size, num_heads, mlp_ratio, ...)
    for _ in range(depth)  # depth=12
])
```

```
输入: [B, H'*W'*T*S, D=768]
    ↓
12层Transformer Blocks (每层: Attention + MLP)
    ↓
输出: [B, H'*W'*T*S, D=768]  # Shape不变
```

### 6.3 Split回原始维度

**源代码：** `nn/flexi_vit.py` 第984-1040行

```python
# split_tokens_masks_and_dims: 恢复原始空间维度
```

```
输入: [B, H'*W'*T*S, D=768]
    ↓
rearrange: "b (h w t s) d -> b h w t s d"
    ↓
输出: [B, H', W', T, S, D=768]
```

---

## 7. TokensAndMasks 输出

**源代码：** `nn/flexi_vit.py` 第480-572行，`datatypes.py` 第480-630行

```python
class TokensAndMasks(NamedTuple):
    sentinel2_l2a: Tensor | None      # [B, H', W', T, S, D]
    sentinel2_l2a_mask: Tensor | None  # [B, H', W', T, S]
```

### Encoder输出

**源代码：** `nn/flexi_vit.py` 第1046-1110行

```python
output = {
    "tokens_and_masks": TokensAndMasks(
        sentinel2_l2a=tokens,         # [B, H', W', T, S, D]
        sentinel2_l2a_mask=masks,     # [B, H', W', T, S]
    ),
    ...
}
```

```
Encoder最终输出:
{
    'sentinel2_l2a': [B=1, H/4, W/4, T=1, S=3, D=768],
    'sentinel2_l2a_mask': [B=1, H/4, W/4, T=1, S=3]
}
```

---

## 8. 推理脚本中的后处理

**源代码：** `olmoearth_inference_jp2_simple.py` 第347-360行

### 8.1 提取特征

```python
output = model(sample, fast_pass=True, patch_size=4)
tokens_and_masks = output["tokens_and_masks"]
features = tokens_and_masks.sentinel2_l2a  # [B, H', W', T, S, D]
```

```
features: [B=1, H/4, W/4, T=1, S=3, D=768]
```

### 8.2 Pooled（时间+BandSet维度平均）

```python
pooled = features.mean(dim=[3, 4])  # 对T和S维度求平均
```

```
输入: [B=1, H', W', T=1, S=3, D=768]
    ↓
mean(dim=[3, 4])  # T=1, S=3 → 被压缩
    ↓
输出: [B=1, H', W', D=768]
```

---

## 9. 完整Shape变化汇总

| 步骤 | Tensor | Shape | 说明 |
|------|--------|-------|------|
| 1 | JP2 → NumPy | `[H, W, C=12]` | Channel-Last格式 |
| 2 | 归一化后 | `[H, W, C=12]` | 值域[0,1] |
| 3 | PyTorch Tensor | `[B=1, H, W, T=1, C=12]` | 添加Batch和Time |
| 4 | Mask Tensor | `[B=1, H, W, T=1, S=3]` | BandSet掩码 |
| 5 | Timestamps | `[B=1, T=1, D=3]` | 时间信息 |
| 6 | **MultiModalPatchEmbeddings** | | |
| 6a | BandSet 0 提取 | `[B, H, W, T, C=4]` | 4通道 |
| 6b | BandSet 0 Patch化 | `[B, H/4, W/4, T, D=768]` | FlexiPatchEmbed |
| 6c | BandSet 1 提取 | `[B, H, W, T, C=6]` | 6通道 |
| 6d | BandSet 1 Patch化 | `[B, H/4, W/4, T, D=768]` | FlexiPatchEmbed |
| 6e | BandSet 2 提取 | `[B, H, W, T, C=2]` | 2通道 |
| 6f | BandSet 2 Patch化 | `[B, H/4, W/4, T, D=768]` | FlexiPatchEmbed |
| 6g | Stack后 | `[B, H/4, W/4, T, S=3, D=768]` | 3个BandSet合并 |
| 7 | **CompositeEncodings** | `[B, H', W', T, S=3, D=768]` | Shape不变，添加编码 |
| 8 | **ViT Encoder** | | |
| 8a | Collapse | `[B, H'*W'*T*S, D]` | 展平空间维度 |
| 8b | Transformer | `[B, H'*W'*T*S, D]` | 12层Block处理 |
| 8c | Split | `[B, H', W', T, S, D=768]` | 恢复空间维度 |
| 9 | **TokensAndMasks** | `[B, H', W', T=1, S=3, D=768]` | Encoder输出 |
| 10 | **Pooled** | `[B, H', W', D=768]` | 时间+BandSet平均 |

其中 `H' = H / patch_size`, `W' = W / patch_size`, 默认 `patch_size=4`

---

## 10. 具体数值示例

假设输入图像为 128×128：

| 步骤 | Shape |
|------|-------|
| 原始输入 | `[1, 128, 128, 1, 12]` |
| Patch化后 | `[1, 32, 32, 1, 3, 768]` |
| ViT后 | `[1, 32, 32, 1, 3, 768]` |
| Pooled | `[1, 32, 32, 768]` |

---

## 11. 关键代码位置索引

| 模块 | 文件 | 行号 |
|------|------|------|
| JP2读取 | `olmoearth_inference_jp2_simple.py` | 29-79 |
| 归一化 | `olmoearth_inference_jp2_simple.py` | 82-102 |
| Tensor准备 | `olmoearth_inference_jp2_simple.py` | 105-134 |
| Sample组装 | `olmoearth_inference_jp2_simple.py` | 339-345 |
| BandSet定义 | `data/constants.py` | 246-259 |
| MultiModalPatchEmbeddings | `nn/flexi_vit.py` | 183-419 |
| FlexiPatchEmbed | `nn/flexi_patch_embed.py` | - |
| CompositeEncodings | `nn/flexi_vit.py` | 612-853 |
| ViT Encoder | `nn/flexi_vit.py` | 856-1890 |
| TokensAndMasks | `datatypes.py` | 480-630 |
| Pooled处理 | `olmoearth_inference_jp2_simple.py` | 354 |
