import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from PIL import Image

class ViTFeatureExtractor(nn.Module):
    def __init__(self, model_name_or_path, layers_to_extract=[3, 6, -1], freeze_backbone=True):
        """
        初始化 ViT 特征提取器
        
        Args:
            model_name_or_path (str): 模型路径 (如 './dinov2-base-local') 或 HuggingFace ID
            layers_to_extract (list): 需要提取的层索引列表，例如 [3, 6, -1]
            freeze_backbone (bool): 是否冻结 Backbone 的参数
        """
        super().__init__()
        
        self.layers_to_extract = layers_to_extract
        
        # 1. 加载模型，关键配置：output_hidden_states=True
        # 这样 forward 时 outputs 中会包含 hidden_states
        print(f"正在加载模型: {model_name_or_path} ...")
        self.backbone = AutoModel.from_pretrained(
            model_name_or_path, 
            output_hidden_states=True
        )
        
        # 2. 根据需要冻结参数 (通常提取特征时不需要反向传播更新 ViT)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone 参数已冻结。")

    def forward(self, pixel_values):
        """
        前向传播
        
        Args:
            pixel_values (torch.Tensor): 经过 processor 处理后的图像张量 
                                         Shape: (Batch_Size, Channels, Height, Width)
        Returns:
            dict: 包含指定层特征的字典，格式为 {层索引: 特征Tensor}
        """
        # 模型推理
        outputs = self.backbone(pixel_values)
        
        # 获取所有隐藏层状态 tuple
        # hidden_states[0] 通常是 Embedding 层输出
        # hidden_states[1] 是第 1 个 Block 输出 ... 以此类推
        all_hidden_states = outputs.hidden_states
        
        extracted_features = {}
        
        # 提取指定层
        for layer_idx in self.layers_to_extract:
            # 支持负数索引 (如 -1 代表最后一层)
            # 为了字典 key 清晰，我们将负数索引转换为实际的正数索引
            # actual_idx = layer_idx if layer_idx >= 0 else len(all_hidden_states) + layer_idx
            
            feat = all_hidden_states[layer_idx]
            extracted_features[layer_idx] = feat
            
        return extracted_features

# -----------------------------------------------------------------------------
# 使用示例 (参考你的原始脚本逻辑)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 配置路径
    local_model_dir = "./dinov2-base-local"  # 你原本的路径
    image_path = 'v4_overlay.png'            # 你原本的图片
    
    # 1. 准备图片处理工具 (Processor 通常不放在模型类里，而是在 Dataset 中使用)
    try:
        processor = AutoImageProcessor.from_pretrained(local_model_dir)
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"] # 获取 tensor
    except Exception as e:
        print(f"为了演示代码，请确保路径正确。错误信息: {e}")
        # 如果没有本地文件，这里创建一个假数据用于演示 shape
        pixel_values = torch.randn(1, 3, 224, 224) 

    # 2. 实例化特征提取器
    # 提取第 3 层 (浅层), 第 6 层 (中层), 第 -1 层 (深层)
    extractor = ViTFeatureExtractor(
        model_name_or_path=local_model_dir, 
        layers_to_extract=[3, 6, -1],
        freeze_backbone=True
    )
    
    # 3. 提取特征
    extractor.eval() # 设为评估模式
    with torch.no_grad():
        features_dict = extractor(pixel_values)
    
    # 4. 打印结果
    print("\n--- 提取结果 ---")
    for layer_idx, tensor in features_dict.items():
        # ViT 输出 Shape 通常为 (Batch, Sequence_Length, Hidden_Dim)
        # Sequence_Length = (H/14 * W/14) + 1 (CLS Token)
        print(f"Layer {layer_idx} feature shape: {tensor.shape}")
        
    # 额外提示：如果需要验证最后一层
    # deep_feat = features_dict[-1]
    # print("深层特征提取成功")