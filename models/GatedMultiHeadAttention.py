import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        """
        支持 Cross Attention 的 Gated Attention。
        基于 arXiv:2505.06708v1 实现，采用 G1 位置 (SDPA Output) + Element-wise Sigmoid Gating。
        """
        super().__init__()
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"
        
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.dropout_p = dropout

        # 1. Q, K, V 投影
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 2. Gating Projection (G1)
        # 门控总是基于 Query 侧的输入计算，以匹配输出维度
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)

        # 3. Output Projection (Wo)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, query, key=None, value=None, is_causal=False, attn_mask=None):
        """
        Args:
            query: Query 输入张量 (Batch_Size, Seq_Len_Q, d_model)
            key:   Key 输入张量 (Batch_Size, Seq_Len_KV, d_model)。如果为 None，则默认为 Self-Attention (key=query)
            value: Value 输入张量 (Batch_Size, Seq_Len_KV, d_model)。如果为 None，则默认为 key
            is_causal: 是否使用因果掩码 (Cross Attention 通常为 False)
            attn_mask: 掩码 (Batch_Size, Seq_Len_Q, Seq_Len_KV)
        """
        # 处理输入：支持 Cross Attention 和 Self Attention
        if key is None:
            key = query
        if value is None:
            value = key

        # 获取维度信息
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_kv = key.size(1)

        # --- 第一步：投影生成 Q, K, V ---
        q = self.q_proj(query) # Shape: (B, T_q, D)
        k = self.k_proj(key)   # Shape: (B, T_kv, D)
        v = self.v_proj(value) # Shape: (B, T_kv, D)

        # 拆分多头: (B, T, C) -> (B, n_head, T, d_head) -> (B, n_head, T, d_head)
        q = q.view(batch_size, seq_len_q, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len_kv, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len_kv, self.n_head, self.d_head).transpose(1, 2)

        # --- 第二步：Scaled Dot-Product Attention (SDPA) ---
        # 输出 y 的长度将与 query 保持一致 (T_q)
        y = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            attn_mask=attn_mask, 
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal
        )
        
        # 合并多头: (B, n_head, T_q, d_head) -> (B, T_q, C)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)

        # --- 第三步：应用 Gating 机制 (G1 Position) ---
        # 关键点：在 Cross Attention 中，Gate 必须由 Query 输入 (X_q) 计算
        # 这样才能保证 gate 的形状 (B, T_q, D) 与 attention 输出 y 的形状一致
        
        gate_score = self.gate_proj(query)  # 使用 query 输入计算门控
        gate = torch.sigmoid(gate_score)    # Element-wise Sigmoid
        
        y_gated = y * gate  # 稀疏化信息流，这一步实现了论文提到的 Input-Dependent Sparsity

        # --- 第四步：最终输出投影 (Wo) ---
        output = self.out_proj(y_gated)
        
        return self.resid_dropout(output)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.dropout_p = dropout

        # Q, K, V projection
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, query, key=None, value=None, is_causal=False, attn_mask=None):
        """
        Args:
            query: (B, T_q, d_model)
            key:   (B, T_kv, d_model)，None 时默认 self-attention
            value: (B, T_kv, d_model)，None 时默认 value=key
            is_causal: 是否使用因果掩码
            attn_mask: 掩码
        """
        if key is None:
            key = query
        if value is None:
            value = key

        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_kv = key.size(1)

        # 1. 线性投影得到 Q/K/V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 2. 拆成多头
        q = q.view(batch_size, seq_len_q, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len_kv, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len_kv, self.n_head, self.d_head).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        y = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal
        )

        # 4. 合并多头
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)

        # 5. 直接输出投影（无 gate）
        output = self.out_proj(y)

        return self.resid_dropout(output)

# --- 测试代码 ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 使用较小的维度以便观察
    model = GatedMultiHeadAttention(d_model=512, n_head=8).to(device)
    
    # 模拟 Cross Attention 场景 (例如 Object Query 关注 Image Features)
    # Query: 10 个 object queries (B, 10, 512)
    x_q = torch.randn(2, 10, 512).to(device)
    # Key/Value: 196 个 image patches (B, 196, 512)
    x_kv = torch.randn(2, 196, 512).to(device)
    
    # 前向传播 (传入独立的 query 和 key/value)
    # CV 任务中通常 is_causal=False
    output = model(query=x_q, key=x_kv, value=x_kv, is_causal=False)
    
    print(f"Query shape: {x_q.shape}")
    print(f"KV shape:    {x_kv.shape}")
    print(f"Output shape:{output.shape}") # 预期: [2, 10, 512]，与 Query 形状一致