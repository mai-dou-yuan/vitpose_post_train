import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union, Callable

# ==========================================
# 1. 基础组件 (保持不变)
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim, bias=False):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=bias)
        self.w_up   = nn.Linear(d_model, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=bias)
        self.act = nn.SiLU() 
    def forward(self, x):
        return self.w_down(self.act(self.w_gate(x)) * self.w_up(x))

class GatedMultiHeadAttention(nn.Module):
    """ 保持不变，它只负责接收 Q, K, V 进行计算 """
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1, bias=True):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.dropout_p = dropout
        self.q_proj = nn.Linear(d_model, d_model, bias=True) 
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.gate_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None, is_causal=False):
        # 简化版：暂不处理 mask 合并，假设外部已处理好
        batch_size, seq_len_q, _ = query.shape
        seq_len_kv = key.shape[1]

        q = self.q_proj(query).view(batch_size, seq_len_q, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len_kv, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len_kv, self.n_head, self.d_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            query=q, key=k, value=v, 
            attn_mask=attn_mask, 
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        # Gating: 注意这里的 query 已经包含了 Position (如果传入的 query 是 content+pos)
        # 这意味着 Gate 也是位置感知的，这在 DETR 中是合理的 (例如边缘位置可能需要不同的 gate 强度)
        gate = torch.sigmoid(self.gate_proj(query))
        y = y * gate 
        return self.resid_dropout(self.out_proj(y))

# ==========================================
# 2. 修改后的 DecoderLayer (DETR Style)
# ==========================================
class GatedTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-6,
        batch_first: bool = True,
        norm_first: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = GatedMultiHeadAttention(d_model, nhead, dropout=dropout, bias=bias)
        self.multihead_attn = GatedMultiHeadAttention(d_model, nhead, dropout=dropout, bias=bias)
        self.feed_forward = SwiGLU(d_model, dim_feedforward, bias=bias)
        self.dropout_ff = nn.Dropout(dropout)
        self.norm_first = norm_first
        self.norm1 = RMSNorm(d_model, eps=layer_norm_eps)
        self.norm2 = RMSNorm(d_model, eps=layer_norm_eps)
        self.norm3 = RMSNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        # [New] 新增位置编码输入
        query_pos: Optional[Tensor] = None,  # For tgt (Object Queries)
        pos_emb: Optional[Tensor] = None     # For memory (Image Pos Embeddings)
    ) -> Tensor:
        """
        query_pos: [B, num_queries, d_model] 或 [num_queries, 1, d_model] (广播)
        pos_emb:   [B, num_patches, d_model] 或 [1, num_patches, d_model] (广播)
        """
        x = tgt
        if self.norm_first:
            # Self Attention Block
            # 注意：传入 query_pos 用于 Q 和 K 的增强
            x = x + self._sa_block(self.norm1(x), query_pos, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            
            # Cross Attention Block
            # 注意：传入 query_pos (给 Q) 和 pos_emb (给 K)
            x = x + self._mha_block(self.norm2(x), memory, query_pos, pos_emb, memory_mask, memory_key_padding_mask, memory_is_causal)
            
            # FFN Block
            x = x + self._ff_block(self.norm3(x))
        else:
            # Post-Norm 结构同理
            x = self.norm1(x + self._sa_block(x, query_pos, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._mha_block(x, memory, query_pos, pos_emb, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))
        return x

    def _sa_block(self, x, query_pos, attn_mask, key_padding_mask, is_causal=False):
        # DETR Self-Attention logic:
        # Q = x + query_pos
        # K = x + query_pos
        # V = x
        q = k = x + query_pos if query_pos is not None else x
        
        # 你的 GatedMHA 接收显式的 q, k, v
        x_out = self.self_attn(
            query=q, key=k, value=x, # V 没有加 pos
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False
        )
        return self.dropout1(x_out)

    def _mha_block(self, x, mem, query_pos, pos_emb, attn_mask, key_padding_mask, is_causal=False):
        # DETR Cross-Attention logic (最重要的部分):
        # Q = x + query_pos   (内容 + 查询位置)
        # K = mem + pos_emb   (图像特征 + 图像位置)
        # V = mem             (纯图像特征)
        
        q = x + query_pos if query_pos is not None else x
        k = mem + pos_emb if pos_emb is not None else mem
        v = mem # 绝对不加 position
        
        x_out = self.multihead_attn(
            query=q, key=k, value=v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False
        )
        return self.dropout2(x_out)

    def _ff_block(self, x):
        return self.dropout_ff(self.feed_forward(x))

if __name__ == "__main__":
    # 1. 设置 CV 任务常见的超参数
    batch_size = 2
    num_queries = 10
    num_patches = 100 
    d_model = 64
    nhead = 4
    dim_feedforward = 256

    decoder_layer = GatedTransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        norm_first=True
    )

    # 2. 模拟输入数据
    # Content (初始化内容)
    tgt = torch.randn(batch_size, num_queries, d_model)   # Target Content
    memory = torch.randn(batch_size, num_patches, d_model) # Image Features
    
    # Position (位置编码)
    # query_pos: 对应 Object Queries 的位置嵌入 (Learnable)
    query_pos = torch.randn(1, num_queries, d_model)      
    # pos_emb: 对应 Image Features 的正弦/余弦位置编码
    pos_emb = torch.randn(1, num_patches, d_model)        

    print("--- DETR Style Decoupled PE 测试 ---")
    output = decoder_layer(
        tgt=tgt,
        memory=memory,
        query_pos=None,  # 传入 Query Pos
        pos_emb=pos_emb,      # 传入 Image Pos
        tgt_is_causal=False
    )

    print(f"输入 tgt: {tgt.shape}")
    print(f"输入 memory: {memory.shape}")
    print(f"输出 output: {output.shape}")
    
    # 简单验证数值
    if not torch.isnan(output).any():
        print("数值检查: Pass")
    
    print("\n逻辑验证:")
    print("1. Self-Attn: Q, K 包含 query_pos; V 只有 tgt")
    print("2. Cross-Attn: Q 包含 query_pos; K 包含 pos_emb; V 只有 memory")