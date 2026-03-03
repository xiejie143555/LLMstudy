import torch
import torch.nn as nn

# ----------------- 准备输入数据 (与learn_attention.py中相同) -----------------
batch_size = 8
context_size = 4
output_dim = 256 # 这在Transformer论文中通常被称为 d_model

torch.manual_seed(123)
input_embeddings = torch.randn(batch_size, context_size, output_dim)
print("--- 输入 ---")
print("输入嵌入向量 (x) 的形状:", input_embeddings.shape)


# ----------------- 模块 1: 单个注意力头 (复用之前的代码) -----------------
# 这是多头注意力的基本构建模块
class CausalSelfAttentionHead(nn.Module):
    def __init__(self, d_model, head_dim, context_length, dropout_prob=0.0):
        super().__init__()
        # 注意：这里的输入维度是d_model, 输出维度是head_dim
        # head_dim 是 d_model / num_heads
        self.W_query = nn.Linear(d_model, head_dim, bias=False)
        self.W_key = nn.Linear(d_model, head_dim, bias=False)
        self.W_value = nn.Linear(d_model, head_dim, bias=False)
        self.dropout = nn.Dropout(dropout_prob)
        mask = torch.ones(context_length, context_length).triu(diagonal=1).bool()
        self.register_buffer('mask', mask)

    def forward(self, x):
        d_k = self.W_key.out_features
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)
        
        attn_scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
        attn_scores = attn_scores.masked_fill(self.mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ V
        return context_vec

# ----------------- 模块 2: 多头注意力机制 (Multi-Head Attention) -----------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, context_length, dropout_prob=0.0):
        """
        Args:
            d_model (int): 模型的总维度, 必须能被num_heads整除
            num_heads (int): 注意力头的数量
            context_length (int): 输入序列的最大长度
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        # 每个头的维度
        self.head_dim = d_model // num_heads

        # 步骤 1: 创建多个独立的注意力头
        # nn.ModuleList是一个特殊的列表，可以正确地注册它包含的所有模块
        self.heads = nn.ModuleList(
            [CausalSelfAttentionHead(d_model, self.head_dim, context_length, dropout_prob)
             for _ in range(num_heads)]
        )
        
        # 步骤 3: 创建一个最终的线性投影层
        # 这个层的作用是将所有头的输出拼接起来后，再进行一次线性变换
        # 这使得模型可以学习如何最好地融合来自不同头的信息
        self.proj_out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        """
        x: 输入嵌入向量，形状为 (batch, context, d_model)
        """
        # 步骤 2: 计算每个头的输出，并将它们拼接起来
        # self.heads(x) 会对x并行计算num_heads次注意力
        # - head(x) 的输出形状是 (batch, context, head_dim)
        # - torch.cat([...], dim=-1) 将所有头的输出在最后一个维度上拼接
        # - 拼接后的形状是 (batch, context, num_heads * head_dim)，也就是 (batch, context, d_model)
        concatenated_heads = torch.cat([head(x) for head in self.heads], dim=-1)
        
        # 步骤 3: 将拼接后的结果通过最终的投影层
        # 形状保持不变: (batch, context, d_model)
        output = self.proj_out(concatenated_heads)
        output = self.dropout(output)

        return output

# --- 实例化并运行模型 ---
d_model = output_dim
num_heads = 4 # 假设我们使用4个注意力头

# 实例化多头注意力模块
multi_head_attention = MultiHeadAttention(d_model, num_heads, context_size)

# 将输入送入模型
output = multi_head_attention(input_embeddings)

print("\n--- 输出 ---")
print("多头注意力模块的输出形状:", output.shape)
print("\n输出的形状与输入完全相同，这对于后续的残差连接至关重要。")
print(f"在内部，输入被投影到了{num_heads}个独立的{d_model // num_heads}维子空间中进行注意力计算，")
print("然后将结果拼接并融合，从而获得了比单头更丰富的上下文表示。")
