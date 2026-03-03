import torch
import torch.nn as nn

# ----------------- 准备输入数据 -----------------

# 假设这是从上一步 (learn_data_loader.py) 得到的、已经包含了Token和位置信息的嵌入向量
# batch_size = 8  (8个句子)
# context_size = 4 (每个句子4个token)
# output_dim = 256 (每个token用256维向量表示)
batch_size = 8
context_size = 4
output_dim = 256

torch.manual_seed(123)
input_embeddings = torch.randn(batch_size, context_size, output_dim)
print("--- 输入 ---")
print("输入嵌入向量的形状:", input_embeddings.shape)


# ----------------- 因果自注意力机制实现 -----------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout_prob=0.0):
        """
        Args:
            d_in (int): 输入向量的维度 (例如 256)
            d_out (int): 输出向量的维度 (Q, K, V的维度)
            context_length (int): 输入序列的最大长度
            dropout_prob (float): Dropout的概率
        """
        super().__init__()
        self.d_out = d_out
        
        # 步骤 1: QKV 线性变换
        # 创建三个独立的线性层，将输入向量x投影到Q, K, V空间
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        
        # 步骤 6: Dropout层
        self.dropout = nn.Dropout(dropout_prob)

        # 步骤 4: 创建因果掩码 (Causal Mask)
        # 这是一个形状为 (context_length, context_length) 的矩阵
        # torch.ones(context_length, context_length) 创建一个全1矩阵
        # .triu(diagonal=1) 将矩阵的上三角部分（不含对角线）保留，其余部分置为0
        # mask.bool() 将 0/1 矩阵转为布尔矩阵 (True/False)
        # 最终的mask中，未来位置（上三角）为True，当前及过去位置为False
        mask = torch.ones(context_length, context_length).triu(diagonal=1).bool()
        
        # 将mask注册为缓冲区(buffer)。缓冲区是模型的一部分，但不是可训练的参数。
        # 这样在模型移动到GPU时，mask也会被自动移动。
        self.register_buffer('mask', mask)


    def forward(self, x):
        """
        x: 输入嵌入向量，形状为 (batch_size, context_size, d_in)
        """
        # 1. QKV 线性变换
        # 将输入x分别送入三个线性层，得到Q, K, V
        # Q, K, V 的形状都为 (batch_size, context_size, d_out)
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)

        # 2. 计算注意力分数
        # 计算 Q 和 K 的转置的点积
        # Q: (b, c, d) | K: (b, c, d) -> K.transpose: (b, d, c)
        # attn_scores: (b, c, d) @ (b, d, c) -> (b, c, c)
        attn_scores = Q @ K.transpose(-2, -1)
        
        # 3. 缩放
        # 除以维度的平方根，防止梯度过小
        attn_scores = attn_scores / (self.d_out ** 0.5)

        # 4. 应用因果掩码
        # mask.masked_fill_会查找self.mask中为True的位置
        # 并将attn_scores中对应位置的值替换为-torch.inf (负无穷)
        # 这样可以确保在softmax后，未来位置的注意力权重为0
        attn_scores = attn_scores.masked_fill(self.mask, -torch.inf)
        
        # 5. 归一化 (Softmax)
        # 沿最后一个维度（键的维度）进行softmax，得到注意力权重
        # 形状仍为 (batch_size, context_size, context_size)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # 6. 应用Dropout
        attn_weights = self.dropout(attn_weights)

        # 7. 计算加权V (上下文向量)
        # 将注意力权重乘以V矩阵
        # (b, c, c) @ (b, c, d) -> (b, c, d)
        # 最终输出的形状与Q, K, V相同
        context_vec = attn_weights @ V
        
        return context_vec


# --- 实例化并运行模型 ---
# d_in必须与input_embeddings的维度匹配
d_in = output_dim
# d_out可以自定义，这里为了方便设为与d_in相同
d_out = output_dim

# 实例化因果自注意力模块
causal_attention = CausalSelfAttention(d_in, d_out, context_size)

# 将输入送入模型
output = causal_attention(input_embeddings)

print("\n--- 输出 ---")
print("输出的上下文向量形状:", output.shape)
print("\n输出的形状与输入完全相同，但每个Token的向量表示")
print("都已融合了它自身以及它之前所有Token的上下文信息。")
