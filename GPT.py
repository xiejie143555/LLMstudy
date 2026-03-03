import torch
import torch.nn as nn

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "emb_dim": 768,          # 嵌入维度
    "n_heads": 12,           # 注意力头的数量
    "n_layers": 12,          # 层数
    "drop_rate": 0.1,        # dropout率
    "qkv_bias": False        # 查询-键-值偏置
}

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__() #调用父类的初始化（__init__）方法，即调用 torch.nn.Module 类的 __init__方法
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # 词嵌入层，将 token ID → 向量
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 位置嵌入层，将位置索引 → 向量
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # Dropout层，正则化
        self.trf_blocks = nn.Sequential(  # Sequential，12个 DummyTransformerBlock
            # 使用占位符代替TransformerBlock
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        # 使用占位符代替LayerNorm
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])  # LayerNorm，最终归一化
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # Linear，输出投影到词汇表大小

    def forward(self, in_ids):
        batch_size, seq_len = in_ids.shape
        tok_embeds = self.tok_emb(in_ids)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_ids.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# 使用占位符代替TransformerBlock
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x

# 使用占位符代替LayerNorm
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
# print(batch)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
# print(logits)

torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)
