import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

# ----------------- 步骤 1: 创建PyTorch数据集 (Dataset) -----------------

# 在PyTorch中，`Dataset`是一个抽象类，我们通过继承它来创建自己的数据集
# 我们的目标是创建一个数据集，它能根据给定的长文本，自动生成模型训练所需的 (输入, 目标) 对
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        """
        初始化函数，负责准备所有的数据样本
        Args:
            txt (str): 原始文本数据
            tokenizer: 用于分词和编码的tokenizer对象
            max_length (int): 每个输入序列的最大长度 (即上下文窗口大小)
            stride (int): 生成样本时，滑动窗口的步长
        """
        self.input_ids = []
        self.target_ids = []

        # 1. 首先，将全部文本编码成一个长长的整数ID列表
        token_ids = tokenizer.encode(txt)

        # 2. 使用“滑动窗口”的方法，从token_ids中切分出大量的训练样本
        # 例如，如果 max_length=4, stride=1, token_ids=[1, 2, 3, 4, 5, 6]
        # 第一次循环: i=0, input_chunk=[1, 2, 3, 4], target_chunk=[2, 3, 4, 5]
        # 第二次循环: i=1, input_chunk=[2, 3, 4, 5], target_chunk=[3, 4, 5, 6]
        # ...以此类推
        for i in range(0, len(token_ids) - max_length, stride):
            # 提取输入序列
            input_chunk = token_ids[i:i + max_length]
            # 提取目标序列 (输入序列向右平移一位)
            target_chunk = token_ids[i + 1:i + max_length + 1]
            
            # 将切片转换为PyTorch张量并存储
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # 这个方法需要返回数据集中样本的总数
        return len(self.input_ids)

    def __getitem__(self, idx):
        # 这个方法需要根据索引idx，返回一个 (输入, 目标) 对
        return self.input_ids[idx], self.target_ids[idx]


# ----------------- 步骤 2: 创建数据加载器 (DataLoader) -----------------

def create_dataloader_V1(txt, batch_size=4, max_length=256, stride=128, shuffle=True):
    """
    一个辅助函数，用于封装数据集的创建和DataLoader的实例化
    """
    # 初始化tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # 创建数据集实例
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    
    # 创建DataLoader实例
    # DataLoader会自动从Dataset中获取数据，并将它们打包成一个“批次(batch)”
    # - batch_size: 每批包含多少个样本
    # - shuffle: 是否在每个epoch开始时打乱数据顺序，这有助于提高模型泛化能力
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

# ----------------- 步骤 3: 实际使用DataLoader并进行词嵌入 -----------------

# --- 数据准备 ---
# 读取原始文本
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 为了演示，我们使用较小的参数
# batch_size=8: 每批8个样本
# max_length=4: 每个样本的上下文长度为4
# stride=4: 窗口每次滑动4个token来生成下一个样本 (无重叠)
dataloader = create_dataloader_V1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

# 从dataloader中获取一个数据批次
# data_iter是一个迭代器，next()函数会返回下一批数据
data_iter = iter(dataloader)
first_batch = next(data_iter)

# 查看批次数据的结构
inputs, targets = first_batch
print("--- DataLoader 输出 ---")
print("输入 (Inputs) 的形状:", inputs.shape) # 形状应为 [batch_size, max_length]
print("输入的前2个样本:\n", inputs[:2])
print("\n目标 (Targets) 的形状:", targets.shape) # 形状也为 [batch_size, max_length]
print("目标的前2个样本:\n", targets[:2])
print("可以看到，目标是输入向右平移一位的结果，这正是模型要学习预测的内容。")


# --- 词嵌入过程 (与done_embedding.py中类似) ---
print("\n--- 词嵌入过程 ---")

# 设置随机种子以保证结果可复现
torch.manual_seed(123)

# 初始化tokenizer来获取词汇表大小
tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab
output_dim = 256 # 嵌入维度

# Token Embedding层
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# Position Embedding层
context_size = inputs.shape[1] # 从输入的形状获取上下文长度
pos_embedding_layer = torch.nn.Embedding(context_size, output_dim)

# --- 将词嵌入应用于从DataLoader获取的输入批次 ---
token_embeddings = token_embedding_layer(inputs)
pos_embeddings = pos_embedding_layer(torch.arange(context_size))

# 使用广播机制将位置嵌入添加到批次中的每个样本上
# token_embeddings: [8, 4, 256]
# pos_embeddings: [4, 256] -> PyTorch会自动广播为 [8, 4, 256]
input_embeddings = token_embeddings + pos_embeddings

print("输入批次 (Inputs) 的形状:", inputs.shape)
print("经过Token Embedding后的形状:", token_embeddings.shape)
print("位置嵌入 (Position Embeddings) 的形状:", pos_embeddings.shape)
print("最终输入到模型的嵌入向量形状:", input_embeddings.shape)
print("\n这批形状为[8, 4, 256]的张量，就是准备好进入Transformer模型的最终输入了。")
print("目标(Targets)则保持其[8, 4]的整数ID形状，用于计算损失。")
