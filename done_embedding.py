import torch
import tiktoken

# 为了解释和复现性，我们设置一个随机种子
torch.manual_seed(123)

# 首先，我们从文件中读取原始文本数据
# 这里的 a-verdict.txt 是一个英文小说文本
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("原始文本的前100个字符:", raw_text[:100])

# 初始化一个预训练好的Tokenizer
# 我们使用OpenAI的"gpt2"模型对应的Tokenizer
# 这个Tokenizer基于BPE算法，并内置了一个包含50257个Token的词汇表
# 也就是说，分词和词表查询这两个步骤已经一次性完成了
# 接下来只需要使用tokenizer的词表，将原始文本转换为整数ID
tokenizer = tiktoken.get_encoding("gpt2")

# 使用tokenizer的encode方法，将原始文本字符串转换为一个整数ID列表
encoded_text = tokenizer.encode(raw_text)

print(f"文本被编码成了 {len(encoded_text)} 个整数ID")
print("编码后的前10个ID:", encoded_text[:10])

# 为了演示，我们只取一小段编码后的ID作为后续处理的输入
# 假设我们的模型一次只能处理8个Token
context_size = 8
inputs = torch.tensor(encoded_text[:context_size])
print(f"\n用于演示的输入 (前{context_size}个Token ID):", inputs)
print("输入的形状:", inputs.shape)

# 现在，需要把整数ID转换为向量，也就是Token Embedding
# 也就是词表中的每个token，都用一个向量来表示
# 所以需要定义词汇表大小和词嵌入向量的维度
# vocab_size = 50257  # GPT-2的词汇表大小
vocab_size = tokenizer.n_vocab  # 从tokenizer中直接获取词汇表大小
output_dim = 256     # 嵌入向量的维度，即每个ID将被映射成一个256维的向量

# 定义完词表大小和嵌入维度后，创建Token Embedding层
# 也就是词表中的每个token，都用一个向量来表示
# 词表总共有50257个词，每个词用一个256维的向量来表示
# 所以需要创建一个50257×256的矩阵，每一行代表一个词的向量
# 它的权重是随机初始化的，会在模型训练过程中被学习
# 这行代码就是创建一个50257×256的矩阵，每一行代表一个词的向量
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# 创建完50257×256的矩阵后，就可以把需要转为向量的整数ID传入Embedding层，把ID转为向量
# 将输入的整数ID张量传入Embedding层
# 对于输入中的每一个ID，Embedding层会返回其对应的256维向量
token_embeddings = token_embedding_layer(inputs)

print("\n经过Token Embedding后的向量形状:", token_embeddings.shape)
print("这代表我们有8个Token，每个Token都被转换成了一个256维的向量。")

# 接下来需要加入位置信息，创建Position Embedding层
# 同样使用nn.Embedding，但这次的"词汇表"大小是句子的最大长度 (context_size)
# 它为序列中的每一个位置 (0, 1, 2, ...) 学习一个对应的位置向量
pos_embedding_layer = torch.nn.Embedding(context_size, output_dim)
print("前几个向量",pos_embedding_layer.weight[:5])

# 生成位置索引 (0, 1, 2, ..., 7)
positions = torch.arange(context_size)
print("\n位置索引:", positions)

# 获取位置向量
pos_embeddings = pos_embedding_layer(positions)
print("位置嵌入向量的形状:", pos_embeddings.shape)

# 将Token Embedding和Position Embedding直接相加
# 这样得到的最终向量就同时包含了语义信息和位置信息
input_embeddings = token_embeddings + pos_embeddings

print("\n最终的输入嵌入向量形状 (Token Embedding + Position Embedding):", input_embeddings.shape)
print("这些向量将作为Transformer模型中Attention层的最终输入。")
