import torch
import torch.nn as nn

torch.manual_seed(123)   # 设置随机种子
batch_example = torch.randn(2, 5)   # 生成一个形状为 (2, 5) 的张量（Tensor），其中的数值符合标准正态分布（均值为 0，方差为 1）。
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())   # nn.Linear(5, 6)：全连接层（线性层）。它接收 5 个输入特征，并将其转换为 6 个输出特征。内部进行的是 $y = xA^T + b$ 的矩阵运算。
out = layer(batch_example)
print(out)

# out.mean(...): 计算平均值。
# dim=-1 意味着程序会横向计算：分别计算第 1 个样本那 6 个数字的平均值，以及第 2 个样本那 6 个数字的平均值。
# keepdim=True: 保持维度不变。如果不加这个，结果形状会变成 (2,)。加上后，结果形状维持为 (2, 1)。这样做是为了方便后续进行减法或除法运算（广播机制）。
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)   # out.var(...): 计算方差。方差反映了这组数据的离散程度（即这 6 个数字相互之间差得有多远）。
print("Mean:\n", mean)
print("Variance:\n", var)

# 接下来，对之前得到的层输出进行层归一化操作。具体方法是减去均值，并将结果除以方差的平方根（也就是标准差）：
out_norm = (out - mean) / torch.sqrt(var)

# 再计算输出结果的平均值和方差
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
torch.set_printoptions(sci_mode=False)
print("Normalized layer outputs:\n", out_norm)
print("Mean:\n", mean)
print("Variance:\n", var)
