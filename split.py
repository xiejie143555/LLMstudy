import re

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# 获取预处理后文本中所有不重复的单词，并按字典序排序
all_words = sorted(set(preprocessed))

# 创建词汇表，将每个单词映射到一个整数
vocab = {token:integer for integer,token in enumerate(all_words)}

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab #将词汇表作为类属性存储,以便在 encode方法和 decode 方法中访问
        self.int_to_str = {i:s for s,i in vocab.items()} # 创建逆向词汇表将词元 ID 映射回原始文本词元

    # 处理输入文本，将其转换为词元 ID
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    # 处理词元 ID，将其转换为原始文本
    def decode(self, ids):
        text = ' '.join([self.int_to_str[i] for i in ids])
        text = re.sub(r'([,.:;?_!"()\']|--|\s)', r'\1', text)
        return text

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    # 用<|unk|>替换不在词汇表中的词元
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    # 用<|unk|>替换不在词汇表中的词元
    def decode(self, ids):
        text = ' '.join([self.int_to_str[i] for i in ids])
        text = re.sub(r'([,.:;?_!"()\']|--|\s)', r'\1', text)
        return text

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))