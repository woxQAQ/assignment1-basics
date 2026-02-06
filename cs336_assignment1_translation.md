# CS336 作业 1（基础）：构建 Transformer 语言模型

**版本 1.0.6**  
**CS336 教学团队**  
**2025年春季**

---

## 1. 作业概述

在本作业中，你将从零开始构建训练标准 Transformer 语言模型（LM）所需的所有组件，并训练一些模型。

### 你需要实现的内容

1. 字节对编码（BPE）分词器（§2）
2. Transformer 语言模型（LM）（§3）
3. 交叉熵损失函数和 AdamW 优化器（§4）
4. 训练循环，支持序列化和加载模型及优化器状态（§5）

### 你需要运行的内容

1. 在 TinyStories 数据集上训练 BPE 分词器。
2. 在数据集上运行训练好的分词器，将其转换为整数 ID 序列。
3. 在 TinyStories 数据集上训练 Transformer LM。
4. 使用训练好的 Transformer LM 生成样本并评估困惑度。
5. 在 OpenWebText 上训练模型，并将你获得的困惑度提交到排行榜。

### 你可以使用的工具

我们希望你从基础开始构建这些组件。特别是，**不得**使用 `torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的任何定义，以下情况除外：

- `torch.nn.Parameter`
- `torch.nn` 中的容器类（例如 `Module`、`ModuleList`、`Sequential` 等）¹
- `torch.optim.Optimizer` 基类

你可以使用任何其他 PyTorch 定义。如果你想使用某个函数或类但不确定是否允许，请随时在 Slack 上询问。如有疑问，请考虑使用它是否会违背本作业"从零开始"的精神。

> ¹参见 [PyTorch 文档](https://PyTorch.org/docs/stable/nn.html#containers)获取完整列表。

---

### 关于 AI 工具的声明

允许使用 ChatGPT 等 LLM 来处理低级编程问题或关于语言模型的高级概念问题，但**禁止**直接使用它来解决作业问题。

我们强烈建议你在完成作业时在 IDE 中禁用 AI 自动补全功能（例如 Cursor Tab、GitHub CoPilot）（尽管非 AI 自动补全，例如自动补全函数名，是完全没问题的）。我们发现 AI 自动补全会大大降低你与内容深度互动的能力。

---

### 代码结构

所有作业代码以及本说明文档都可以在 GitHub 上找到：

**github.com/stanford-cs336/assignment1-basics**

请克隆该仓库。如果有任何更新，我们会通知你，你可以通过 `git pull` 获取最新版本。

1. **`cs336_basics/*`**：这是你编写代码的地方。注意这里没有任何代码——你可以完全从零开始！
2. **`adapters.py`**：你的代码必须具备一组功能。对于每个功能（例如缩放点积注意力），填写其实现（例如 `run_scaled_dot_product_attention`），只需调用你的代码即可。注意：你对 `adapters.py` 的修改不应包含任何实质性逻辑；这只是胶水代码。
3. **`test_*.py`**：这包含你必须通过的所有测试（例如 `test_scaled_dot_product_attention`），它们会调用 `adapters.py` 中定义的钩子。不要编辑测试文件。

### 如何提交

你将向 Gradescope 提交以下文件：

- **`writeup.pdf`**：回答所有书面问题。请使用排版工具（如 LaTeX）撰写你的回答。
- **`code.zip`**：包含你编写的所有代码。

要向排行榜提交，请向以下仓库提交 PR：

**github.com/stanford-cs336/assignment1-basics-leaderboard**

有关详细提交说明，请参阅排行榜仓库中的 README.md。

### 数据集获取

本作业将使用两个预处理的数据集：**TinyStories** [Eldan and Li, 2023] 和 **OpenWebText** [Gokaslan et al., 2019]。两个数据集都是单个大型纯文本文件。如果你正在修读这门课程，可以在任何非头节点的 `/data` 目录下找到这些文件。如果你在家学习，可以使用 README.md 中的命令下载这些文件。

---

> **💡 低资源/降规模提示：初始化**
>
> 在整个课程的作业说明中，我们将提供关于如何在更少或没有 GPU 资源的情况下完成作业的建议。例如，我们有时会建议缩小数据集或模型规模，或解释如何在 MacOS 集成 GPU 或 CPU 上运行训练代码。
>
> 你会在这些"低资源提示"框中找到这些建议（就像这个框一样）。即使你是斯坦福大学的在读学生，可以访问课程机器，这些提示也可能帮助你更快地迭代并节省时间，因此我们建议你阅读它们！

---

> **💡 低资源/降规模提示：在 Apple Silicon 或 CPU 上完成作业 1**
>
> 使用工作人员提供的解决方案代码，我们可以在 Apple M3 Max 芯片（36 GB RAM）上训练一个 LM，在 Metal GPU（MPS）上不到 5 分钟，使用 CPU 约 30 分钟，就能生成相当流畅的文本。如果这些术语对你来说不太熟悉，不用担心！只要知道如果你有一台配置合理的最新笔记本电脑，并且你的实现正确且高效，你就能够训练一个能生成简单儿童故事的小型 LM，流畅度还不错。
>
> 在本作业后面，我们会解释如果你在 CPU 或 MPS 上需要做哪些更改。

---

## 2. 字节对编码（BPE）分词器

在作业的第一部分，我们将训练和实现字节级字节对编码（BPE）分词器 [Sennrich et al., 2016, Wang et al., 2019]。

具体来说，我们将任意（Unicode）字符串表示为字节序列，并在此字节序列上训练我们的 BPE 分词器。稍后，我们将使用此分词器将文本（字符串）编码为用于语言建模的词元（整数序列）。

### 2.1 Unicode 标准

Unicode 是一种文本编码标准，将字符映射为整数代码点。截至 Unicode 16.0（2024年9月发布），该标准定义了 154,998 个字符，涵盖 168 种文字。例如，字符 "s" 的代码点是 115（通常表示为 U+0073，其中 U+ 是常规前缀，0073 是 115 的十六进制表示），字符 "牛" 的代码点是 29275。在 Python 中，你可以使用 `ord()` 函数将单个 Unicode 字符转换为其整数表示。`chr()` 函数则将整数 Unicode 代码点转换为包含相应字符的字符串。

```python
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

**问题（unicode1）：理解 Unicode（1分）**

(a) `chr(0)` 返回什么 Unicode 字符？  
**交付物**：一句话回答。

(b) 这个字符的字符串表示（`__repr__()`）与其打印表示有何不同？  
**交付物**：一句话回答。

(c) 当这个字符出现在文本中时会发生什么？在 Python 解释器中尝试以下内容可能会有帮助：

```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```
**交付物**：一句话回答。

### 2.2 Unicode 编码

虽然 Unicode 标准定义了从字符到代码点（整数）的映射，但直接在 Unicode 代码点上训练分词器是不切实际的，因为词汇量会过大（约 150K 项）且稀疏（因为许多字符非常罕见）。相反，我们将使用 Unicode 编码，将 Unicode 字符转换为字节序列。Unicode 标准本身定义了三种编码：UTF-8、UTF-16 和 UTF-32，其中 UTF-8 是互联网上的主导编码（超过 98% 的网页使用）。

要在 Python 中将 Unicode 字符串编码为 UTF-8，我们可以使用 `encode()` 函数。要访问 Python bytes 对象的字节值，我们可以对其进行迭代（例如调用 `list()`）。最后，我们可以使用 `decode()` 函数将 UTF-8 字节字符串解码为 Unicode 字符串。

```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # 获取编码字符串的字节值（0到255之间的整数）
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
>>> # 一个字节不一定对应一个 Unicode 字符！
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

通过将 Unicode 代码点转换为字节序列（例如通过 UTF-8 编码），我们本质上是将代码点序列（0 到 154,997 范围内的整数）转换为字节值序列（0 到 255 范围内的整数）。256 长度的字节词汇量更容易处理。使用字节级分词时，我们不必担心词汇外词元，因为我们知道任何输入文本都可以表示为 0 到 255 范围内的整数序列。

**问题（unicode2）：Unicode 编码（3分）**

(a) 相对于 UTF-16 或 UTF-32，在 UTF-8 编码字节上训练分词器有哪些优势？比较各种输入字符串的这些编码输出可能会有帮助。  
**交付物**：一到两句话回答。

(b) 考虑以下（错误的）函数，它旨在将 UTF-8 字节字符串解码为 Unicode 字符串。为什么这个函数是错误的？提供一个产生错误结果的输入字节字符串示例。

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```
**交付物**：一个 `decode_utf8_bytes_to_str_wrong` 产生错误输出的输入字节字符串示例，以及一句话解释为什么该函数是错误的。

(c) 给出一个无法解码为任何 Unicode 字符的双字节序列。  
**交付物**：一个示例，并附一句话解释。

### 2.3 子词分词

虽然字节级分词可以缓解词级分词器面临的词汇外问题，但将文本分词为字节会导致输入序列极长。这会减慢模型训练速度，因为一个包含 10 个词的句子在词级语言模型中可能只有 10 个词元，但在字符级模型中可能长达 50 个或更多词元（取决于词的长度）。处理这些更长的序列需要在模型的每一步进行更多计算。此外，在字节序列上进行语言建模很困难，因为更长的输入序列会在数据中创建长期依赖关系。

子词分词是词级分词器和字节级分词器之间的中间点。注意字节级分词器的词汇量有 256 个条目（字节值范围是 0 到 255）。子词分词器以更大的词汇量换取更好的输入字节序列压缩。例如，如果字节序列 `b'the'` 经常出现在我们的原始文本训练数据中，将其分配为词汇表中的一个条目会将这个 3 词元序列减少为单个词元。

我们如何选择这些子词单元添加到词汇表中？Sennrich 等人 [2016] 提出使用字节对编码（BPE；Gage, 1994），这是一种压缩算法，迭代地将最频繁的字节对替换（"合并"）为单个新的未使用索引。注意，此算法将子词词元添加到我们的词汇表中，以最大化输入序列的压缩——如果一个词在我们的输入文本中出现足够多次，它将被表示为单个子词单元。

通过 BPE 构建词汇表的子词分词器通常称为 BPE 分词器。在本作业中，我们将实现字节级 BPE 分词器，其中词汇项是字节或合并的字节序列，这让我们在词汇外处理和可管理的输入序列长度两方面都能获得最佳效果。构建 BPE 分词器词汇表的过程称为"训练"BPE 分词器。

### 2.4 BPE 分词器训练

BPE 分词器训练过程包括三个主要步骤。

#### 词汇表初始化

分词器词汇表是从字节字符串词元到整数 ID 的一对一映射。由于我们训练的是字节级 BPE 分词器，我们的初始词汇表就是所有字节的集合。由于有 256 个可能的字节值，我们的初始词汇表大小为 256。

#### 预分词

一旦你有了词汇表，原则上你可以统计字节在文本中相邻出现的频率，并从最频繁的字节对开始合并。然而，这在计算上非常昂贵，因为每次合并我们都需要对语料库进行一次完整遍历。此外，直接跨语料库合并字节可能会导致仅在标点符号上不同的词元（例如 `dog!` 与 `dog.`）。这些词元会得到完全不同的词元 ID，尽管它们可能具有很高的语义相似性（因为它们只在标点符号上不同）。

为了避免这种情况，我们对语料库进行预分词。你可以将其视为对语料库进行的粗粒度分词，帮助我们统计字符对出现的频率。例如，词 `'text'` 可能是一个预词元，出现了 10 次。在这种情况下，当我们统计字符 't' 和 'e' 相邻出现的频率时，我们会看到词 'text' 中 't' 和 'e' 相邻，可以将它们的计数增加 10，而不是遍历整个语料库。由于我们训练的是字节级 BPE 模型，每个预词元都表示为 UTF-8 字节序列。

Sennrich 等人 [2016] 的原始 BPE 实现通过简单地在空白处分割进行预分词（即 `s.split(" ")`）。相比之下，我们将使用基于正则表达式的预分词器（GPT-2 使用；Radford et al., 2019），来自 [github.com/openai/tiktoken/pull/234/files](https://github.com/openai/tiktoken/pull/234/files)：

```python
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

使用此预分词器交互式地分割一些文本可能有助于更好地理解其行为：

```python
>>> # 需要 `regex` 包
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

在代码中使用它时，你应该使用 `re.finditer` 来避免在构建从预词元到其计数的映射时存储预分词的词。

#### 计算 BPE 合并

现在我们已经将输入文本转换为预词元，并将每个预词元表示为 UTF-8 字节序列，我们可以计算 BPE 合并（即训练 BPE 分词器）。

在高层次上，BPE 算法迭代地统计每对字节并识别频率最高的对 ("A", "B")。然后，这个最频繁的对 ("A", "B") 的每次出现都会被合并，即替换为新的词元 "AB"。这个新的合并词元被添加到我们的词汇表中；因此，BPE 训练后的最终词汇表大小是初始词汇表大小（在我们的情况下是 256）加上 BPE 训练期间执行的合并操作数量。为了在 BPE 训练期间提高效率，我们不考虑跨越预词元边界的对²。在计算合并时，通过优先选择字典序更大的对来确定性打破对频率的平局。例如，如果对 ("A", "B")、("A", "C")、("B", "ZZ") 和 ("BA", "A") 都具有最高频率，我们将合并 ("BA", "A")：

```python
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
```

#### 特殊词元

通常，一些字符串（例如 `<|endoftext|>`）用于编码元数据（例如文档之间的边界）。在编码文本时，通常希望将某些字符串视为"特殊词元"，这些词元永远不应被分割成多个词元（即始终保留为单个词元）。例如，序列结束字符串 `<|endoftext|>` 应始终保留为单个词元（即单个整数 ID），这样我们就知道何时停止从语言模型生成。这些特殊词元必须添加到词汇表中，以便它们有相应的固定词元 ID。

Sennrich 等人 [2016] 的算法 1 包含一个低效的 BPE 分词器训练实现（本质上遵循我们上面概述的步骤）。作为第一个练习，实现并测试此函数以检验你的理解可能会有帮助。

---

**示例（bpe_example）：BPE 训练示例**

这是来自 Sennrich 等人 [2016] 的一个风格化示例。考虑一个包含以下文本的语料库：

```
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```

词汇表有一个特殊词元 `<|endoftext|>`。

**词汇表** 我们用特殊词元 `<|endoftext|>` 和 256 个字节值初始化我们的词汇表。

**预分词** 为简单起见并专注于合并过程，我们在此示例中假设预分词只是按空白分割。当我们预分词并计数时，我们得到频率表：

```
{low: 5, lower: 2, widest: 3, newest: 6}
```

将其表示为 `dict[tuple[bytes], int]` 很方便，例如 `{(l,o,w): 5 ...}`。注意，即使在 Python 中单个字节也是 bytes 对象。Python 中没有 byte 类型来表示单个字节，就像没有 char 类型来表示单个字符一样。

**合并** 我们首先查看每对连续字节，并求和它们出现的词的频率 `{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}`。对 ('es') 和 ('st') 平局，因此我们取字典序更大的对 ('st')。然后我们将合并预词元，得到 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}`。

在第二轮中，我们看到 (e, st) 是最常见的对（计数为 9），我们将合并为 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}`。继续这个过程，我们最终得到的合并序列将是 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']`。

如果我们取 6 次合并，我们有 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']`，我们的词汇表元素将是 `[<|endoftext|>, [...256 字节字符], st, est, ow, low, west, ne]`。

使用这个词汇表和合并集，词 newest 将分词为 `[ne, west]`。

---

### 2.5 BPE 分词器训练实验

让我们在 TinyStories 数据集上训练一个字节级 BPE 分词器。查找/下载数据集的说明见第 1 节。开始前，我们建议查看 TinyStories 数据集以了解数据内容。

#### 并行化预分词

你会发现预分词步骤是一个主要瓶颈。你可以使用内置库 `multiprocessing` 并行化代码来加速预分词。具体而言，我们建议在预分词的并行实现中，对语料库进行分块，同时确保分块边界出现在特殊词元的开头。你可以自由使用以下链接的入门代码来获取分块边界，然后你可以使用这些边界在进程间分配工作：

[https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py)

这种分块总是有效的，因为我们永远不想跨文档边界合并。就本作业而言，你始终可以以这种方式分割。不必担心接收一个不包含 `<|endoftext|>` 的超大语料库的边缘情况。

#### 预分词前移除特殊词元

在使用正则表达式模式运行预分词（使用 `re.finditer`）之前，你应该从语料库（或你的分块，如果使用并行实现）中剥离所有特殊词元。确保你在特殊词元上分割，以便不会发生跨越它们所界定文本的合并。例如，如果你的语料库（或分块）像 `[Doc 1]<|endoftext|>[Doc 2]`，你应该在特殊词元 `<|endoftext|>` 上分割，并分别预分词 `[Doc 1]` 和 `[Doc 2]`，以便不会发生跨越文档边界的合并。这可以使用 `re.split` 并以 `"|".join(special_tokens)` 作为分隔符来完成（小心使用 `re.escape`，因为 `|` 可能出现在特殊词元中）。测试 `test_train_bpe_special_tokens` 将测试这一点。

#### 优化合并步骤

上面风格化示例中 BPE 训练的朴素实现很慢，因为对于每次合并，它都会迭代所有字节对以识别最频繁的对。然而，每次合并后唯一改变的对计数是那些与合并对重叠的。因此，可以通过索引所有对的计数并增量更新这些计数来提高 BPE 训练速度，而不是显式迭代每对字节来统计对频率。通过这种缓存过程可以获得显著的加速，尽管我们注意到 BPE 训练的合并部分在 Python 中无法并行化。

---

> **💡 低资源/降规模提示：性能分析**
>
> 你应该使用 cProfile 或 scalene 等性能分析工具来识别实现中的瓶颈，并专注于优化这些部分。

---

> **💡 低资源/降规模提示："降规模"**
>
> 我们建议不要直接在完整的 TinyStories 数据集上训练分词器，而是先在数据的一个小子集上训练："调试数据集"。例如，你可以在 TinyStories 验证集上训练你的分词器，验证集有 22K 个文档，而不是 2.12M 个。这说明了尽可能降规模以加速开发的一般策略：例如，使用更小的数据集、更小的模型大小等。选择调试数据集的大小或超参数配置需要仔细考虑：你希望你的调试集足够大，具有与完整配置相同的瓶颈（这样你做的优化可以推广），但又不要太大以至于运行时间过长。

---

**问题（train_bpe）：BPE 分词器训练（15分）**

**交付物**：编写一个函数，给定输入文本文件的路径，训练一个（字节级）BPE 分词器。你的 BPE 训练函数应处理（至少）以下输入参数：

- `input_path: str` - BPE 分词器训练数据的文本文件路径。
- `vocab_size: int` - 定义最大最终词汇表大小的正整数（包括初始字节词汇表、合并产生的词汇项以及任何特殊词元）。
- `special_tokens: list[str]` - 要添加到词汇表的字符串列表。这些特殊词元不会影响 BPE 训练。

你的 BPE 训练函数应返回生成的词汇表和合并：

- `vocab: dict[int, bytes]` - 分词器词汇表，从 int（词汇表中的词元 ID）到 bytes（词元字节）的映射。
- `merges: list[tuple[bytes, bytes]]` - BPE 训练产生的合并列表。每个列表项是一个字节元组 (`<token1>`, `<token2>`)，表示 `<token1>` 与 `<token2>` 合并。合并应按创建顺序排序。

要针对我们提供的测试测试你的 BPE 训练函数，你首先需要实现 [adapters.run_train_bpe] 处的测试适配器。然后，运行 `uv run pytest tests/test_train_bpe.py`。

你的实现应该能够通过所有测试。可选地（这可能需要大量时间投入），你可以使用某些系统语言实现训练方法的关键部分，例如 C++（考虑使用 cppyy）或 Rust（使用 PyO3）。如果你这样做，请注意哪些操作需要复制与直接从 Python 内存读取，并确保留下构建说明，或确保它仅使用 pyproject.toml 构建。另请注意，GPT-2 正则表达式在大多数正则表达式引擎中支持不佳，在大多数支持的引擎中速度会很慢。我们已经验证 Oniguruma 速度合理且支持负向前瞻，但 Python 中的 regex 包（如果有的话）甚至更快。

**问题（train_bpe_tinystories）：在 TinyStories 上训练 BPE（2分）**

(a) 在 TinyStories 数据集上训练一个字节级 BPE 分词器，使用最大词汇表大小 10,000。确保将 TinyStories `<|endoftext|>` 特殊词元添加到词汇表中。将生成的词汇表和合并序列化到磁盘以供进一步检查。训练花了多少小时和内存？词汇表中最长的词元是什么？它有意义吗？

**资源需求**：≤ 30 分钟（无需 GPU），≤ 30GB RAM

**提示**：你应该能够在 2 分钟内完成 BPE 训练，在预分词期间使用多进程，并利用以下两个事实：
- (a) `<|endoftext|>` 词元分隔数据文件中的文档。
- (b) `<|endoftext|>` 词元在应用 BPE 合并之前作为特殊情况处理。

**交付物**：一到两句话回答。

(b) 分析你的代码。分词器训练过程的哪个部分耗时最多？  
**交付物**：一到两句话回答。

接下来，我们将尝试在 OpenWebText 数据集上训练字节级 BPE 分词器。与之前一样，我们建议查看数据集以更好地了解其内容。

**问题（train_bpe_expts_owt）：在 OpenWebText 上训练 BPE（2分）**

(a) 在 OpenWebText 数据集上训练一个字节级 BPE 分词器，使用最大词汇表大小 32,000。将生成的词汇表和合并序列化到磁盘以供进一步检查。词汇表中最长的词元是什么？它有意义吗？

**资源需求**：≤ 12 小时（无需 GPU），≤ 100GB RAM

**交付物**：一到两句话回答。

(b) 比较和对比你在 TinyStories 与 OpenWebText 上训练得到的分词器。  
**交付物**：一到两句话回答。

### 2.6 BPE 分词器：编码和解码

在作业的前一部分，我们实现了一个函数来在输入文本上训练 BPE 分词器，以获得分词器词汇表和 BPE 合并列表。现在，我们将实现一个 BPE 分词器，加载提供的词汇表和合并列表，并使用它们将文本编码/解码为/从词元 ID。

#### 2.6.1 编码文本

BPE 编码文本的过程反映了我们如何训练 BPE 词汇表。有几个主要步骤。

**步骤 1：预分词**。我们首先预分词序列，并将每个预词元表示为 UTF-8 字节序列，就像我们在 BPE 训练中做的那样。我们将在每个预词元内将这些字节合并为词汇元素，独立处理每个预词元（不跨预词元边界合并）。

**步骤 2：应用合并**。然后我们获取 BPE 训练期间创建的词汇元素合并序列，并按创建顺序将其应用于我们的预词元。

---

**示例（bpe_encoding）：BPE 编码示例**

例如，假设我们的输入字符串是 `'the cat ate'`，我们的词汇表是 `{0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}`，我们学到的合并是 `[(b't', b'h'), (b' ', b'c'), (b' ', 'a'), (b'th', b'e'), (b' a', b't')]`。

首先，我们的预分词器会将此字符串分割为 `['the', ' cat', ' ate']`。

然后，我们将查看每个预词元并应用 BPE 合并。

第一个预词元 `'the'` 最初表示为 `[b't', b'h', b'e']`。查看我们的合并列表，我们识别出第一个适用的合并是 `(b't', b'h')`，并使用它将预词元转换为 `[b'th', b'e']`。然后，我们回到合并列表并识别下一个适用的合并是 `(b'th', b'e')`，它将预词元转换为 `[b'the']`。最后，查看合并列表，我们看到没有更多适用于该字符串的合并（因为整个预词元已合并为单个词元），所以我们完成了 BPE 合并的应用。相应的整数序列是 `[9]`。

对剩余的预词元重复此过程，我们看到预词元 `' cat'` 在应用 BPE 合并后表示为 `[b' c', b'a', b't']`，成为整数序列 `[7, 1, 5]`。最后一个预词元 `' ate'` 在应用 BPE 合并后是 `[b' at', b'e']`，成为整数序列 `[10, 3]`。因此，编码输入字符串的最终结果是 `[9, 7, 1, 5, 10, 3]`。

**特殊词元**。你的分词器应该能够在编码文本时正确处理用户定义的特殊词元（在构建分词器时提供）。

**内存考虑**。假设我们想分词一个无法放入内存的大型文本文件。为了高效地分词这个大型文件（或任何其他数据流），我们需要将其分解为可管理的块并依次处理每个块，以便内存复杂度是常数而不是与文本大小成线性关系。这样做时，我们需要确保词元不跨越块边界，否则我们会得到与在内存中分词整个序列的朴素方法不同的分词结果。

#### 2.6.2 解码文本

要将整数词元 ID 序列解码回原始文本，我们可以简单地查找词汇表中每个 ID 的相应条目（字节序列），将它们连接起来，然后将字节解码为 Unicode 字符串。注意，输入 ID 不保证映射到有效的 Unicode 字符串（因为用户可以输入任何整数 ID 序列）。

在输入词元 ID 不产生有效 Unicode 字符串的情况下，你应该用官方 Unicode 替换字符 U+FFFD 替换格式错误的字节³。`bytes.decode` 的 `errors` 参数控制如何处理 Unicode 解码错误，使用 `errors='replace'` 将自动用替换标记替换格式错误的数据。

**问题（tokenizer）：实现分词器（15分）**

**交付物**：实现一个 `Tokenizer` 类，给定词汇表和合并列表，将文本编码为整数 ID，并将整数 ID 解码为文本。你的分词器还应支持用户提供的特殊词元（如果它们尚未在词汇表中，则将其附加到词汇表）。我们建议以下接口：

```python
def __init__(self, vocab, merges, special_tokens=None)
```

从给定的词汇表、合并列表和（可选的）特殊词元列表构建分词器。此函数应接受以下参数：
- `vocab: dict[int, bytes]`
- `merges: list[tuple[bytes, bytes]]`
- `special_tokens: list[str] | None = None`

```python
def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)
```

类方法，从序列化的词汇表和合并列表（与你的 BPE 训练代码输出的格式相同）以及（可选的）特殊词元列表构建并返回 Tokenizer。此方法应接受以下附加参数：
- `vocab_filepath: str`
- `merges_filepath: str`
- `special_tokens: list[str] | None = None`

```python
def encode(self, text: str) -> list[int]
```

将输入文本编码为词元 ID 序列。

```python
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]
```

给定字符串的可迭代对象（例如 Python 文件句柄），返回一个惰性产生词元 ID 的生成器。这对于无法直接加载到内存的大型文件的内存高效分词是必需的。

```python
def decode(self, ids: list[int]) -> str
```

将词元 ID 序列解码为文本。

要针对我们提供的测试测试你的 Tokenizer，你首先需要实现 [adapters.get_tokenizer] 处的测试适配器。然后，运行 `uv run pytest tests/test_tokenizer.py`。你的实现应该能够通过所有测试。

### 2.7 实验

**问题（tokenizer_experiments）：分词器实验（4分）**

(a) 从 TinyStories 和 OpenWebText 中各采样 10 个文档。使用你之前训练的 TinyStories 和 OpenWebText 分词器（分别为 10K 和 32K 词汇表大小），将这些采样的文档编码为整数 ID。每个分词器的压缩比（字节/词元）是多少？  
**交付物**：一到两句话回答。

(b) 如果你用 TinyStories 分词器分词你的 OpenWebText 样本会发生什么？比较压缩比和/或定性地描述发生了什么。  
**交付物**：一到两句话回答。

(c) 估计你的分词器吞吐量（例如，以字节/秒为单位）。分词 Pile 数据集（825GB 文本）需要多长时间？  
**交付物**：一到两句话回答。

(d) 使用你的 TinyStories 和 OpenWebText 分词器，将各自的训练和开发数据集编码为整数词元 ID 序列。我们稍后将使用此来训练我们的语言模型。我们建议将词元 ID 序列化为 NumPy 数组，数据类型为 `uint16`。为什么 `uint16` 是一个合适的选择？  
**交付物**：一到两句话回答。

---

## 3. Transformer 语言模型架构

语言模型将一批整数词元 ID 序列（即形状为 `(batch_size, sequence_length)` 的 `torch.Tensor`）作为输入，并返回（批量的）词汇表上的归一化概率分布（即形状为 `(batch_size, sequence_length, vocab_size)` 的 PyTorch Tensor），其中预测分布是针对每个输入词元的下一个词。在训练语言模型时，我们使用这些下一个词预测来计算实际下一个词与预测下一个词之间的交叉熵损失。在推理期间从语言模型生成文本时，我们从最后一个时间步（即序列中的最后一项）获取预测的下一个词分布来生成序列中的下一个词元（例如，通过取概率最高的词元、从分布中采样等），将生成的词元添加到输入序列，然后重复。

在本部分作业中，你将从零开始构建这个 Transformer 语言模型。我们将从模型的高级描述开始，然后逐步详细介绍各个组件。

### 3.1 Transformer LM

给定一个词元 ID 序列，Transformer 语言模型使用输入嵌入将词元 ID 转换为密集向量，将嵌入的词元通过 `num_layers` 个 Transformer 块，然后应用学习的线性投影（"输出嵌入"或"LM 头"）以产生预测的下一个词元对数几率。参见图 1 的示意图。

#### 3.1.1 词元嵌入

在第一步中，Transformer 将（批量的）词元 ID 序列嵌入为包含词元身份信息的向量序列（图 1 中的红色块）。

更具体地说，给定一个词元 ID 序列，Transformer 语言模型使用词元嵌入层产生向量序列。每个嵌入层接收形状为 `(batch_size, sequence_length)` 的整数张量，并产生形状为 `(batch_size, sequence_length, d_model)` 的向量序列。

#### 3.1.2 Pre-norm Transformer 块

嵌入后，激活值由几个结构相同的神经网络层处理。标准仅解码器 Transformer 语言模型由 `num_layers` 个相同的层组成（通常称为 Transformer"块"）。每个 Transformer 块接收形状为 `(batch_size, sequence_length, d_model)` 的输入，并返回形状为 `(batch_size, sequence_length, d_model)` 的输出。每个块通过自注意力聚合序列中的信息，并通过前馈层进行非线性变换。

### 3.2 输出归一化和嵌入

经过 `num_layers` 个 Transformer 块后，我们将获取最终激活值并将其转换为词汇表上的分布。

我们将实现"pre-norm" Transformer 块（详见 §3.5），它还需要在最终 Transformer 块后使用层归一化（详见下文）以确保其输出正确缩放。

在此归一化之后，我们将使用标准学习线性变换将 Transformer 块的输出转换为预测的下一个词元对数几率（参见例如 Radford et al. [2018] 方程 2）。

### 3.3 备注：批处理、Einsum 和高效计算

在整个 Transformer 中，我们将对许多类似批量的输入执行相同的计算。以下是一些示例：

- 批量的元素：我们对每个批量元素应用相同的 Transformer 前向操作。
- 序列长度："位置级"操作如 RMSNorm 和前馈对每个序列位置的操作相同。
- 注意力头：注意力操作在"多头"注意力操作中跨注意力头进行批处理。

拥有一种符合人体工程学的方式来执行此类操作，充分利用 GPU，并且易于阅读和理解是很有用的。许多 PyTorch 操作可以接受张量开头多余的"类似批量"维度，并高效地重复/广播这些维度上的操作。

例如，假设我们正在执行位置级、批处理操作。我们有一个"数据张量" D，形状为 `(batch_size, sequence_length, d_model)`，我们想对一个形状为 `(d_model, d_model)` 的矩阵 A 执行批处理向量-矩阵乘法。在这种情况下，`D @ A` 将执行批处理矩阵乘法，这是 PyTorch 中的高效原语，其中 `(batch_size, sequence_length)` 维度被批处理。

因此，假设你的函数可能被给予额外的类似批量的维度，并将这些维度保留在 PyTorch 形状的开头是很有帮助的。为了以这种方式组织张量以便可以批处理，它们可能需要使用许多 `view`、`reshape` 和 `transpose` 步骤进行整形。这可能有点痛苦，而且通常很难阅读代码在做什么以及张量的形状是什么。

一个更符合人体工程学的选择是在 `torch.einsum` 中使用 einsum 表示法，或者使用框架无关的库如 `einops` 或 `einx`。两个关键操作是 `einsum`，它可以对输入张量的任意维度进行张量收缩，以及 `rearrange`，它可以重新排序、连接和拆分任意维度。

> 事实证明，机器学习中的几乎所有操作都是维度调整和张量收缩的某种组合，偶尔（通常是逐点）非线性函数。这意味着当你使用 einsum 表示法时，你的很多代码可以更易读和灵活。

我们强烈建议学习并使用 einsum 表示法来完成本课程。之前没有接触过 einsum 表示法的学生应该使用 einops（文档[在此](https://einops.rocks/)），已经熟悉 einops 的学生应该学习更通用的 einx（[在此](https://github.com/fferflo/einx)）。两个包都已安装在我们提供的环境中。

---

> ⁴值得注意的是，虽然 einops 有大量支持，但 einx 并没有经过那么多实战测试。如果你发现 einx 有任何限制或错误，请随时回退到使用 einops 和一些普通 PyTorch。

---

### 3.3.1 数学符号和内存排序

许多机器学习论文在其符号中使用行向量，这导致与 NumPy 和 PyTorch 默认使用的行主序内存排序很好地配合的表示。使用行向量时，线性变换看起来像这样：

$$y = xW^\top, \quad (1)$$

对于行主序 $W \in \mathbb{R}^{d_{out} \times d_{in}}$ 和行向量 $x \in \mathbb{R}^{1 \times d_{in}}$。

在线性代数中，更常见的是使用列向量，其中线性变换看起来像这样：

$$y = Wx, \quad (2)$$

给定行主序 $W \in \mathbb{R}^{d_{out} \times d_{in}}$ 和列向量 $x \in \mathbb{R}^{d_{in}}$。我们将在本作业的数学符号中使用列向量，因为通常这样更容易理解数学。你应该记住，如果你想使用普通矩阵乘法符号，你将不得不应用行向量约定的矩阵，因为 PyTorch 使用行主序内存排序。如果你使用 einsum 进行矩阵操作，这应该不是问题。

### 3.4 基本构建块：线性和嵌入模块

#### 3.4.1 参数初始化

有效训练神经网络通常需要仔细初始化模型参数——糟糕的初始化可能导致不良行为，如梯度消失或爆炸。Pre-norm Transformer 对初始化异常稳健，但它们仍然会对训练速度和收敛产生重大影响。由于本作业已经很长了，我们将把细节留到作业 3，而是给你一些在大多数情况下都能很好工作的近似初始化。目前，使用：

- **线性权重**：$\mathcal{N}\left(\mu = 0, \sigma^2 = \frac{2}{d_{in}+d_{out}}\right)$，截断于 $[-3\sigma, 3\sigma]$。
- **嵌入**：$\mathcal{N}(\mu = 0, \sigma^2 = 1)$，截断于 $[-3, 3]$。
- **RMSNorm**：1

你应该使用 `torch.nn.init.trunc_normal_` 来初始化截断正态权重。

#### 3.4.2 线性模块

线性层是 Transformer 和神经网络的基本构建块。首先，你将实现自己的 `Linear` 类，继承自 `torch.nn.Module` 并执行线性变换：

$$y = Wx. \quad (3)$$

注意，我们不包含偏置项，遵循大多数现代 LLM。

**问题（linear）：实现线性模块（1分）**

**交付物**：实现一个继承自 `torch.nn.Module` 的 `Linear` 类，执行线性变换。你的实现应遵循 PyTorch 内置 `nn.Linear` 模块的接口，除了没有 `bias` 参数或参数。我们建议以下接口：

```python
def __init__(self, in_features, out_features, device=None, dtype=None)
```

构建线性变换模块。此函数应接受以下参数：
- `in_features: int` - 输入的最终维度
- `out_features: int` - 输出的最终维度
- `device: torch.device | None = None` - 存储参数的设备
- `dtype: torch.dtype | None = None` - 参数的数据类型

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

对输入应用线性变换。

确保：
- 继承 `nn.Module`
- 调用超类构造函数
- 将参数构造并存储为 W（不是 $W^\top$）以符合内存排序原因，将其放入 `nn.Parameter`
- 当然，不要使用 `nn.Linear` 或 `nn.functional.linear`

对于初始化，使用上面的设置以及 `torch.nn.init.trunc_normal_` 来初始化权重。

要测试你的 Linear 模块，在 [adapters.run_linear] 处实现测试适配器。适配器应将给定权重加载到你的 Linear 模块中。你可以为此目的使用 `Module.load_state_dict`。然后，运行 `uv run pytest -k test_linear`。

#### 3.4.3 嵌入模块

如上所述，Transformer 的第一层是将整数词元 ID 映射到维度为 $d_{model}$ 的向量空间的嵌入层。我们将实现一个自定义的 `Embedding` 类，继承自 `torch.nn.Module`（因此你不应该使用 `nn.Embedding`）。`forward` 方法应通过索引形状为 `(vocab_size, d_model)` 的嵌入矩阵，使用形状为 `(batch_size, sequence_length)` 的词元 ID `torch.LongTensor` 来选择每个词元 ID 的嵌入向量。

**问题（embedding）：实现嵌入模块（1分）**

**交付物**：实现继承自 `torch.nn.Module` 的 `Embedding` 类，执行嵌入查找。你的实现应遵循 PyTorch 内置 `nn.Embedding` 模块的接口。我们建议以下接口：

```python
def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)
```

构建嵌入模块。此函数应接受以下参数：
- `num_embeddings: int` - 词汇表大小
- `embedding_dim: int` - 嵌入向量的维度，即 $d_{model}$
- `device: torch.device | None = None` - 存储参数的设备
- `dtype: torch.dtype | None = None` - 参数的数据类型

```python
def forward(self, token_ids: torch.Tensor) -> torch.Tensor
```

查找给定词元 ID 的嵌入向量。

确保：
- 继承 `nn.Module`
- 调用超类构造函数
- 将嵌入矩阵初始化为 `nn.Parameter`
- 存储嵌入矩阵，使 $d_{model}$ 为最终维度
- 当然，不要使用 `nn.Embedding` 或 `nn.functional.embedding`

同样，使用上面的设置进行初始化，并使用 `torch.nn.init.trunc_normal_` 来初始化权重。

要测试你的实现，在 [adapters.run_embedding] 处实现测试适配器。然后，运行 `uv run pytest -k test_embedding`。

### 3.5 Pre-Norm Transformer 块

每个 Transformer 块有两个子层：多头自注意力机制和位置级前馈网络（Vaswani et al., 2017，第 3.1 节）。

在原始 Transformer 论文中，模型在每个子层周围使用残差连接，后跟层归一化。这种架构通常称为"post-norm" Transformer，因为层归一化应用于子层输出。然而，各种研究发现将层归一化从每个子层的输出移动到每个子层的输入（在最终 Transformer 块后添加额外的层归一化）可以提高 Transformer 训练稳定性 [Nguyen and Salazar, 2019, Xiong et al., 2020]——参见图 2 的"pre-norm" Transformer 块视觉表示。然后，每个 Transformer 块子层的输出通过残差连接添加到子层输入（Vaswani et al., 2017，第 5.4 节）。Pre-norm 的直觉是有一个从输入嵌入到 Transformer 最终输出的干净"残差流"，没有任何归一化，这被认为可以改善梯度流。这种 pre-norm Transformer 现在是当今语言模型中使用的标准（例如 GPT-3、LLaMA、PaLM 等），因此我们将实现这种变体。我们将按顺序实现 pre-norm Transformer 块的每个组件。

#### 3.5.1 均方根层归一化

Vaswani 等人 [2017] 的原始 Transformer 实现使用层归一化 [Ba et al., 2016] 来归一化激活值。遵循 Touvron 等人 [2023]，我们将使用均方根层归一化（RMSNorm；Zhang and Sennrich, 2019，方程 4）进行层归一化。给定激活向量 $a \in \mathbb{R}^{d_{model}}$，RMSNorm 将每个激活 $a_i$ 重新缩放如下：

$$\text{RMSNorm}(a_i) = \frac{a_i}{\text{RMS}(a)} g_i, \quad (4)$$

其中 $\text{RMS}(a) = \sqrt{\frac{1}{d_{model}} \sum_{i=1}^{d_{model}} a_i^2 + \varepsilon}$。这里，$g_i$ 是一个可学习的"增益"参数（总共有 $d_{model}$ 个这样的参数），$\varepsilon$ 是一个通常固定为 $1e-5$ 的超参数。

你应该将输入上转换为 `torch.float32` 以防止平方输入时溢出。总体而言，你的 `forward` 方法应该如下所示：

```python
in_dtype = x.dtype
x = x.to(torch.float32)
# 在这里执行 RMSNorm 的代码
...
result = ...
# 以原始数据类型返回结果
return result.to(in_dtype)
```

**问题（rmsnorm）：均方根层归一化（1分）**

**交付物**：将 RMSNorm 实现为 `torch.nn.Module`。我们建议以下接口：

```python
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)
```

构建 RMSNorm 模块。此函数应接受以下参数：
- `d_model: int` - 模型的隐藏维度
- `eps: float = 1e-5` - 数值稳定性 epsilon 值
- `device: torch.device | None = None` - 存储参数的设备
- `dtype: torch.dtype | None = None` - 参数的数据类型

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

处理形状为 `(batch_size, sequence_length, d_model)` 的输入张量，并返回相同形状的张量。

注意：记住在执行归一化之前将输入上转换为 `torch.float32`（稍后下转换为原始数据类型），如上所述。

要测试你的实现，在 [adapters.run_rmsnorm] 处实现测试适配器。然后，运行 `uv run pytest -k test_rmsnorm`。

#### 3.5.2 位置级前馈网络

在原始 Transformer 论文（Vaswani et al. [2017] 第 3.3 节）中，Transformer 前馈网络由两个线性变换组成，中间使用 ReLU 激活（$\text{ReLU}(x) = \max(0, x)$）。内部前馈层的维度通常是输入维度的 4 倍。

然而，与现代语言模型相比，现代语言模型倾向于对这种原始设计进行两项主要更改：它们使用另一种激活函数并采用门控机制。具体来说，我们将实现 Llama 3 [Grattafiori et al., 2024] 和 Qwen 2.5 [Yang et al., 2024] 等 LLM 采用的"SwiGLU"激活函数，它将 SiLU（通常称为 Swish）激活与称为门控线性单元（GLU）的门控机制相结合。我们还将省略线性层中有时使用的偏置项，遵循 PaLM [Chowdhery et al., 2022] 和 LLaMA [Touvron et al., 2023] 之后的大多数现代 LLM。

SiLU 或 Swish 激活函数 [Hendrycks and Gimpel, 2016, Elfwing et al., 2017] 定义如下：

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}} \quad (5)$$

如图 3 所示，SiLU 激活函数与 ReLU 激活函数相似，但在零点处是平滑的。

门控线性单元（GLU）最初由 Dauphin 等人 [2017] 定义为通过 sigmoid 函数的线性变换与另一个线性变换的逐元素乘积：

$$\text{GLU}(x, W_1, W_2) = \sigma(W_1x) \odot W_2x, \quad (6)$$

其中 $\odot$ 表示逐元素乘法。门控线性单元被认为可以"通过为梯度提供线性路径，同时保留非线性能力，来减少深层架构的梯度消失问题。"

将 SiLU/Swish 和 GLU 结合起来，我们得到 SwiGLU，我们将用于我们的前馈网络：

$$\text{FFN}(x) = \text{SwiGLU}(x, W_1, W_2, W_3) = W_2(\text{SiLU}(W_1x) \odot W_3x), \quad (7)$$

其中 $x \in \mathbb{R}^{d_{model}}$，$W_1, W_3 \in \mathbb{R}^{d_{ff} \times d_{model}}$，$W_2 \in \mathbb{R}^{d_{model} \times d_{ff}}$，规范地，$d_{ff} = \frac{8}{3} d_{model}$。

Shazeer [2020] 首先提出将 SiLU/Swish 激活与 GLU 结合，并进行了实验，表明 SwiGLU 在语言建模任务上优于 ReLU 和 SiLU（无门控）等基线。在本作业后面，你将比较 SwiGLU 和 SiLU。虽然我们提到了这些组件的一些启发式论点（论文提供了更多支持证据），但保持实证视角是很好的：Shazeer 论文中现在著名的引用是：

> "我们对这些架构为什么有效没有解释；我们将它们的成功，以及其他一切，归因于神圣的仁慈。"

**问题（positionwise_feedforward）：实现位置级前馈网络（2分）**

**交付物**：实现 SwiGLU 前馈网络，由 SiLU 激活函数和 GLU 组成。

注意：在这种特殊情况下，你应该可以在实现中使用 `torch.sigmoid` 以获得数值稳定性。

在你的实现中，你应该将 $d_{ff}$ 设置为大约 $\frac{8}{3} \times d_{model}$，同时确保内部前馈层的维度是 64 的倍数以充分利用你的硬件。要针对我们提供的测试测试你的实现，你需要在 [adapters.run_swiglu] 处实现测试适配器。然后，运行 `uv run pytest -k test_swiglu` 来测试你的实现。

#### 3.5.3 相对位置嵌入

为了将位置信息注入模型，我们将实现旋转位置嵌入 [Su et al., 2021]，通常称为 RoPE。对于给定的查询词元 $q^{(i)} = W_q x^{(i)} \in \mathbb{R}^d$ 在词元位置 $i$，我们将应用成对旋转矩阵 $R_i$，得到 $q'^{(i)} = R_i q^{(i)} = R_i W_q x^{(i)}$。这里，$R_i$ 将以角度 $\theta_{i,k} = \Theta^{(2k-2)/d}$ 旋转 $2d$ 向量，其中 $k \in \{1, \ldots, d/2\}$，$\Theta$ 是某个常数。因此，我们可以将 $R_i$ 视为大小为 $d \times d$ 的块对角矩阵，块 $R_i^k$ 对于 $k \in \{1, \ldots, d/2\}$，其中：

$$R_i^k = \begin{bmatrix} \cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\ \sin(\theta_{i,k}) & \cos(\theta_{i,k}) \end{bmatrix}. \quad (8)$$

因此我们得到完整的旋转矩阵：

$$R_i = \begin{bmatrix} R_i^1 & 0 & 0 & \cdots & 0 \\ 0 & R_i^2 & 0 & \cdots & 0 \\ 0 & 0 & R_i^3 & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & R_i^{d/2} \end{bmatrix}, \quad (9)$$

其中 0 表示 $2 \times 2$ 零矩阵。虽然可以构建完整的 $d \times d$ 矩阵，但一个好的解决方案应该利用这个矩阵的属性来更高效地实现变换。由于我们只关心给定序列内词元的相对旋转，我们可以在不同层和不同批次之间重用我们为 $\cos(\theta_{i,k})$ 和 $\sin(\theta_{i,k})$ 计算的值。如果你想优化它，你可以使用所有层引用的单个 RoPE 模块，它可以在 `__init__` 期间创建一个 $2d$ 预计算的 sin 和 cos 值缓冲区，使用 `self.register_buffer(persistent=False)`，而不是 `nn.Parameter`（因为我们不想学习这些固定的余弦和正弦值）。然后对 $k^{(j)}$ 执行与我们对 $q^{(i)}$ 执行的完全相同的旋转过程，旋转相应的 $R_j$。注意，这一层没有可学习参数。

**问题（rope）：实现 RoPE（2分）**

**交付物**：实现一个 `RotaryPositionalEmbedding` 类，将 RoPE 应用于输入张量。建议以下接口：

```python
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)
```

构建 RoPE 模块，并在需要时创建缓冲区。
- `theta: float` - RoPE 的 $\Theta$ 值
- `d_k: int` - 查询和键向量的维度
- `max_seq_len: int` - 将输入的最大序列长度
- `device: torch.device | None = None` - 存储缓冲区的设备

```python
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
```

处理形状为 `(..., seq_len, d_k)` 的输入张量，并返回相同形状的张量。注意，你应该容忍具有任意数量批量维度的 x。你应该假设词元位置是一个形状为 `(..., seq_len)` 的张量，指定 x 沿序列维度的词元位置。

你应该使用词元位置沿序列维度切片你的（可能预计算的）cos 和 sin 张量。

要测试你的实现，完成 [adapters.run_rope] 并确保它通过 `uv run pytest -k test_rope`。

#### 3.5.4 缩放点积注意力

我们现在将实现 Vaswani 等人 [2017]（第 3.2.1 节）中描述的缩放点积注意力。

作为预备步骤，注意力操作的定义将使用 softmax，这是一种将非归一化分数向量转换为归一化分布的操作：

$$\text{softmax}(v)_i = \frac{\exp(v_i)}{\sum_{j=1}^{n} \exp(v_j)}. \quad (10)$$

注意，对于大值，$\exp(v_i)$ 可能变为 inf（然后，inf/inf = NaN）。我们可以通过注意到 softmax 操作对所有输入加上任何常数 c 都是不变的来避免这一点。我们可以利用这个属性来获得数值稳定性——通常，我们会从 $o_i$ 的所有元素中减去 $o_i$ 的最大元素，使新的最大元素为 0。你现在将实现 softmax，使用这个技巧来获得数值稳定性。

**问题（softmax）：实现 softmax（1分）**

**交付物**：编写一个函数来对张量应用 softmax 操作。你的函数应该接受两个参数：一个张量和一个维度 i，并对输入张量的第 i 维应用 softmax。输出张量应与输入张量具有相同的形状，但其第 i 维现在将具有归一化概率分布。使用从第 i 维的所有元素中减去第 i 维的最大值来避免数值稳定性问题。

要测试你的实现，完成 [adapters.run_softmax] 并确保它通过 `uv run pytest -k test_softmax_matches_pytorch`。

我们现在可以数学地定义注意力操作如下：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q^\top K}{\sqrt{d_k}}\right) V \quad (11)$$

其中 $Q \in \mathbb{R}^{n \times d_k}$，$K \in \mathbb{R}^{m \times d_k}$，$V \in \mathbb{R}^{m \times d_v}$。这里，Q、K 和 V 都是此操作的输入——注意这些不是可学习参数。如果你想知道为什么不是 $QK^\top$，请参见 3.3.1。

**掩码**：有时掩码注意力操作的输出会很方便。掩码应该具有形状 $M \in \{\text{True}, \text{False}\}^{n \times m}$，这个布尔矩阵的每一行 i 表示查询 i 应该关注哪些键。规范地（有点令人困惑），位置 $(i, j)$ 的 True 值表示查询 i 确实关注键 j，False 值表示查询不关注键。换句话说，"信息流"发生在值为 True 的 $(i, j)$ 对。例如，考虑一个条目为 `[[True, True, False]]` 的 $1 \times 3$ 掩码矩阵。单个查询向量只关注前两个键。

在计算上，使用掩码比计算子序列的注意力要高效得多，我们可以通过取 softmax 前的值 $\left(\frac{Q^\top K}{\sqrt{d_k}}\right)$ 并在掩码矩阵为 False 的任何条目中添加 $-\infty$ 来做到这一点。

**问题（scaled_dot_product_attention）：实现缩放点积注意力（5分）**

**交付物**：实现缩放点积注意力函数。你的实现应该处理形状为 `(batch_size, ..., seq_len, d_k)` 的键和查询，以及形状为 `(batch_size, ..., seq_len, d_v)` 的值，其中 `...` 表示任何其他类似批量的维度（如果提供）。实现应返回形状为 `(batch_size, ..., d_v)` 的输出。参见第 3.3 节关于类似批量维度的讨论。

你的实现还应支持可选的用户提供的布尔掩码，形状为 `(seq_len, seq_len)`。掩码值为 True 的位置的注意力概率总和应为 1，掩码值为 False 的位置的注意力概率应为零。

要针对我们提供的测试测试你的实现，你需要在 [adapters.run_scaled_dot_product_attention] 处实现测试适配器。`uv run pytest -k test_scaled_dot_product_attention` 在第三阶输入张量上测试你的实现，而 `uv run pytest -k test_4d_scaled_dot_product_attention` 在第四阶输入张量上测试你的实现。

#### 3.5.5 因果多头自注意力

我们将实现 Vaswani 等人 [2017] 第 3.2.2 节中描述的多头自注意力。回想一下，数学上应用多头注意力的操作定义如下：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \quad (12)$$
$$\text{其中 } \text{head}_i = \text{Attention}(Q_i, K_i, V_i) \quad (13)$$

其中 $Q_i, K_i, V_i$ 是嵌入维度上 Q、K 和 V 的第 $i \in \{1, \ldots, h\}$ 个切片，大小为 $d_k$ 或 $d_v$。Attention 是 §3.5.4 中定义的缩放点积注意力操作。由此我们可以形成多头自注意力操作：

$$\text{MultiHeadSelfAttention}(x) = W^O \text{MultiHead}(W^Q x, W^K x, W^V x) \quad (14)$$

这里，可学习参数是 $W^Q \in \mathbb{R}^{hd_k \times d_{model}}$，$W^K \in \mathbb{R}^{hd_k \times d_{model}}$，$W^V \in \mathbb{R}^{hd_v \times d_{model}}$，和 $W^O \in \mathbb{R}^{d_{model} \times hd_v}$。由于 Q、K 和 V 在多头注意力操作中被切片，我们可以认为 $W^Q$、$W^K$ 和 $W^V$ 沿输出维度为每个头分开。当你完成这个工作时，你应该总共计算三个矩阵乘法（键、值和查询投影）。⁵

---

> ⁵作为延伸目标，尝试将键、查询和值投影组合成单个权重矩阵，这样你只需要单个矩阵乘法。

---

**因果掩码**。你的实现应该防止模型关注序列中的未来词元。换句话说，如果模型被给予词元序列 $t_1, \ldots, t_n$，我们想计算前缀 $t_1, \ldots, t_i$（其中 $i < n$）的下一个词预测，模型不应该能够访问位置 $t_{i+1}, \ldots, t_n$ 的词元表示，因为在推理期间生成文本时它无法访问这些词元（而且这些未来词元会泄露关于真实下一个词身份的信息，使语言建模预训练目标变得平凡）。对于输入词元序列 $t_1, \ldots, t_n$，我们可以通过运行多头自注意力 n 次（对于序列中的 n 个唯一前缀）来朴素地防止访问未来词元。相反，我们将使用因果注意力掩码，允许词元 i 关注序列中的所有位置 $j \leq i$。你可以使用 `torch.triu` 或广播索引比较来构建此掩码，你应该利用你在 §3.5.4 中的缩放点积注意力实现已经支持注意力掩码的事实。

**应用 RoPE**。RoPE 应该应用于查询和键向量，但不应用于值向量。此外，头维度应作为批量维度处理，因为在多头注意力中，注意力是独立应用于每个头的。这意味着相同的 RoPE 旋转应该应用于每个头的查询和键向量。

**问题（multihead_self_attention）：实现因果多头自注意力（5分）**

**交付物**：将因果多头自注意力实现为 `torch.nn.Module`。你的实现应接受（至少）以下参数：
- `d_model: int` - Transformer 块输入的维度。
- `num_heads: int` - 多头自注意力中使用的头数。

遵循 Vaswani 等人 [2017]，设置 $d_k = d_v = d_{model}/h$。要针对我们提供的测试测试你的实现，在 [adapters.run_multihead_self_attention] 处实现测试适配器。然后，运行 `uv run pytest -k test_multihead_self_attention` 来测试你的实现。

### 3.6 完整的 Transformer LM

让我们开始组装 Transformer 块（参考图 2 会很有帮助）。一个 Transformer 块包含两个"子层"，一个用于多头自注意力，另一个用于前馈网络。在每个子层中，我们首先执行 RMSNorm，然后是主操作（MHA/FF），最后通过残差连接添加。

具体而言，Transformer 块的第一半（第一个"子层"）应该实现以下一组更新，从输入 x 产生输出 y：

$$y = x + \text{MultiHeadSelfAttention}(\text{RMSNorm}(x)). \quad (15)$$

**问题（transformer_block）：实现 Transformer 块（3分）**

**交付物**：实现 §3.5 中描述并在图 2 中说明的 pre-norm Transformer 块。你的 Transformer 块应接受（至少）以下参数：
- `d_model: int` - Transformer 块输入的维度。
- `num_heads: int` - 多头自注意力中使用的头数。
- `d_ff: int` - 位置级前馈内层的维度。

要测试你的实现，在 [adapters.run_transformer_block] 处实现适配器。然后运行 `uv run pytest -k test_transformer_block` 来测试你的实现。

**交付物**：通过所提供测试的 Transformer 块代码。

现在我们将这些块组合在一起，遵循图 1 的高级图。按照我们在第 3.1.1 节中的嵌入描述，将其输入到 `num_layers` 个 Transformer 块中，然后将其传递到三个输出层以获得词汇表上的分布。

**问题（transformer_lm）：实现 Transformer LM（3分）**

是时候把所有东西组合在一起了！

**交付物**：实现 §3.1 中描述并在图 1 中说明的 Transformer 语言模型。至少，你的实现应接受 Transformer 块的所有上述构造参数，以及这些附加参数：
- `vocab_size: int` - 词汇表大小，用于确定词元嵌入矩阵的维度。
- `context_length: int` - 最大上下文长度，用于确定位置嵌入矩阵的维度。
- `num_layers: int` - 要使用的 Transformer 块数量。

要针对我们提供的测试测试你的实现，你首先需要实现 [adapters.run_transformer_lm] 处的测试适配器。然后，运行 `uv run pytest -k test_transformer_lm` 来测试你的实现。

**交付物**：通过上述测试的 Transformer LM 模块。

#### 资源核算

能够理解 Transformer 的各个部分如何消耗计算和内存是很有用的。我们将介绍一些基本的"FLOPs 核算"步骤。Transformer 中的绝大多数 FLOPS 是矩阵乘法，因此我们核心方法很简单：

1. 写下 Transformer 前向传递中的所有矩阵乘法。
2. 将每个矩阵乘法转换为所需的 FLOPs。

对于第二步，以下事实会很有用：

> **规则**：给定 $A \in \mathbb{R}^{m \times n}$ 和 $B \in \mathbb{R}^{n \times p}$，矩阵-矩阵乘积 $AB$ 需要 $2mnp$ FLOPs。

要理解这一点，注意 $(AB)[i, j] = A[i, :] \cdot B[:, j]$，这个点积需要 n 次加法和 n 次乘法（2n FLOPs）。然后，由于矩阵-矩阵乘积 $AB$ 有 $m \times p$ 个条目，FLOPS 总数为 $(2n)(mp) = 2mnp$。

现在，在你做下一个问题之前，遍历 Transformer 块和 Transformer LM 的每个组件，列出所有矩阵乘法及其相关的 FLOPs 成本可能会有帮助。

**问题（transformer_accounting）：Transformer LM 资源核算（5分）**

(a) 考虑 GPT-2 XL，它具有以下配置：
- `vocab_size`: 50,257
- `context_length`: 1,024
- `num_layers`: 48
- `d_model`: 1,600
- `num_heads`: 25
- `d_ff`: 6,400

假设我们使用此配置构建模型。我们的模型将有多少可训练参数？假设每个参数使用单精度浮点表示，仅加载此模型需要多少内存？  
**交付物**：一到两句话回答。

(b) 识别完成 GPT-2 XL 形状模型前向传递所需的矩阵乘法。这些矩阵乘法总共需要多少 FLOPs？假设我们的输入序列具有 `context_length` 个词元。  
**交付物**：矩阵乘法列表（带描述）和所需的总 FLOPs 数。

(c) 根据你上面的分析，模型的哪些部分需要最多的 FLOPs？  
**交付物**：一到两句话回答。

(d) 用 GPT-2 small（12 层，768 d_model，12 个头）、GPT-2 medium（24 层，1024 d_model，16 个头）和 GPT-2 large（36 层，1280 d_model，20 个头）重复你的分析。随着模型大小的增加，Transformer LM 的哪些部分在总 FLOPs 中占的比例更大或更小？  
**交付物**：对于每个模型，提供模型组件及其相关 FLOPs 的细分（作为前向传递所需总 FLOPs 的比例）。此外，提供一到两句话描述改变模型大小如何改变每个组件的比例 FLOPs。

(e) 取 GPT-2 XL 并将上下文长度增加到 16,384。单次前向传递的总 FLOPs 如何变化？模型组件的 FLOPs 相对贡献如何变化？  
**交付物**：一到两句话回答。

---

## 4. 训练 Transformer LM

我们现在有了预处理数据的步骤（通过分词器）和模型（Transformer）。剩下的就是构建支持训练的所有代码。这包括：

- **损失**：我们需要定义损失函数（交叉熵）。
- **优化器**：我们需要定义最小化此损失的优化器（AdamW）。
- **训练循环**：我们需要所有支持加载数据、保存检查点和管理训练的基础设施。

### 4.1 交叉熵损失

回想一下，Transformer 语言模型为每个长度为 $m+1$ 的序列 $x$ 和 $i = 1, \ldots, m$ 定义了一个分布 $p_\theta(x_{i+1} | x_{1:i})$。给定一个由长度为 $m$ 的序列组成的训练集 $D$，我们定义标准的交叉熵（负对数似然）损失函数：

$$\ell(\theta; D) = \frac{1}{|D|m} \sum_{x \in D} \sum_{i=1}^{m} -\log p_\theta(x_{i+1} | x_{1:i}).$$

（注意，Transformer 中的单次前向传递会为所有 $i = 1, \ldots, m$ 产生 $p_\theta(x_{i+1} | x_{1:i})$。）

特别是，Transformer 为每个位置 i 计算对数几率 $o_i \in \mathbb{R}^{vocab\_size}$，这产生：⁶

$$p(x_{i+1} | x_{1:i}) = \text{softmax}(o_i)[x_{i+1}] = \frac{\exp(o_i[x_{i+1}])}{\sum_{a=1}^{vocab\_size} \exp(o_i[a])}. \quad (16, 17)$$

交叉熵损失通常是相对于对数几率向量 $o_i \in \mathbb{R}^{vocab\_size}$ 和目标 $x_{i+1}$ 定义的。⁷

实现交叉熵损失需要像 softmax 一样注意数值问题。

**问题（cross_entropy）：实现交叉熵（1分）**

**交付物**：编写一个函数来计算交叉熵损失，它接收预测的对数几率 ($o_i$) 和目标 ($x_{i+1}$) 并计算交叉熵 $\ell_i = -\log \text{softmax}(o_i)[x_{i+1}]$。你的函数应处理：
- 减去最大元素以获得数值稳定性。
- 尽可能消去 log 和 exp。
- 处理任何额外的批量维度并返回批量的平均值。与第 3.3 节一样，我们假设类似批量的维度总是在词汇表大小维度之前。

实现 [adapters.run_cross_entropy]，然后运行 `uv run pytest -k test_cross_entropy` 来测试你的实现。

#### 困惑度

交叉熵足以用于训练，但当我们评估模型时，我们还希望报告困惑度。对于长度为 m 的序列，我们遭受交叉熵损失 $\ell_1, \ldots, \ell_m$：

$$\text{perplexity} = \exp\left(\frac{1}{m} \sum_{i=1}^{m} \ell_i\right). \quad (18)$$

---

> ⁶注意 $o_i[k]$ 指的是向量 $o_i$ 中索引 k 处的值。

> ⁷这对应于 $x_{i+1}$ 上的 Dirac delta 分布与预测的 softmax($o_i$) 分布之间的交叉熵。

---

### 4.2 SGD 优化器

现在我们有了损失函数，我们将开始探索优化器。最简单的基于梯度的优化器是随机梯度下降（SGD）。我们从随机初始化的参数 $\theta_0$ 开始。然后对于每一步 $t = 0, \ldots, T-1$，我们执行以下更新：

$$\theta_{t+1} \leftarrow \theta_t - \alpha_t \nabla L(\theta_t; B_t), \quad (19)$$

其中 $B_t$ 是从数据集 $D$ 中采样的随机批量数据，学习率 $\alpha_t$ 和批量大小 $|B_t|$ 是超参数。

#### 4.2.1 在 PyTorch 中实现 SGD

为了实现我们的优化器，我们将继承 PyTorch `torch.optim.Optimizer` 类。Optimizer 子类必须实现两个方法：

```python
def __init__(self, params, ...)
```

应该初始化你的优化器。这里，`params` 将是要优化的参数集合（或参数组，以防用户想对模型的不同部分使用不同的超参数，如学习率）。确保将 `params` 传递给基类的 `__init__` 方法，它将为你在 `step` 中存储这些参数。你可以根据优化器接受附加参数（例如学习率是常见参数），并将它们作为字典传递给基类构造函数，其中键是你为这些参数选择的名称（字符串）。

```python
def step(self, closure: Optional[Callable] = None)
```

应该对参数进行一次更新。在训练循环中，这将在反向传递后调用，因此你可以访问上一批量的梯度。此方法应遍历每个参数张量 p 并就地修改它们，即设置 `p.data`，它保存与该参数相关联的张量，基于梯度 `p.grad`（如果存在），表示损失相对于该参数的梯度的张量。

PyTorch 优化器 API 有一些微妙之处，因此用示例解释会更容易。为了使我们的示例更丰富，我们将实现 SGD 的一个轻微变体，其中学习率随训练衰减，从初始学习率 $\alpha$ 开始，并随时间采取越来越小的步骤：

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{t + 1}} \nabla L(\theta_t; B_t) \quad (20)$$

让我们看看这个版本的 SGD 如何作为 PyTorch Optimizer 实现：

```python
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # 获取学习率
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # 获取与 p 相关联的状态
                t = state.get("t", 0)  # 从状态获取迭代次数，或初始值
                grad = p.grad.data  # 获取损失相对于 p 的梯度
                p.data -= lr / math.sqrt(t + 1) * grad  # 就地更新权重张量
                state["t"] = t + 1  # 递增迭代次数
        return loss
```

在 `__init__` 中，我们将参数传递给优化器，以及默认超参数，给基类构造函数（参数可能以组的形式传入，每组有不同的超参数）。如果参数只是单个 `torch.nn.Parameter` 对象集合，基构造函数将创建单个组并为其分配默认超参数。然后，在 `step` 中，我们遍历每个参数组，然后遍历该组中的每个参数，并应用方程 20。这里，我们将迭代次数作为与每个参数相关联的状态保持：我们首先读取此值，在梯度更新中使用它，然后更新它。

API 指定用户可能传入可调用的 `closure` 以在优化器步骤之前重新计算损失。我们不需要这个用于我们将使用的优化器，但我们添加它以符合 API。

要查看此工作，我们可以使用以下训练循环的最小示例：

```python
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)
for t in range(100):
    opt.zero_grad()  # 重置所有可学习参数的梯度
    loss = (weights**2).mean()  # 计算标量损失值
    print(loss.cpu().item())
    loss.backward()  # 运行反向传递，计算梯度
    opt.step()  # 运行优化器步骤
```

这是训练循环的典型结构：在每次迭代中，我们将计算损失并运行优化器的一步。在训练语言模型时，我们的可学习参数将来自模型（在 PyTorch 中，`m.parameters()` 给我们这个集合）。损失将在采样的数据批量上计算，但训练循环的基本结构将是相同的。

**问题（learning_rate_tuning）：调整学习率（1分）**

正如我们将看到的，影响训练最多的超参数之一是学习率。让我们在我们的玩具示例中实际看看。用另外三个学习率值运行上面的 SGD 示例：1e1、1e2 和 1e3，仅进行 10 次训练迭代。对于这些学习率中的每一个，损失会发生什么？它是衰减得更快、更慢，还是发散（即在训练过程中增加）？  
**交付物**：一到两句话回答你观察到的行为。

### 4.3 AdamW

现代语言模型通常使用更复杂的优化器进行训练，而不是 SGD。最近使用的大多数优化器都是 Adam 优化器 [Kingma and Ba, 2015] 的变体。我们将使用 AdamW [Loshchilov and Hutter, 2019]，这在最近的工作中被广泛使用。AdamW 提出了对 Adam 的修改，通过添加权重衰减（在每次迭代中，我们将参数拉向 0）来改进正则化，以一种与梯度更新解耦的方式。我们将按照 Loshchilov 和 Hutter [2019] 的算法 2 实现 AdamW。

AdamW 是有状态的：对于每个参数，它保持其第一和第二矩的运行估计。因此，AdamW 使用额外的内存来换取改进的稳定性和收敛性。除了学习率 $\alpha$ 之外，AdamW 还有一对超参数 $(\beta_1, \beta_2)$ 控制矩估计的更新，以及权重衰减率 $\lambda$。典型应用将 $(\beta_1, \beta_2)$ 设置为 (0.9, 0.999)，但像 LLaMA [Touvron et al., 2023] 和 GPT-3 [Brown et al., 2020] 这样的大型语言模型通常使用 (0.9, 0.95) 进行训练。算法可以写成如下，其中 $\epsilon$ 是一个小值（例如 $10^{-8}$），用于在我们得到极小的 v 值时提高数值稳定性：

---

**算法 1：AdamW 优化器**

```
init(θ)           # 初始化可学习参数
m ← 0             # 第一矩向量的初始值；与 θ 形状相同
v ← 0             # 第二矩向量的初始值；与 θ 形状相同

for t = 1, ..., T do
    采样数据批量 B_t
    g ← ∇_θ ℓ(θ; B_t)           # 计算当前时间步损失的梯度
    m ← β₁m + (1 - β₁)g         # 更新第一矩估计
    v ← β₂v + (1 - β₂)g²        # 更新第二矩估计
    α_t ← α · (1-(β₁)^t) / (1-(β₂)^t)  # 计算迭代 t 的调整后 α
    θ ← θ - α_t · m / (√v + ε)  # 更新参数
    θ ← θ - αλθ                 # 应用权重衰减
end for
```

---

注意 t 从 1 开始。你现在将实现这个优化器。

**问题（adamw）：实现 AdamW（2分）**

**交付物**：将 AdamW 优化器实现为 `torch.optim.Optimizer` 的子类。你的类应在 `__init__` 中接受学习率 $\alpha$，以及 $\beta$、$\epsilon$ 和 $\lambda$ 超参数。为了帮助你保持状态，基 Optimizer 类为你提供了一个字典 `self.state`，它将 `nn.Parameter` 对象映射到存储该参数所需任何信息的字典（对于 AdamW，这将是矩估计）。实现 [adapters.get_adamw_cls] 并确保它通过 `uv run pytest -k test_adamw`。

**问题（adamwAccounting）：AdamW 训练的资源核算（2分）**

让我们计算运行 AdamW 需要多少内存和计算。假设我们对每个张量都使用 float32。

(a) 运行 AdamW 需要多少峰值内存？根据参数、激活值、梯度和优化器状态的内存使用情况分解你的答案。用 `batch_size` 和模型超参数（`vocab_size`、`context_length`、`num_layers`、`d_model`、`num_heads`）表示你的答案。假设 $d_{ff} = 4 \times d_{model}$。

为简化起见，在计算激活值的内存使用时，仅考虑以下组件：
- Transformer 块
  - RMSNorm(s)
  - 多头自注意力子层：QKV 投影、$Q^\top K$ 矩阵乘法、softmax、值的加权和、输出投影
  - 位置级前馈：$W_1$ 矩阵乘法、SiLU、$W_2$ 矩阵乘法
- 最终 RMSNorm
- 输出嵌入
- 对数几率的交叉熵

**交付物**：参数、激活值、梯度和优化器状态的代数表达式，以及总数。

(b) 实例化你对 GPT-2 XL 形状模型的答案，得到一个仅依赖于 `batch_size` 的表达式。你能在仍适合 80GB 内存的情况下使用的最大批量大小是多少？  
**交付物**：看起来像 $a \cdot batch\_size + b$ 的表达式，其中 a、b 是数值，以及一个表示最大批量大小的数字。

(c) 运行 AdamW 的一步需要多少 FLOPs？  
**交付物**：代数表达式，并附简要说明。

(d) 模型 FLOPs 利用率（MFU）定义为相对于硬件理论峰值 FLOP 吞吐量的观察吞吐量（每秒词元）[Chowdhery et al., 2022]。NVIDIA A100 GPU 对于 float32 操作的理论峰值为 19.5 teraFLOP/s。假设你能够达到 50% MFU，在单个 A100 上训练 GPT-2 XL 进行 400K 步、批量大小为 1024 需要多长时间？遵循 Kaplan 等人 [2020] 和 Hoffmann 等人 [2022]，假设反向传递的 FLOPs 是前向传递的两倍。  
**交付物**：训练所需的天数，并附简要说明。

### 4.4 学习率调度

导致损失最快下降的学习率值通常在训练过程中会变化。在训练 Transformer 时，通常使用学习率调度，我们从较大的学习率开始，在开始时进行更快的更新，并随着模型训练慢慢衰减到较小的值。⁸

在本作业中，我们将实现用于训练 LLaMA [Touvron et al., 2023] 的余弦退火调度。

调度器只是一个函数，它接受当前步 t 和其他相关参数（如初始和最终学习率），并返回步 t 的梯度更新要使用的学习率。最简单的调度是常数函数，它将对任何 t 返回相同的学习率。

余弦退火学习率调度采用 (i) 当前迭代 t，(ii) 最大学习率 $\alpha_{max}$，(iii) 最小（最终）学习率 $\alpha_{min}$，(iv) 预热迭代数 $T_w$，和 (v) 余弦退火迭代数 $T_c$。迭代 t 的学习率定义为：

- **（预热）** 如果 $t < T_w$，则 $\alpha_t = \frac{t}{T_w} \alpha_{max}$。
- **（余弦退火）** 如果 $T_w \leq t \leq T_c$，则 $\alpha_t = \alpha_{min} + \frac{1}{2}\left(1 + \cos\left(\frac{t-T_w}{T_c-T_w}\pi\right)\right)(\alpha_{max} - \alpha_{min})$。
- **（退火后）** 如果 $t > T_c$，则 $\alpha_t = \alpha_{min}$。

---

> ⁸有时使用学习率回升（重启）的调度来帮助越过局部最小值是常见的。

---

**问题（learning_rate_schedule）：实现带预热的余弦学习率调度（1分）**

编写一个函数，接受 t、$\alpha_{max}$、$\alpha_{min}$、$T_w$ 和 $T_c$，并根据上面定义的调度器返回学习率 $\alpha_t$。然后实现 [adapters.get_lr_cosine_schedule] 并确保它通过 `uv run pytest -k test_get_lr_cosine_schedule`。

### 4.5 梯度裁剪

在训练期间，我们有时会遇到产生大梯度的训练样本，这可能会破坏训练稳定性。为了缓解这一点，实践中经常采用的一种技术是梯度裁剪。其思想是在每次反向传递后、采取优化器步骤之前，对梯度的范数强制执行一个限制。

给定（所有参数的）梯度 $g$，我们计算其 $\ell_2$-范数 $\|g\|_2$。如果这个范数小于最大值 M，我们将 g 保持不变；否则，我们将 g 按因子 $\frac{M}{\|g\|_2+\epsilon}$ 缩小（其中添加了一个小 $\epsilon$，例如 $10^{-6}$，用于数值稳定性）。注意，结果范数将略低于 M。

**问题（gradient_clipping）：实现梯度裁剪（1分）**

编写一个实现梯度裁剪的函数。你的函数应该接受参数列表和最大 $\ell_2$-范数。它应该就地修改每个参数梯度。使用 $\epsilon = 10^{-6}$（PyTorch 默认值）。然后，在 [adapters.run_gradient_clipping] 处实现适配器并确保它通过 `uv run pytest -k test_gradient_clipping`。

---

## 5. 训练循环

我们现在终于将我们构建的主要组件组合在一起：分词数据、模型和优化器。

### 5.1 数据加载器

分词数据（例如你在 `tokenizer_experiments` 中准备的）是单个词元序列 $x = (x_1, \ldots, x_n)$。即使源数据可能由单独的文档组成（例如不同的网页或源代码文件），常见的做法是将所有这些连接成单个词元序列，在它们之间添加分隔符（如 `<|endoftext|>` 词元）。

数据加载器将其转换为批量流，其中每个批量由 B 个长度为 m 的序列组成，配对的相应下一个词元，长度也为 m。例如，对于 B = 1，m = 3，`([x_2, x_3, x_4], [x_3, x_4, x_5])` 将是一个潜在的批量。

以这种方式加载数据简化了训练，原因有几个。首先，任何 $1 \leq i < n - m$ 都给出有效的训练序列，因此采样序列是微不足道的。由于所有训练序列具有相同的长度，因此不需要填充输入序列，这提高了硬件利用率（还通过增加批量大小 B）。最后，我们也不需要完全加载完整数据集来采样训练数据，因此很容易处理可能不适合内存的大型数据集。

**问题（data_loading）：实现数据加载（2分）**

**交付物**：编写一个函数，它接收 numpy 数组 x（包含词元 ID 的整数数组）、`batch_size`、`context_length` 和 PyTorch 设备字符串（例如 `'cpu'` 或 `'cuda:0'`），并返回一对张量：采样的输入序列和相应的下一个词元目标。两个张量都应具有形状 `(batch_size, context_length)`，包含词元 ID，并且都应放在请求的设备上。要针对我们提供的测试测试你的实现，你首先需要实现 [adapters.run_get_batch] 处的测试适配器。然后，运行 `uv run pytest -k test_get_batch` 来测试你的实现。

---

> **💡 低资源/降规模提示：在 CPU 或 Apple Silicon 上加载数据**
>
> 如果你计划在 CPU 或 Apple Silicon 上训练你的 LM，你需要将数据移动到正确的设备（类似地，你稍后应该对模型使用相同的设备）。如果你在 CPU 上，可以使用 `'cpu'` 设备字符串，在 Apple Silicon（M* 芯片）上，可以使用 `'mps'` 设备字符串。
>
> 有关 MPS 的更多信息，请查看这些资源：
> - https://developer.apple.com/metal/pytorch/
> - https://pytorch.org/docs/main/notes/mps.html

---

**数据集太大无法加载到内存怎么办？** 我们可以使用名为 mmap 的 Unix 系统调用，将磁盘上的文件映射到虚拟内存，并在访问该内存位置时惰性加载文件内容。因此，你可以"假装"整个数据集都在内存中。Numpy 通过 `np.memmap`（或 `np.load` 的标志 `mmap_mode='r'`，如果你最初用 `np.save` 保存数组）实现这一点，它将返回一个 numpy 数组状对象，在你访问它们时按需加载条目。在训练期间从数据集（即 numpy 数组）采样时，确保通过 `np.memmap` 或标志 `mmap_mode='r'` 到 `np.load`（取决于你如何保存数组）以内存映射模式加载数据集。确保你还指定与你要加载的数组匹配的 dtype。显式验证内存映射数据是否正确（例如，不包含超出预期词汇表大小的值）可能会有帮助。

### 5.2 检查点

除了加载数据外，我们还需要在训练时保存模型。在运行作业时，我们通常希望能够恢复因某种原因中途停止的训练运行（例如，由于作业超时、机器故障等）。即使一切顺利，我们可能也希望稍后能够访问中间模型（例如，事后研究训练动态、从训练不同阶段采样模型等）。

检查点应包含恢复训练所需的所有状态。我们当然希望能够至少恢复模型权重。如果使用有状态优化器（如 AdamW），我们还需要保存优化器的状态（例如，对于 AdamW，矩估计）。最后，为了恢复学习率调度，我们需要知道我们在哪个迭代次数停止的。PyTorch 使保存所有这些变得容易：每个 `nn.Module` 都有一个 `state_dict()` 方法，返回包含所有可学习权重的字典；我们可以稍后使用姐妹方法 `load_state_dict()` 恢复这些权重。任何 `nn.optim.Optimizer` 也是如此。最后，`torch.save(obj, dest)` 可以将对象（例如包含某些值中的张量的字典，还有常规 Python 对象如整数）转储到文件（路径）或类文件对象，然后可以用 `torch.load(src)` 将其加载回内存。

**问题（checkpointing）：实现模型检查点（1分）**

实现以下两个函数来加载和保存检查点：

```python
def save_checkpoint(model, optimizer, iteration, out)
```

应该将前三个参数的所有状态转储到类文件对象 `out` 中。你可以使用模型和优化器的 `state_dict` 方法来获取它们的相关状态，并使用 `torch.save(obj, out)` 将 `obj` 转储到 `out`（PyTorch 支持路径或类文件对象）。一个典型的选择是让 `obj` 是一个字典，但你可以使用任何格式，只要你能稍后加载你的检查点。

此函数期望以下参数：
- `model: torch.nn.Module`
- `optimizer: torch.optim.Optimizer`
- `iteration: int`
- `out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`

```python
def load_checkpoint(src, model, optimizer)
```

应该从 `src`（路径或类文件对象）加载检查点，然后从该检查点恢复模型和优化器状态。你的函数应返回保存到检查点的迭代次数。你可以使用 `torch.load(src)` 恢复你在 `save_checkpoint` 实现中保存的内容，以及模型和优化器中的 `load_state_dict` 方法将它们恢复到之前的状态。

此函数期望以下参数：
- `src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`
- `model: torch.nn.Module`
- `optimizer: torch.optim.Optimizer`

实现 [adapters.run_save_checkpoint] 和 [adapters.run_load_checkpoint] 适配器，并确保它们通过 `uv run pytest -k test_checkpointing`。

### 5.3 训练循环

现在，终于该把你实现的所有组件组合到你的主训练脚本中了。使使用不同超参数（例如通过将它们作为命令行参数）启动训练运行变得容易会得到回报，因为你稍后会多次这样做来研究不同选择如何影响训练。

**问题（training_together）：组合在一起（4分）**

**交付物**：编写一个运行训练循环的脚本，在用户提供的输入上训练你的模型。特别是，我们建议你的训练脚本至少允许以下功能：
- 能够配置和控制各种模型和优化器超参数。
- 使用 `np.memmap` 内存高效地加载训练和验证大型数据集。
- 将检查点序列化到用户提供的路径。
- 定期记录训练和验证性能（例如，到控制台和/或 Weights and Biases 等外部服务）。

---

## 6. 生成文本

现在我们可以训练模型了，我们需要的最后一部分是从模型生成文本的能力。

回想一下，语言模型接收长度为 `sequence_length` 的（可能是批量的）整数序列，并产生大小为 `(sequence_length × vocab size)` 的矩阵，其中序列的每个元素都是预测该位置之后词的分布。我们现在将编写几个函数，将其转换为新序列的采样方案。

### Softmax

按照标准约定，语言模型输出是最终线性层（"对数几率"）的输出，因此我们必须通过 softmax 操作将其转换为归一化概率，我们之前在方程 10 中看到过。

### 解码

要从我们的模型生成文本（解码），我们将为模型提供前缀词元序列（"提示"），并要求它产生预测序列中下一个词的词汇表分布。然后，我们将从此词汇表项分布中采样以确定下一个输出词元。

具体而言，解码过程的一步应该接收序列 $x_{1...t}$ 并通过以下方程返回词元 $x_{t+1}$：

$$P(x_{t+1} = i | x_{1...t}) = \frac{\exp(v_i)}{\sum_j \exp(v_j)}$$
$$v = \text{TransformerLM}(x_{1...t})_t \in \mathbb{R}^{vocab\_size}$$

其中 TransformerLM 是我们的模型，它接收长度为 `sequence_length` 的序列输入，并产生大小为 `(sequence_length × vocab_size)` 的矩阵，我们取该矩阵的最后一个元素，因为我们在寻找位置 t 的下一个词预测。

这通过重复从这些单步条件采样（将我们之前生成的输出词元附加到下一个解码时间步的输入）直到我们生成序列结束词元 `<|endoftext|>`（或用户指定的要生成的最大词元数）来给我们一个基本解码器。

### 解码技巧

我们将使用小模型进行实验，小模型有时会生成非常低质量的文本。两个简单的解码技巧可以帮助解决这些问题。首先，在温度缩放中，我们用温度参数 $\tau$ 修改我们的 softmax，其中新的 softmax 是：

$$\text{softmax}(v, \tau)_i = \frac{\exp(v_i/\tau)}{\sum_{j=1}^{|vocab\_size|} \exp(v_j/\tau)}. \quad (24)$$

注意，设置 $\tau \rightarrow 0$ 使得 v 的最大元素占主导地位，softmax 的输出成为集中在这个最大元素的单热向量。

其次，另一个技巧是核采样或 top-p 采样，我们通过截断低概率词来修改采样分布。设 q 是我们从（温度缩放的）softmax 得到的大小为 `(vocab_size)` 的概率分布。具有超参数 p 的核采样根据以下方程产生下一个词元：

$$P(x_{t+1} = i|q) = \begin{cases} \frac{q_i}{\sum_{j \in V(p)} q_j} & \text{如果 } i \in V(p) \\ 0 & \text{否则} \end{cases}$$

其中 $V(p)$ 是满足 $\sum_{j \in V(p)} q_j \geq p$ 的最小索引集。你可以通过首先按大小对概率分布 q 进行排序，并选择最大的词汇表元素直到达到目标水平 $\alpha$ 来轻松计算这个量。

**问题（decoding）：解码（3分）**

**交付物**：实现一个从你的语言模型解码的函数。我们建议支持以下功能：
- 为用户提供的提示生成补全（即接收一些 $x_{1...t}$ 并采样补全直到你遇到 `<|endoftext|>` 词元）。
- 允许用户控制生成的最大词元数。
- 给定所需的温度值，在采样前将 softmax 温度缩放应用于预测的下一个词分布。
- Top-p 采样（Holtzman et al., 2020；也称为核采样），给定用户指定的阈值。

---

## 7. 实验

现在是时候把所有东西放在一起，在预训练数据集上训练（小型）语言模型了。

### 7.1 如何运行实验和交付物

理解 Transformer 架构组件背后原理的最佳方法是实际修改它并自己运行。实践经验无可替代。

为此，能够快速、一致地实验并记录你所做的事情非常重要。为了快速实验，我们将在小型模型（17M 参数）和简单数据集（TinyStories）上运行许多实验。为了保持一致性，你将以系统的方式消融组件和改变超参数，为了记录，我们将要求你提交实验日志和与每个实验相关的学习曲线。

为了使提交损失曲线成为可能，确保定期评估验证损失并记录步数和挂钟时间。你可能会发现 Weights and Biases 等日志基础设施很有帮助。

**问题（experiment_log）：实验日志（3分）**

为你的训练和评估代码创建实验跟踪基础设施，使你能够跟踪实验和关于梯度步数和挂钟时间的损失曲线。  
**交付物**：实验的日志基础设施代码和本部分下面作业问题的实验日志（你尝试的所有事情的文档）。

### 7.2 TinyStories

我们将从一个非常简单的数据集（TinyStories；Eldan and Li, 2023）开始，模型将快速训练，我们可以看到一些有趣的行为。获取此数据集的说明在第 1 节。

以下是此数据集样子的示例。

---

**示例（tinystories_example）：TinyStories 中的一个示例**

从前有一个叫 Ben 的小男孩。Ben 喜欢探索他周围的世界。他看到了许多神奇的东西，比如商店里陈列的美丽花瓶。一天，Ben 走过商店时，发现了一个非常特别的花瓶。当 Ben 看到它时，他非常惊讶！他说："哇，那真是一个神奇的花瓶！我能买吗？"店主微笑着说："当然可以。你可以把它带回家，向所有朋友展示它有多神奇！"于是 Ben 把花瓶带回家，他为此感到非常自豪！他叫来朋友，向他们展示这个神奇的花瓶。他所有的朋友都认为花瓶很漂亮，不敢相信 Ben 有多幸运。这就是 Ben 如何在商店里找到一个神奇的花瓶的故事！

---

#### 超参数调整

我们将告诉你一些非常基本的超参数，并要求你找到其他一些效果良好的设置。

| 参数 | 值 | 说明 |
|------|-----|------|
| `vocab_size` | 10000 | 典型词汇表大小在数万到数十万之间。你应该改变这个并看看词汇表和模型行为如何变化。 |
| `context_length` | 256 | 像 TinyStories 这样的简单数据集可能不需要长序列长度，但对于后面的 OpenWebText 数据，你可能想改变这个。尝试改变这个并看看它对每次迭代运行时间和最终困惑度的影响。 |
| `d_model` | 512 | 这比许多小型 Transformer 论文中使用的 768 维度略小，但这会使事情更快。 |
| `d_ff` | 1344 | 这大约是 $\frac{8}{3} d_{model}$，同时是 64 的倍数，有利于 GPU 性能。 |
| RoPE theta 参数 $\Theta$ | 10000 | |
| 层数和头数 | 4 层，16 个头 | 一起，这将产生约 17M 个非嵌入参数，这是一个相当小的 Transformer。 |
| 处理的总词元数 | 327,680,000 | 你的批量大小 × 总步数 × 上下文长度应大致等于此值。 |

你应该做一些试错，为以下其他超参数找到良好的默认值：学习率、学习率预热、其他 AdamW 超参数 ($\beta_1, \beta_2, \epsilon$) 和权重衰减。你可以在 Kingma 和 Ba [2015] 中找到这些超参数的一些典型选择。

#### 组合在一起

现在你可以通过获取训练好的 BPE 分词器、对训练数据集进行分词并在你编写的训练循环中运行来把所有东西组合在一起。重要提示：如果你的实现正确且高效，上述超参数应该导致在 1 个 H100 GPU 上大约 30-40 分钟的运行时间。如果你的运行时间要长得多，请检查并确保你的数据加载、检查点或验证损失代码不会成为运行时间的瓶颈，并且你的实现已正确批处理。

#### 调试模型架构的技巧和窍门

我们强烈建议熟悉你 IDE 的内置调试器（例如 VSCode/PyCharm），与使用 print 语句调试相比，这将节省你的时间。如果你使用文本编辑器，可以使用类似 pdb 的工具。调试模型架构时的一些其他良好实践包括：

- 开发任何神经网络架构时的常见第一步是对单个 minibatch 过拟合。如果你的实现正确，你应该能够快速将训练损失驱动到接近零。
- 在各种模型组件中设置调试断点，并检查中间张量的形状以确保它们符合你的预期。
- 监控激活值、模型权重和梯度的范数，确保它们不会爆炸或消失。

**问题（learning_rate）：调整学习率（3分）（4 H100 小时）**

学习率是要调整的最重要的超参数之一。以你训练的基础模型为例，回答以下问题：

(a) 对学习率执行超参数扫描并报告最终损失（或注意优化器发散时发散）。  
**交付物**：与多个学习率相关的学习曲线。解释你的超参数搜索策略。

**交付物**：在 TinyStories 上验证损失（每词元）最多为 1.45 的模型。

---

> **💡 低资源/降规模提示：在 CPU 或 Apple Silicon 上训练少量步数**
>
> 如果你在 cpu 或 mps 上运行，你应该将处理的总词元数减少到 40,000,000，这将足以产生相当流畅的文本。你也可以将目标验证损失从 1.45 增加到 2.00。
>
> 使用调整好的学习率在 M3 Max 芯片和 36 GB RAM 上运行我们的解决方案代码，我们使用批量大小 × 总步数 × 上下文长度 = $32 \times 5000 \times 256 = 40,960,000$ 个词元，在 cpu 上需要 1 小时 22 分钟，在 mps 上需要 36 分钟。在第 5000 步，我们达到验证损失 1.80。
>
> 一些额外的提示：
> - 当使用 X 个训练步数时，我们建议调整余弦学习率衰减调度，使其在精确的 X 步终止衰减（即达到最小学习率）。
> - 当使用 mps 时，不要使用 TF32 内核，即不要像使用 cuda 设备那样设置 `torch.set_float32_matmul_precision('high')`。我们尝试在 mps（torch 版本 2.6.0）上启用 TF32 内核，发现后端会使用静默损坏的内核，导致训练不稳定。
> - 你可以通过 JIT 编译模型来加速训练。具体来说：
>   - 在 cpu 上，用 `model = torch.compile(model)` 编译模型
>   - 在 mps 上，你可以使用 `model = torch.compile(model, backend="aot_eager")` 某种程度上优化反向传递。截至 torch 版本 2.6.0，Inductor 编译在 mps 上不受支持。

---

(b) 民间智慧认为最佳学习率是"在稳定性边缘"。调查学习率发散的点与你最佳学习率的关系。  
**交付物**：包含至少一个发散运行的增加学习率的学习曲线，以及关于这与收敛率如何相关的分析。

现在让我们改变批量大小，看看训练会发生什么。批量大小很重要——它们让我们通过执行更大的矩阵乘法从 GPU 获得更高的效率，但我们总是希望批量大小大是真的吗？让我们运行一些实验来找出答案。

**问题（batch_size_experiment）：批量大小变化（1分）（2 H100 小时）**

将你的批量大小从 1 一直改变到 GPU 内存限制。尝试中间至少几个批量大小，包括典型大小如 64 和 128。  
**交付物**：不同批量大小运行的学习曲线。如果需要，应再次优化学习率。

**交付物**：几句话讨论你对批量大小及其对训练影响的发现。

有了你的解码器，我们现在可以生成文本了！我们将从模型生成并看看它有多好。作为参考，你应该得到至少与下面示例一样好的输出。

---

**示例（ts_generate_example）：TinyStories 语言模型的样本输出**

从前有一个漂亮的女孩叫 Lily。她喜欢吃口香糖，尤其是大黑的那个。一天，Lily 的妈妈请她帮忙做晚饭。Lily 非常兴奋！她喜欢帮助妈妈。Lily 的妈妈做了一大锅汤当晚餐。Lily 非常高兴地说："谢谢你，妈妈！我爱你。"她帮妈妈把汤倒进一个大碗里。晚饭后，Lily 的妈妈做了一些美味的汤。Lily 很喜欢！她说："谢谢你，妈妈！这汤太好喝了！"她的妈妈微笑着说："我很高兴你喜欢，Lily。"他们做完饭后继续一起做饭。故事结束。

---

> **💡 低资源/降规模提示：在 CPU 或 Apple Silicon 上生成文本**
>
> 相反，如果你使用处理 40M 词元的低资源配置，你应该看到生成的文本仍然类似英语，但不如上面流畅。例如，我们在处理 40M 词元训练的 TinyStories 语言模型的样本输出如下：
>
> 从前有一个叫 Sue 的小女孩。Sue 有一颗她非常喜欢的牙齿。这是他最好的头。一天，Sue 去散步，遇到了一只瓢虫！他们成为了好朋友，一起在小路上玩耍。
>
> "嘿，Polly！我们出去吧！"Tim 说。Sue 看着天空，发现很难找到一种跳舞发光的方式。她微笑着同意帮助说话！"
>
> 当 Sue 看着天空移动时，它是什么。她

---

以下是精确的问题陈述和我们要求的内容：

**问题（generate）：生成文本（1分）**

使用你的解码器和训练好的检查点，报告你的模型生成的文本。你可能需要操作解码器参数（温度、top-p 等）以获得流畅的输出。  
**交付物**：至少 256 个词元的文本转储（或直到第一个 `<|endoftext|>` 词元），以及关于此输出流畅性的简要评论和至少两个影响此输出好坏的因素。

### 7.3 消融和架构修改

理解 Transformer 的最佳方法是实际修改它并观察其行为。我们现在将做一些简单的消融和修改。

#### 消融 1：层归一化

人们常说层归一化对 Transformer 训练的稳定性很重要。但也许我们想冒险一下。让我们从每个 Transformer 块中移除 RMSNorm，看看会发生什么。

**问题（layer_norm_ablation）：移除 RMSNorm 并训练（1分）（1 H100 小时）**

从你的 Transformer 中移除所有 RMSNorm 并训练。在之前的最佳学习率下会发生什么？你能通过使用较低的学习率获得稳定性吗？  
**交付物**：移除 RMSNorm 并训练时的学习曲线，以及最佳学习率的学习曲线。

**交付物**：关于 RMSNorm 影响的几句话评论。

让我们现在调查另一个层归一化选择，乍一看似乎很随意。Pre-norm Transformer 块定义为：

$$z = x + \text{MultiHeadedSelfAttention}(\text{RMSNorm}(x))$$
$$y = z + \text{FFN}(\text{RMSNorm}(z)).$$

这是对原始 Transformer 架构的少数"共识"修改之一，原始架构使用 post-norm 方法：

$$z = \text{RMSNorm}(x + \text{MultiHeadedSelfAttention}(x))$$
$$y = \text{RMSNorm}(z + \text{FFN}(z)).$$

让我们恢复到 post-norm 方法，看看会发生什么。

**问题（pre_norm_ablation）：实现 post-norm 并训练（1分）（1 H100 小时）**

将你的 pre-norm Transformer 实现修改为 post-norm 实现。用 post-norm 模型训练，看看会发生什么。  
**交付物**：post-norm Transformer 的学习曲线，与 pre-norm 的比较。

我们看到层归一化对 Transformer 的行为有重大影响，甚至层归一化的位置也很重要。

#### 消融 2：位置嵌入

接下来，我们将调查位置嵌入对模型性能的影响。具体来说，我们将比较我们的基础模型（使用 RoPE）与完全不包含位置嵌入（NoPE）。事实证明，仅解码器 Transformer，即我们实现的具有因果掩码的 Transformer，理论上可以在不明确提供位置嵌入的情况下推断相对或绝对位置信息 [Tsai et al., 2019, Kazemnejad et al., 2023]。我们现在将实证测试 NoPE 与 RoPE 相比的表现如何。

**问题（no_pos_emb）：实现 NoPE（1分）（1 H100 小时）**

修改你的带有 RoPE 的 Transformer 实现，完全移除位置嵌入信息，看看会发生什么。  
**交付物**：比较 RoPE 和 NoPE 性能的学习曲线。

#### 消融 3：SwiGLU vs. SiLU

接下来，我们将遵循 Shazeer [2020] 并测试前馈网络中门控的重要性，通过比较 SwiGLU 前馈网络与使用 SiLU 激活但没有门控线性单元（GLU）的前馈网络的性能：

$$\text{FFN}_{\text{SiLU}}(x) = W_2 \text{SiLU}(W_1 x). \quad (25)$$

回想一下，在我们的 SwiGLU 实现中，我们将内部前馈层的维度设置为大约 $d_{ff} = \frac{8}{3} d_{model}$（同时确保 $d_{ff} \mod 64 = 0$ 以利用 GPU 张量核心）。在你的 FFNSiLU 实现中，你应该设置 $d_{ff} = 4 \times d_{model}$，以大致匹配 SwiGLU 前馈网络的参数数量（它有三个而不是两个权重矩阵）。

**问题（swiglu_ablation）：SwiGLU vs. SiLU（1分）（1 H100 小时）**

**交付物**：比较 SwiGLU 和 SiLU 前馈网络性能的学习曲线，参数数量大致匹配。

**交付物**：几句话讨论你的发现。

---

> **💡 低资源/降规模提示：GPU 资源有限的在线学生应在 TinyStories 上测试修改**
>
> 在作业的剩余部分，我们将转向更大规模、更嘈杂的网络数据集（OpenWebText），进行架构修改实验和（可选）向课程排行榜提交。
>
> 在 OpenWebText 上训练 LM 到流畅需要很长时间，因此我们建议 GPU 访问有限的在线学生继续在 TinyStories 上测试修改（使用验证损失作为评估性能的指标）。

---

### 7.4 在 OpenWebText 上运行

我们现在将转向从网络爬取创建的标准预训练数据集。OpenWebText [Gokaslan et al., 2019] 的小样本也作为单个文本文件提供：参见第 1 节了解如何访问此文件。

以下是 OpenWebText 的示例。注意文本是多么真实、复杂和多样。你可能想浏览训练数据集，了解网络抓取语料库的训练数据是什么样的。

---

**示例（owt_example）：OWT 中的一个示例**

Baseball Prospectus 技术总监 Harry Pavlidis 在雇用 Jonathan Judge 时冒了风险。Pavlidis 知道，正如 Alan Schwarz 在《数字游戏》中所写，"美国文化中没有一个角落比棒球运动员的表现更精确地计算、更热情地量化。"只需点击几下，你就可以发现 Noah Syndergaard 的快速球在到达本垒板的路上每分钟旋转超过 2,100 次，Nelson Cruz 在 2016 年拥有比赛中最高的平均退出速度，以及其他无数似乎从视频游戏或科幻小说中撕下来的花絮。不断增长的数据海洋赋予了棒球文化中越来越重要的角色：分析爱好者。

这种赋权伴随着额外的审查——对测量，也对背后的人和出版物。对于 Baseball Prospectus，Pavlidis 非常了解伴随定量不完美的强烈反对。他还知道网站的捕手指标需要重新设计，而且需要一个有学识的头脑——能够解决复杂统计建模问题的人——来完成这项工作。

"他让我们感到害怕。" Harry Pavlidis

Pavlidis 有一种直觉，Judge "理解它"，基于后者的写作和他们在网站赞助的球场活动中的互动。不久之后，两人边喝边聊。Pavlidis 的直觉得到了验证。Judge 适合这个职位——更好的是，他愿意接受。"我和很多人谈过，"Pavlidis 说，"他是唯一一个足够勇敢接受它的人。"[...]

---

注意：你可能需要为这个实验重新调整超参数，如学习率或批量大小。

**问题（main_experiment）：在 OWT 上的实验（2分）（3 H100 小时）**

用与 TinyStories 相同的模型架构和总训练迭代次数在 OpenWebText 上训练你的语言模型。这个模型表现如何？  
**交付物**：你的语言模型在 OpenWebText 上的学习曲线。描述与 TinyStories 的损失差异——我们应该如何解释这些损失？

**交付物**：OpenWebText LM 生成的文本，格式与 TinyStories 输出相同。这段文本的流畅度如何？为什么即使我们有与 TinyStories 相同的模型和计算预算，输出质量却更差？

### 7.5 你自己的修改 + 排行榜

恭喜你走到这一步。你快完成了！你现在将尝试改进 Transformer 架构，并看看你的超参数和架构如何与班上其他学生竞争。

#### 排行榜规则

除了以下规则外，没有其他限制：

**运行时间**：你的提交最多可以在 H100 上运行 1.5 小时。你可以通过在 slurm 提交脚本中设置 `--time=01:30:00` 来强制执行此操作。

**数据**：你只能使用我们提供的 OpenWebText 训练数据集。

否则，你可以做任何你想做的事情。

如果你正在寻找一些实现思路，你可以查看这些资源：
- 最先进的开源 LLM 家族，如 Llama 3 [Grattafiori et al., 2024] 或 Qwen 2.5 [Yang et al., 2024]。
- NanoGPT speedrun 仓库（https://github.com/KellerJordan/modded-nanogpt），社区成员在其中发布许多有趣的修改用于"速通"小规模语言模型预训练。例如，一个可以追溯到原始 Transformer 论文的常见修改是将输入和输出嵌入的权重绑定在一起（参见 Vaswani et al. [2017]（第 3.4 节）和 Chowdhery et al. [2022]（第 2 节））。如果你尝试权重绑定，你可能需要减小嵌入/LM 头初始化的标准差。

你会想在 OpenWebText 的小子集或 TinyStories 上测试这些，然后再尝试完整的 1.5 小时运行。

作为警告，我们确实注意到你在这个排行榜中发现的一些修改可能在更大规模的预训练中无法推广。我们将在课程的缩放定律单元中进一步探讨这个想法。

**问题（leaderboard）：排行榜（6分）（10 H100 小时）**

你将在上述排行榜规则下训练一个模型，目标是在 1.5 H100 小时内最小化语言模型的验证损失。  
**交付物**：记录的最终验证损失、一条清楚显示小于 1.5 小时的挂钟时间 x 轴的相关学习曲线，以及你做了什么描述。我们期望排行榜提交至少击败 5.0 损失的朴素基线。在这里提交到排行榜：https://github.com/stanford-cs336/assignment1-basics-leaderboard。

---

## 参考文献

[1] Eldan, R., & Li, Y. (2023). TinyStories: How small can language models be and still speak coherent English? arXiv:2305.07759.

[2] Gokaslan, A., Cohen, V., Pavlick, E., & Tellex, S. (2019). OpenWebText corpus. http://Skylion007.github.io/OpenWebTextCorpus

[3] Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. In Proc. of ACL.

[4] Wang, C., Cho, K., & Gu, J. (2019). Neural machine translation with byte-level subwords. arXiv:1909.03341.

[5] Gage, P. (1994). A new algorithm for data compression. C Users Journal, 12(2):23–38.

[6] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners.

[7] Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative pre-training.

[8] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In Proc. of NeurIPS.

[9] Nguyen, T. Q., & Salazar, J. (2019). Transformers without tears: Improving the normalization of self-attention. In Proc. of IWSWLT.

[10] Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., & Liu, T. Y. (2020). On layer normalization in the Transformer architecture. In Proc. of ICML.

[11] Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. arXiv:1607.06450.

[12] Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M. A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., & Lample, G. (2023). Llama: Open and efficient foundation language models. arXiv:2302.13971.

[13] Zhang, B., & Sennrich, R. (2019). Root mean square layer normalization. In Proc. of NeurIPS.

[14] Grattafiori, A., et al. (2024). The llama 3 herd of models. https://arxiv.org/abs/2407.21783.

[15] Yang, A., et al. (2024). Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

[16] Hendrycks, D., & Gimpel, K. (2016). Bridging nonlinearities and stochastic regularizers with gaussian error linear units. arXiv:1606.08415.

[17] Elfwing, S., Uchibe, E., & Doya, K. (2017). Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. https://arxiv.org/abs/1702.03118.

[18] Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2017). Language modeling with gated convolutional networks. https://arxiv.org/abs/1612.08083.

[19] Shazeer, N. (2020). GLU variants improve transformer. arXiv:2002.05202.

[20] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). Roformer: Enhanced transformer with rotary position embedding.

[21] Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In Proc. of ICLR.

[22] Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. In Proc. of ICLR.

[23] Brown, T. B., et al. (2020). Language models are few-shot learners. In Proc. of NeurIPS.

[24] Kaplan, J., et al. (2020). Scaling laws for neural language models. arXiv:2001.08361.

[25] Hoffmann, J., et al. (2022). Training compute-optimal large language models. arXiv:2203.15556.

[26] Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). The curious case of neural text degeneration. In Proc. of ICLR.

[27] Tsai, Y. H. H., Bai, S., Yamada, M., Morency, L. P., & Salakhutdinov, R. (2019). Transformer dissection: An unified understanding for transformer's attention via the lens of kernel. In Proc. of EMNLP-IJCNLP.

[28] Kazemnejad, A., Padhi, I., Natesan, K., Das, P., & Reddy, S. (2023). The impact of positional encoding on length generalization in transformers. In Thirty-seventh Conference on Neural Information Processing Systems.
