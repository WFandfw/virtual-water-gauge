SAM3 模型权重目录

权重文件 sam3.pt 体积约 3.4 GB，不随工程代码分发。
请自行放置模型权重到本目录（或服务器 /root/sam/ 下）：

  sam3.pt          —— SAM3 主权重（checkpoint）
  bpe_simple_vocab_16e6.txt.gz —— SAM3 文本提示所需的 BPE 词表（位于 model_lib/sam3/assets/ 下）

云端代码默认权重路径：
  control_first.py   CHECKPOINT_PATH = "/root/sam/sam3.pt"
