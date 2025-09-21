## 人工智能学习资料（个人整理）(Artificial intelligence learning materials compilation)

> 脑子一热写出来的，后面会不断补充更新。大部分内容我都学过，感觉很优秀、容易理解（数学部分暂缺，需要补）。

---

目标
- 为自己和愿意参考的人提供一个从入门到进阶的“可做”清单：每个项目包含数据来源、最小可运行 demo、关键坑与参考链接。

流程
1. 读 `roadmap.md` 了解阶段与预期产出。
2. 选一个 `projects/` 内的项目，阅读 `README`，运行 notebook。
3. 把运行结果推到 `results/`（可选）并写短总结。

示例项目：projects/02-mnist-cnn/ 包含：
- `notebook.ipynb`（训练与可视化）
- `requirements.txt`
- `README.md`（如何跑）

---

### 学习路线总览

**学习顺序：** Python 基础 → 人工智能导论 → 数据分析与处理 → 机器学习 → 深度学习（PyTorch） → 强化学习 → 拓展实践与研究

搭配“理论 + 实战”学习，每个阶段先看基础理论，再做代码实践。

---

### 学习路径

**Python 基础**

* 语法、数据结构、函数、类
* 工具：Vscode、Anaconda、Jupyter
* 库：Numpy、Pandas、Matplotlib

**人工智能导论**

* 历史：图灵测试、达特茅斯会议、符号主义 vs 连接主义
* AI 分类：分类、预测、策略问题
* 基础算法：逻辑回归、线性回归、梯度下降

**数据分析 & 数据处理**

* 数据标准化、归一化
* 矩阵运算、特征提取、特征选择
* 超参数设置与调优

**机器学习基础（sklearn 实战）**

* 数据集：训练集、验证集、测试集
* 拟合、过拟合、欠拟合
* 评估指标：准确率、召回率、F1 分数
* 损失函数：均方误差、交叉熵

**深度学习（PyTorch）**

* 神经网络：输入层、隐藏层、输出层、权重、偏置
* 激活函数：Sigmoid、ReLU、Softmax
* 前向传播、反向传播、梯度下降
* 优化器：SGD、Adam

**深度学习进阶**

* CNN 卷积神经网络
* RNN / LSTM / GRU 序列建模
* Transformer：注意力机制、编码器、解码器、BERT/GPT
* 实战：MNIST 手写数字识别、房价预测、SimpleCNN、SimpleRNN、MiniChat

**强化学习入门**

* 概念：环境、状态、动作、奖励
* 算法：Q-learning、DQN、策略梯度
* 实战：CartPole、MountainCar（Gym 环境）

---

### 学习资源

**Python 基础**

* B 站搜索“Python 入门”资源丰富

**机器学习**

* 【機器學習2021】預測本頻道觀看人數 (上)：[https://www.youtube.com/watch?v=Ye018rCVvOo\&list=PLJV\_el3uVTsMhtt7\_Y6sgTHGHp1Vb2P2J](https://www.youtube.com/watch?v=Ye018rCVvOo&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J)

**深度学习**

* [吴恩达深度学习 deeplearning.ai（偏理论）](https://www.bilibili.com/video/BV1FT4y1E74V)
* [七大机器学习项目实战（偏实践）](https://www.bilibili.com/video/BV1dM4m1y7KY)
* [PyTorch 深度学习快速入门【小土堆】](https://www.bilibili.com/video/BV1hE411t7RN)
* [动手学深度学习（李沐）](https://www.bilibili.com/video/BV1fsmyYnEfw)
* [Transformer 架构详解](https://www.bilibili.com/video/BV1xoJwzDESD)

**强化学习**

* [强化学习简明教程（理论 + 代码）](https://www.bilibili.com/video/BV1Ge4y1i7L6)
* [动手学强化学习](https://hrl.boyuai.com)

**平台与工具**

* 菜鸟教程：[https://www.runoob.com/](https://www.runoob.com/)
* 动手学深度学习：[https://zh.d2l.ai/](https://zh.d2l.ai/)
* PyTorch 教程：[https://www.runoob.com/pytorch/pytorch-tutorial.html](https://www.runoob.com/pytorch/pytorch-tutorial.html)

---

### 拓展与实践

**本地部署与微调**

* [三分钟部署 Ollama + 微调教程](https://www.bilibili.com/video/BV13e1jY9EmZ)
* [DeepSeek + LoRA + FastAPI 微调部署](https://www.bilibili.com/video/BV1R6P7eVEtd)

**语音合成 GPT-SoVITS**

* [GPT-SoVITS 本地部署教程](https://www.bilibili.com/video/BV116421M7pU)

**图像识别 & 生成**

* [YOLOv5 数据集构建实战](https://www.bilibili.com/video/BV18g4y1t7r2)
* [Stable Diffusion 原理与训练](https://www.bilibili.com/video/BV1x8411m76H)
* [NovelAI LoRA 训练教程](https://www.bilibili.com/video/BV1rhpFzrEHZ)

**进阶项目**

* QQ 群 AI 助手、MCP 任务调度系统
* 虚拟人物对话系统（类 J.A.R.V.I.S. 助手）
* 自主探索：脑机接口、跨模态生成模型

---

### 思考与研究方向

* 记录学习过程：每周总结关键概念、调试经验、灵感
* 研究“生成式认知主体”：探索 AI 如何通过交互逐步形成“我”的概念，研究感知、行动、时间性在人工智能中的涌现机制

---

### 杂项 & 趣味

* [图灵完备计算单元](https://www.bilibili.com/video/BV1eM4y1m7Mx)
* [渗透测试流程](https://www.bilibili.com/video/BV1Y3cceEEoh)
* [遗传算法趣味演示](https://www.bilibili.com/video/BV1dN4y1578u)
* [训练 AI 玩宝可梦（强化学习）](https://www.youtube.com/watch?v=DcYLT37ImBY)
* [Unity2018 入门教程（游戏开发入坑）](https://www.youtube.com/watch?v=99FwnTyyDJg)

---

### 参考资料

* 周北，《Python深度学习与项目实战》，人民邮电出版社，2021
* 伊恩·古德费洛等，《深度学习》，人民邮电出版社，2021
* Attention Is All You Need ([https://arxiv.org/pdf/1706.03762](https://arxiv.org/pdf/1706.03762))
* GoAI CSDN 深度学习知识点总结
* 以上列出的视频教程和开源项目链接
