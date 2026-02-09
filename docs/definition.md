### 问题定义
#### 🧱 第 1 层（核心研究问题，只有这一层算 idea）

能不能学习一个 user-task-adaptive、可解释的 persona，
作为黑盒 LLM 的中间控制抽象？

这一层里只有 4 个关键词：

1、user-task-adaptive（不仅是 user-level）

2、persona（不是 embedding / summary）

3、black-box compatible

4、SFT+ORPO（需要知识蒸馏）

👉 这就是你的论文“灵魂”，一句话能讲清楚。

#### 🧰 第 2 层（方法 instantiation，不是创新点）

这里你可以非常“工程化”，完全不用内疚：

1、encoder：轻量 Transformer encoder（2–4 层） + attention pooling（query-conditioned）

2、decoder：Qwen2.5-1.5B-Instruct

3、teacher：Qwen2.5-7B-Instruct（目标是得到persona/explanation 语言结构）

4、中心黑盒：GPT-4o-mini

4、preference learning：ORPO

5、dataset：PersonalLLM

### TODO:
1、看论文

2、修改完善draft（大修）

3、画图展示idea：便于捋清思路（有哪些模块？怎么端到端训练学习？）

4、修改demo：demo到底要验证什么？
