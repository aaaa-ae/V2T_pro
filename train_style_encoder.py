# train_style_encoder.py
import os
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer

from style_encoder import StyleEncoder, StyleEncoderConfig

BASE_MODEL_DIR = "./models/Qwen2-0.5B-Instruct"
ENCODER_OUTPUT_DIR = "./style_encoder_ckpt"


# ====== 1. 手工数据集（和之前类似） ======
class StyleDemoDataset(Dataset):
    def __init__(self, tokenizer, max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples: List[Dict] = self._build_data()

    def _build_data(self) -> List[Dict]:
        data: List[Dict] = []

        # User 1：技术宅，喜欢简洁 + 条列式 + 中性
        hist1 = [
            {
                "user": "解释一下 LoRA 是什么？",
                "assistant": "LoRA 在冻结大模型参数的前提下加入低秩矩阵，从而大幅减少训练开销。",
            },
            {
                "user": "DPO 和 PPO 简单说说区别？",
                "assistant": "1. PPO 需要 actor-critic；2. DPO 直接用偏好数据；3. 实现更简单。",
            },
        ]
        query1 = "请简单说一下 DPO 和 PPO 的区别。"
        resp_pref1 = "用户希望较为简短、有结构的技术解释，比如使用 1、2、3 这样的条列式回答。"
        data.append({
            "history": hist1,
            "query": query1,
            "preferred_resp": resp_pref1,
        })

        # User 2：生活向，喜欢 friendly + 简短 + 段落
        hist2 = [
            {
                "user": "今天北京穿什么？",
                "assistant": "建议长袖 + 薄外套，温度有点低。",
            },
            {
                "user": "可以再简短点吗？",
                "assistant": "长袖 + 外套就可以。",
            },
        ]
        query2 = "明天如果降温，适合怎么穿去上课？"
        resp_pref2 = "用户喜欢友好的语气、简短又可执行的穿衣建议。"
        data.append({
            "history": hist2,
            "query": query2,
            "preferred_resp": resp_pref2,
        })

        # 再复制两份稍作改写，凑 4 条（toy demo）
        data.append({
            "history": hist1,
            "query": "解释一下什么是指令微调，简单一点。",
            "preferred_resp": resp_pref1,
        })
        data.append({
            "history": hist2,
            "query": "下雨天去图书馆穿什么比较合适？",
            "preferred_resp": resp_pref2,
        })

        return data

    def _style_to_vector(self, text: str):
        """
        把偏好响应文本映射成一个 3 维风格向量:
        [concise(1)/detailed(0), structured(1)/paragraph(0), friendly(1)/neutral(0)]
        """
        # 长度
        concise = 1.0 if len(text.split()) < 40 else 0.0

        # 结构
        structured = 1.0 if any(b in text for b in ["•", "-", "1.", "2.", "3"]) else 0.0

        # 语气
        lower = text.lower()
        friendly = 1.0 if any(w in lower for w in ["友好", "建议", "可以", "别担心", "放心"]) else 0.0

        return torch.tensor([concise, structured, friendly], dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]

        # 拼接 history + query 作为 Encoder 输入
        history_text = ""
        for t in ex["history"]:
            history_text += f"User: {t['user']}\nAssistant: {t['assistant']}\n"
        full_input = history_text + f"\n[Current Query]\n{ex['query']}"

        enc = self.tokenizer(
            full_input,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
        )
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)

        style_vec = self._style_to_vector(ex["preferred_resp"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "style_vec": style_vec,
        }


def train_encoder(num_epochs: int = 30, lr: float = 1e-3, batch_size: int = 2):
    os.makedirs(ENCODER_OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    dataset = StyleDemoDataset(tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    config = StyleEncoderConfig(
        vocab_size=len(tokenizer),
        latent_dim=3,      # 和 style_vec 维度对齐
        hidden_dim=256,
        n_layers=2,
        n_heads=4,
    )
    model = StyleEncoder(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()  # 3 维上做二元交叉熵

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_style = batch["style_vec"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)   # [B, 3]

            loss = loss_fn(logits, target_style)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    # 保存
    torch.save(
        {
            "config": {
                "vocab_size": config.vocab_size,
                "latent_dim": config.latent_dim,
                "hidden_dim": config.hidden_dim,
                "n_layers": config.n_layers,
                "n_heads": config.n_heads,
            },
            "state_dict": model.state_dict(),
        },
        os.path.join(ENCODER_OUTPUT_DIR, "style_encoder.pt"),
    )
    print(f"Encoder saved to {ENCODER_OUTPUT_DIR}/style_encoder.pt")


if __name__ == "__main__":
    train_encoder()
