# train_v2t_adapter_sft.py
"""
Stage B-1: SFT for Vector-to-Text Adapter
- Encoder: 已训练的 StyleEncoder (Text -> zuq)
- Decoder: 冻结的 Qwen2-0.5B-Instruct
- Adapter: 小 MLP，把 zuq -> prefix embeddings
训练目标：在 zuq 条件下，生成合理的 Persona + Explanation 文本。
"""

import os
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

from style_encoder import StyleEncoder, StyleEncoderConfig

BASE_MODEL_DIR = "./models/Qwen2-0.5B-Instruct"
ENCODER_CKPT = "./style_encoder_ckpt/style_encoder.pt"
ADAPTER_SFT_DIR = "./v2t_adapter_sft"


# ======== 1. 手工 persona SFT 数据集 ========
class V2TSFTDataset(Dataset):
    def __init__(self, tokenizer, max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples: List[Dict] = self._build_data()

    def _build_data(self) -> List[Dict]:
        data: List[Dict] = []

        # User 1：技术向，偏好简洁 + 条列式 + 中性
        history1 = [
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
        persona1 = (
            "Persona:\n"
            "The user prefers concise technical answers. They like their explanations organized into short bullet points "
            "or numbered steps, instead of long paragraphs. The tone should remain neutral and factual.\n\n"
            "Root Explanation:\n"
            "This persona is inferred from the user's requests for short, clear differences and the use of numbered "
            "structure in previous helpful responses.\n"
        )

        # User 2：生活向，偏好简短 + 友好 + 段落
        history2 = [
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
        persona2 = (
            "Persona:\n"
            "The user prefers brief, practical suggestions for daily-life questions. They are comfortable with a "
            "friendly and supportive tone, as long as the advice is direct and easy to follow.\n\n"
            "Root Explanation:\n"
            "This persona is inferred from their previous requests for shorter answers and their positive response "
            "to simple, actionable outfit suggestions.\n"
        )

        data.append({"history": history1, "query": query1, "persona": persona1})
        data.append({"history": history2, "query": query2, "persona": persona2})

        # 再复制一份，凑 4 条
        data.append({"history": history1, "query": "解释一下指令微调，大概说一下就行。", "persona": persona1})
        data.append({"history": history2, "query": "下雨天去图书馆穿什么比较合适？", "persona": persona2})

        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]

        # encoder 输入：history + query
        history_text = ""
        for t in ex["history"]:
            history_text += f"User: {t['user']}\nAssistant: {t['assistant']}\n"
        enc_input_text = history_text + f"\n[Current Query]\n{ex['query']}"

        # persona 生成 prompt
        prompt = (
            "You are a model that infers user stylistic preferences.\n\n"
            "[User History]\n"
            f"{history_text}\n"
            "[Current Query]\n"
            f"{ex['query']}\n\n"
            "Persona and Explanation:\n"
        )

        full_text = prompt + ex["persona"]

        # tokenization
        enc_enc = self.tokenizer(
            enc_input_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
        )
        prompt_enc = self.tokenizer(prompt, truncation=True, max_length=self.max_len)
        full_enc = self.tokenizer(full_text, truncation=True, max_length=self.max_len)

        input_ids = full_enc["input_ids"]
        labels = full_enc["input_ids"].copy()

        # mask 掉 prompt 部分，只在 persona 上计算 loss
        prompt_len = len(prompt_enc["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len

        attention_mask = full_enc["attention_mask"]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "enc_input_ids": torch.tensor(enc_enc["input_ids"], dtype=torch.long),
            "enc_attention_mask": torch.tensor(enc_enc["attention_mask"], dtype=torch.long),
        }


# ======== 2. Adapter + 冻结模型 ========
class V2TAdapterModel(nn.Module):
    def __init__(self, base_model, style_encoder, latent_dim=3, prefix_len=4):
        super().__init__()
        self.base_model = base_model
        self.style_encoder = style_encoder
        self.latent_dim = latent_dim
        self.prefix_len = prefix_len

        hidden_size = base_model.config.hidden_size
        self.adapter = nn.Sequential(
            nn.Linear(latent_dim, hidden_size * 4),
            nn.Tanh(),
            nn.Linear(hidden_size * 4, hidden_size * prefix_len),
        )

        # 冻结 base_model 和 style_encoder
        for p in self.base_model.parameters():
            p.requires_grad = False
        for p in self.style_encoder.parameters():
            p.requires_grad = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        enc_input_ids=None,
        enc_attention_mask=None,
    ):
        """
        input_ids, attention_mask, labels: 用于生成 persona 的 prompt+目标
        enc_input_ids, enc_attention_mask: 用于 StyleEncoder (Text->zuq)
        """
        device = input_ids.device

        # 1) 计算 zuq
        with torch.no_grad():
            zuq = self.style_encoder(enc_input_ids, enc_attention_mask)  # [B, latent_dim]

        # 2) zuq -> prefix embeddings
        prefix = self.adapter(zuq)  # [B, prefix_len * hidden]
        hidden_size = self.base_model.config.hidden_size
        prefix = prefix.view(-1, self.prefix_len, hidden_size)  # [B, P, H]

        # 3) 把 prefix 拼到 input embeddings 前面
        input_embeds = self.base_model.get_input_embeddings()(input_ids)  # [B, L, H]
        full_embeds = torch.cat([prefix, input_embeds], dim=1)           # [B, P+L, H]

        # attention mask
        batch_size = input_ids.size(0)
        prefix_mask = torch.ones(batch_size, self.prefix_len, device=device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # labels 前面补 -100
        prefix_labels = torch.full(
            (batch_size, self.prefix_len),
            -100,
            dtype=labels.dtype,
            device=device,
        )
        full_labels = torch.cat([prefix_labels, labels], dim=1)

        outputs = self.base_model(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )
        return outputs


def data_collator_sft(features: List[Dict]):
    # padding 到 batch 内最大长度
    keys = ["input_ids", "attention_mask", "labels", "enc_input_ids", "enc_attention_mask"]
    batch = {}
    max_len = max(f["input_ids"].shape[0] for f in features)

    for key in keys:
        tensors = [f[key] for f in features]
        if key in ["enc_input_ids", "enc_attention_mask"]:
            # 编码器那边已经是 max_length 固定长度，不需要再 pad
            batch[key] = torch.stack(tensors, dim=0)
        else:
            # 针对 decoder 的输入做 pad
            padded = []
            for t in tensors:
                pad_len = max_len - t.shape[0]
                if pad_len > 0:
                    if key == "labels":
                        pad_val = -100
                    elif key == "attention_mask":
                        pad_val = 0
                    else:
                        pad_val = 0
                    t = torch.cat([t, torch.full((pad_len,), pad_val, dtype=t.dtype)], dim=0)
                padded.append(t)
            batch[key] = torch.stack(padded, dim=0)

    return batch


def main():
    os.makedirs(ADAPTER_SFT_DIR, exist_ok=True)

    # 1) tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) dataset
    dataset = V2TSFTDataset(tokenizer)
    print("Num SFT samples:", len(dataset))

    # 3) 加载 StyleEncoder
    ckpt = torch.load(ENCODER_CKPT, map_location="cpu")
    cfg = ckpt["config"]
    enc_config = StyleEncoderConfig(
        vocab_size=cfg["vocab_size"],
        latent_dim=cfg["latent_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
    )
    style_encoder = StyleEncoder(enc_config)
    style_encoder.load_state_dict(ckpt["state_dict"])

    # 4) 加载冻结的 Qwen2-0.5B
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    # 5) 组装 Adapter 模型
    model = V2TAdapterModel(base_model, style_encoder, latent_dim=enc_config.latent_dim, prefix_len=4)
    model.to(device)

    # 6) TrainingArguments + Trainer
    training_args = TrainingArguments(
        output_dir=ADAPTER_SFT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        num_train_epochs=5,
        logging_steps=1,
        save_strategy="no",  # ❗ 不自动存 checkpoint
        save_safetensors=False,  # ❗ 避免 safetensors 抱怨共享权重
        fp16=(device == "cuda"),
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator_sft,
    )

    trainer.train()

    # 只保存 adapter 权重
    torch.save(
        {
            "latent_dim": model.latent_dim,
            "prefix_len": model.prefix_len,
            "hidden_size": base_model.config.hidden_size,
            "state_dict": model.adapter.state_dict(),
        },
        os.path.join(ADAPTER_SFT_DIR, "v2t_adapter_sft.pt"),
    )
    print(f"SFT adapter saved to {ADAPTER_SFT_DIR}/v2t_adapter_sft.pt")


if __name__ == "__main__":
    main()
