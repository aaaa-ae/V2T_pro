# train_v2t_adapter_orpo.py
"""
Stage B-2: Preference fine-tuning (ORPO-style) for Vector-to-Text Adapter

- 使用已经 SFT 过的 adapter 初始权重
- 冻结 StyleEncoder + Qwen2-0.5B
- 对比 (chosen persona, rejected persona)，让 chosen 的 logp 更高
"""

import os
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM

from style_encoder import StyleEncoder, StyleEncoderConfig

BASE_MODEL_DIR = "./models/Qwen2-0.5B-Instruct"
ENCODER_CKPT = "./style_encoder_ckpt/style_encoder.pt"
ADAPTER_SFT_PATH = "./v2t_adapter_sft/v2t_adapter_sft.pt"
ADAPTER_ORPO_DIR = "./v2t_adapter_orpo"


# ======== 1. Pairwise 偏好数据集 ========
class V2TORPODataset(Dataset):
    def __init__(self, tokenizer, max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples: List[Dict] = self._build_data()

    def _build_data(self) -> List[Dict]:
        data: List[Dict] = []

        # User 1：技术向，喜欢 concise + bullet + neutral
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

        chosen1 = (
            "Persona:\n"
            "The user prefers concise technical answers with clear bullet points or short numbered lists. "
            "The tone should be neutral and focused on key differences.\n\n"
            "Root Explanation:\n"
            "This persona matches the user's preference for short, structured comparisons between methods.\n"
        )
        rejected1 = (
            "Persona:\n"
            "The user prefers long, narrative explanations with rich storytelling and informal jokes.\n\n"
            "Root Explanation:\n"
            "This persona assumes the user enjoys lengthy, casual discussions, which is not supported by their history.\n"
        )

        # User 2：生活向，喜欢 brief + friendly
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

        chosen2 = (
            "Persona:\n"
            "The user prefers brief, practical outfit suggestions with a friendly and supportive tone.\n\n"
            "Root Explanation:\n"
            "This persona reflects the user's repeated requests for shorter, actionable clothing advice.\n"
        )
        rejected2 = (
            "Persona:\n"
            "The user prefers extremely detailed fashion analysis with formal, academic language.\n\n"
            "Root Explanation:\n"
            "This persona assumes the user wants in-depth theoretical discussion, which is not observed in their history.\n"
        )

        data.append({"history": history1, "query": query1, "chosen": chosen1, "rejected": rejected1})
        data.append({"history": history2, "query": query2, "chosen": chosen2, "rejected": rejected2})

        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]

        history_text = ""
        for t in ex["history"]:
            history_text += f"User: {t['user']}\nAssistant: {t['assistant']}\n"

        enc_input_text = history_text + f"\n[Current Query]\n{ex['query']}"

        prompt = (
            "You are a model that infers user stylistic preferences.\n\n"
            "[User History]\n"
            f"{history_text}\n"
            "[Current Query]\n"
            f"{ex['query']}\n\n"
            "Persona and Explanation:\n"
        )

        # 编码器输入
        enc_enc = self.tokenizer(
            enc_input_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
        )

        # prompt ids
        prompt_enc = self.tokenizer(prompt, truncation=True, max_length=self.max_len)

        # chosen / rejected persona ids（不加 BOS）
        chosen_ids = self.tokenizer(
            ex["chosen"], truncation=True, max_length=self.max_len, add_special_tokens=False
        )["input_ids"]
        rejected_ids = self.tokenizer(
            ex["rejected"], truncation=True, max_length=self.max_len, add_special_tokens=False
        )["input_ids"]

        return {
            "enc_input_ids": torch.tensor(enc_enc["input_ids"], dtype=torch.long),
            "enc_attention_mask": torch.tensor(enc_enc["attention_mask"], dtype=torch.long),
            "prompt_ids": torch.tensor(prompt_enc["input_ids"], dtype=torch.long),
            "chosen_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "rejected_ids": torch.tensor(rejected_ids, dtype=torch.long),
        }


# ======== 2. Adapter 模型（与 SFT 相同结构） ========
class V2TAdapterForORPO(nn.Module):
    def __init__(self, base_model, style_encoder, adapter_cfg):
        super().__init__()
        self.base_model = base_model
        self.style_encoder = style_encoder

        self.latent_dim = adapter_cfg["latent_dim"]
        self.prefix_len = adapter_cfg["prefix_len"]
        hidden_size = adapter_cfg["hidden_size"]

        self.adapter = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_size * 4),
            nn.Tanh(),
            nn.Linear(hidden_size * 4, hidden_size * self.prefix_len),
        )
        self.adapter.load_state_dict(adapter_cfg["state_dict"])

        # 冻结 base_model & encoder
        for p in self.base_model.parameters():
            p.requires_grad = False
        for p in self.style_encoder.parameters():
            p.requires_grad = False

    def build_prefix(self, zuq):
        hidden_size = self.base_model.config.hidden_size
        prefix = self.adapter(zuq)  # [B, P*H]
        prefix = prefix.view(-1, self.prefix_len, hidden_size)
        return prefix

    def compute_logp(self, prefix, prompt_ids, persona_ids):
        """
        计算在 prefix + prompt 条件下，persona_ids 的 log 概率总和。
        prefix: [B, P, H]
        prompt_ids: [B, Lp]
        persona_ids: [B, Lr]
        """
        device = prefix.device
        emb = self.base_model.get_input_embeddings()

        prompt_emb = emb(prompt_ids)       # [B, Lp, H]
        persona_emb = emb(persona_ids)     # [B, Lr, H]

        full_embeds = torch.cat([prefix, prompt_emb, persona_emb], dim=1)  # [B, P+Lp+Lr, H]

        B = prefix.size(0)
        P = prefix.size(1)
        Lp = prompt_emb.size(1)
        Lr = persona_emb.size(1)

        attn_mask = torch.ones(B, P + Lp + Lr, device=device, dtype=torch.long)

        outputs = self.base_model(
            inputs_embeds=full_embeds,
            attention_mask=attn_mask,
        )
        logits = outputs.logits  # [B, T, V]

        # 只在 persona token 上计算 logp
        log_probs = torch.log_softmax(logits, dim=-1)  # [B, T, V]

        # persona 的起始位置 index：P + Lp
        start = P + Lp
        # 我们要用前一个 token 的 logits 来预测当前 persona token
        # 对于 persona 的第 t 个 token，其 logits 位置 = start + t - 1
        idxs = torch.arange(Lr, device=device)
        time_idx = start + idxs - 1  # [Lr]
        time_idx[0] = start - 1      # 第一个 persona token 用 prompt 的最后一个 token logits

        # gather
        # log_probs: [B, T, V] -> [B, Lr, V]
        selected_logits = log_probs[:, time_idx, :]
        # persona_ids: [B, Lr] -> gather
        lp = torch.gather(selected_logits, dim=-1, index=persona_ids.unsqueeze(-1)).squeeze(-1)  # [B, Lr]

        # 对 persona 所有 token 求和
        logp_sum = lp.sum(dim=-1)  # [B]
        return logp_sum


def collate_orpo(batch: List[Dict]):
    # 针对 prompt_ids, chosen_ids, rejected_ids 做 pad
    enc_input_ids = torch.stack([b["enc_input_ids"] for b in batch], dim=0)
    enc_attention_mask = torch.stack([b["enc_attention_mask"] for b in batch], dim=0)

    # 动态 pad
    def pad_1d(tensors, pad_val=0):
        max_len = max(t.size(0) for t in tensors)
        out = []
        for t in tensors:
            pad = max_len - t.size(0)
            if pad > 0:
                t = torch.cat([t, torch.full((pad,), pad_val, dtype=t.dtype)], dim=0)
            out.append(t)
        return torch.stack(out, dim=0)

    prompt_ids = pad_1d([b["prompt_ids"] for b in batch], pad_val=0)
    chosen_ids = pad_1d([b["chosen_ids"] for b in batch], pad_val=0)
    rejected_ids = pad_1d([b["rejected_ids"] for b in batch], pad_val=0)

    return {
        "enc_input_ids": enc_input_ids,
        "enc_attention_mask": enc_attention_mask,
        "prompt_ids": prompt_ids,
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
    }


def main():
    os.makedirs(ADAPTER_ORPO_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) tokenizer + dataset
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = V2TORPODataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_orpo)

    # 2) StyleEncoder
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
    style_encoder.to(device)
    style_encoder.eval()

    # 3) 冻结 Qwen2-0.5B
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    for p in base_model.parameters():
        p.requires_grad = False

    # 4) 加载 SFT adapter 权重
    adapter_ckpt = torch.load(ADAPTER_SFT_PATH, map_location="cpu")

    model = V2TAdapterForORPO(base_model, style_encoder, adapter_ckpt)
    model.to(device)

    # 5) 优化器（只优化 adapter）
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=5e-4)
    beta = 0.1  # ORPO / DPO 风格温度

    model.train()
    num_epochs = 10

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_samples = 0

        for batch in dataloader:
            enc_input_ids = batch["enc_input_ids"].to(device)
            enc_attention_mask = batch["enc_attention_mask"].to(device)
            prompt_ids = batch["prompt_ids"].to(device)
            chosen_ids = batch["chosen_ids"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)

            B = enc_input_ids.size(0)

            with torch.no_grad():
                zuq = style_encoder(enc_input_ids, enc_attention_mask)  # [B, latent_dim]

            prefix = model.build_prefix(zuq)  # [B, P, H]

            # 为了避免 padding token 影响 logp，persona 部分的 padding 用 EOS 代替
            eos_id = tokenizer.eos_token_id
            chosen_ids_mask = (chosen_ids != 0).long()
            rejected_ids_mask = (rejected_ids != 0).long()

            chosen_ids_clean = chosen_ids.clone()
            chosen_ids_clean[chosen_ids_clean == 0] = eos_id

            rejected_ids_clean = rejected_ids.clone()
            rejected_ids_clean[rejected_ids_clean == 0] = eos_id

            logp_chosen = model.compute_logp(prefix, prompt_ids, chosen_ids_clean)  # [B]
            logp_rejected = model.compute_logp(prefix, prompt_ids, rejected_ids_clean)  # [B]

            # 简化版 ORPO / DPO 损失
            # loss = -log σ( β (logp_chosen - logp_rejected) )
            prefs = logp_chosen - logp_rejected
            loss = -torch.log(torch.sigmoid(beta * prefs)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            num_samples += B

        avg_loss = total_loss / max(1, num_samples)
        print(f"Epoch {epoch+1}/{num_epochs} - ORPO-like loss: {avg_loss:.4f}")

    # 保存 ORPO 之后的 adapter
    torch.save(
        {
            "latent_dim": model.latent_dim,
            "prefix_len": model.prefix_len,
            "hidden_size": base_model.config.hidden_size,
            "state_dict": model.adapter.state_dict(),
        },
        os.path.join(ADAPTER_ORPO_DIR, "v2t_adapter_orpo.pt"),
    )

    print(f"ORPO adapter saved to {ADAPTER_ORPO_DIR}/v2t_adapter_orpo.pt")


if __name__ == "__main__":
    main()
