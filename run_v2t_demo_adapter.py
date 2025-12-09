# run_v2t_demo_adapter.py
"""
Inference with full Vector-to-Text pipeline:

History + Query
  -> StyleEncoder (Text->zuq)
  -> Adapter (zuq->prefix)
  -> Qwen2-0.5B generates Persona + Root Explanation
  -> Persona + Query -> Black-box Answer
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from style_encoder import StyleEncoder, StyleEncoderConfig

BASE_MODEL_DIR = "./models/Qwen2-0.5B-Instruct"
ENCODER_CKPT = "./style_encoder_ckpt/style_encoder.pt"
ADAPTER_ORPO_CKPT = "./v2t_adapter_orpo/v2t_adapter_orpo.pt"   # 用 ORPO 后的 adapter


class V2TAdapterGenerator:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # StyleEncoder
        enc_ckpt = torch.load(ENCODER_CKPT, map_location="cpu")
        cfg = enc_ckpt["config"]
        enc_config = StyleEncoderConfig(
            vocab_size=cfg["vocab_size"],
            latent_dim=cfg["latent_dim"],
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
        )
        self.style_encoder = StyleEncoder(enc_config)
        self.style_encoder.load_state_dict(enc_ckpt["state_dict"])
        self.style_encoder.to(self.device)
        self.style_encoder.eval()

        # Qwen2-0.5B 作为 persona 生成 + black-box
        self.llm = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_DIR,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        self.llm.eval()

        # Adapter
        adapter_ckpt = torch.load(ADAPTER_ORPO_CKPT, map_location="cpu")
        self.latent_dim = adapter_ckpt["latent_dim"]
        self.prefix_len = adapter_ckpt["prefix_len"]
        hidden_size = adapter_ckpt["hidden_size"]

        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, hidden_size * 4),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size * 4, hidden_size * self.prefix_len),
        )
        self.adapter.load_state_dict(adapter_ckpt["state_dict"])
        self.adapter.to(self.device)
        self.adapter.eval()

    def encode_history_query(self, history_text, query, max_len=256):
        full = history_text + f"\n[Current Query]\n{query}"
        enc = self.tokenizer(
            full,
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long, device=self.device).unsqueeze(0)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long, device=self.device).unsqueeze(0)
        return input_ids, attention_mask

    def infer_zuq(self, history_text, query):
        enc_ids, enc_mask = self.encode_history_query(history_text, query)
        with torch.no_grad():
            zuq = self.style_encoder(enc_ids, enc_mask)[0]  # [latent_dim]
        return zuq

    def generate_persona(self, history_text, query, max_new_tokens=160, temperature=0.4):
        """
        使用 adapter + Qwen，根据 zuq 生成 Persona + Root Explanation。
        """
        # 1) 计算 zuq
        enc_ids, enc_mask = self.encode_history_query(history_text, query)
        with torch.no_grad():
            zuq = self.style_encoder(enc_ids, enc_mask)  # [1, latent_dim]

        # 2) zuq -> prefix embeddings
        hidden_size = self.llm.config.hidden_size
        with torch.no_grad():
            prefix = self.adapter(zuq)  # [1, P*H]
            prefix = prefix.view(1, self.prefix_len, hidden_size)  # [1, P, H]

        # 3) persona 生成 prompt（不带 persona 本身）
        prompt = (
            "You are a model that infers user stylistic preferences.\n\n"
            "[User History]\n"
            f"{history_text}\n"
            "[Current Query]\n"
            f"{query}\n\n"
            "Please describe:\n"
            "1) A Persona about the user's preferred answering style.\n"
            "2) A Root Explanation linking this persona to their past behavior.\n\n"
            "Persona and Explanation:\n"
        )
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_ids = enc["input_ids"]  # [1, Lp]
        prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)  # [1, Lp, H]

        # 拼接 prefix + prompt embeddings
        full_embeds = torch.cat([prefix, prompt_embeds], dim=1)  # [1, P+Lp, H]
        attn_mask = torch.ones(1, full_embeds.size(1), device=self.device, dtype=torch.long)

        # 4) generate
        with torch.no_grad():
            outputs = self.llm.generate(
                inputs_embeds=full_embeds,
                attention_mask=attn_mask,
                do_sample=True,
                top_p=0.9,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.1,
            )

        # decode：去掉 prefix+prompt 的部分，只看新生成的 token
        # 由于 generate 返回的是 token ids（不是 embeds），我们需要再算一次长度：
        # Qwen 会在内部把 inputs_embeds 映射到一个“虚拟”起始长度，这里简化处理：
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 简单策略：从第一次出现 "Persona" 位置开始截取
        lower = text.lower()
        idx = lower.find("persona")
        if idx != -1:
            text = text[idx:]
        return text.strip()

    def blackbox_answer(self, persona_text, query, temperature=0.4):
        """
        使用 persona 作为前缀，生成最终答案。
        """
        prompt = f"""
You are a helpful assistant.

[Persona describing the user's preferred style]
{persona_text}

[User Query]
{query}

Your task:
- Follow the persona strictly (tone, length, structure).
- Answer the query directly.
- Do NOT repeat the persona.
- Output only the final answer.

Final answer:
"""
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.llm.generate(
                **enc,
                do_sample=True,
                top_p=0.9,
                temperature=temperature,
                max_new_tokens=256,
                repetition_penalty=1.1,
            )
        out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return out[len(prompt):].strip()


if __name__ == "__main__":
    gen = V2TAdapterGenerator()

    # 构造和之前类似的 history
    history = """User: 解释一下 LoRA 是什么？
Assistant: LoRA 在冻结大模型参数的前提下加入低秩矩阵，从而大幅减少训练开销。
User: DPO 和 PPO 简单说说区别？最好条理清晰一点。
Assistant: 1. PPO 需要 actor-critic；2. DPO 直接在偏好数据上优化；3. 实现更简单。
"""

    q1 = "Explain the attention mechanism in Transformer."
    q2 = "北京今天 10 度左右，上课适合怎么穿？"

    # ===== Technical Question =====
    print("\n===== Technical Question Example (Vector → Text) =====")
    zuq1 = gen.infer_zuq(history, q1)
    print("Latent style vector (zuq):", torch.sigmoid(zuq1).tolist())

    persona1 = gen.generate_persona(history, q1)
    print("\nGenerated Persona + Explanation:\n", persona1)

    ans1 = gen.blackbox_answer(persona1, q1)
    print("\n--- Final Answer (black-box LLM) ---\n", ans1)

    # ===== Daily-life Question =====
    print("\n\n===== Daily-life Question Example (Vector → Text) =====")
    zuq2 = gen.infer_zuq(history, q2)
    print("Latent style vector (zuq):", torch.sigmoid(zuq2).tolist())

    persona2 = gen.generate_persona(history, q2)
    print("\nGenerated Persona + Explanation:\n", persona2)

    ans2 = gen.blackbox_answer(persona2, q2)
    print("\n--- Final Answer (black-box LLM) ---\n", ans2)
