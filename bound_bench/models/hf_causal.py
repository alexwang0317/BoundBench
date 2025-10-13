# models/hf_causal.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

@dataclass
class GenerateResult:
    texts: List[str]
    gen_token_ids: List[List[int]]  # only the newly generated tokens

class HFCausalLM:
    """
    Generic wrapper for HF causal LMs:
      - chat templating via tokenizer.apply_chat_template (Qwen3, etc.)
      - sampling
      - teacher-forced log-probs for completion given messages
    """
    def __init__(self, model_name: str, device: Optional[str] = None, dtype: Optional[str] = "auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=("auto" if dtype == "auto" else getattr(torch, dtype) if isinstance(dtype, str) else dtype),
            device_map="auto" if device is None else None
        )
        if device is not None:
            self.model.to(device)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.eval()

    # ---------- Chat templating ----------
    def build_prompt_text(
        self,
        messages: List[Dict[str, str]],
        enable_thinking: bool = False,
        add_generation_prompt: bool = True
    ) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking
            )
        # Fallback for non-chat models
        text = ""
        for m in messages:
            role = m.get("role", "user")
            content = m["content"]
            text += f"<<{role}>>: {content}\n"
        if add_generation_prompt:
            text += "<<assistant>>:"
        return text

    # ---------- Generation ----------
    @torch.no_grad()
    def sample(
        self,
        messages: List[Dict[str, str]],
        n: int = 1,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        enable_thinking: bool = False
    ) -> GenerateResult:
        text = self.build_prompt_text(messages, enable_thinking=enable_thinking, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
        in_len = inputs.input_ids.shape[1]
        texts, gen_ids = [], []
        for seq in out.sequences:
            new_ids = seq[in_len:].tolist()
            gen_ids.append(new_ids)
            texts.append(self.tokenizer.decode(new_ids, skip_special_tokens=True))
        return GenerateResult(texts=texts, gen_token_ids=gen_ids)

    # ---------- Teacher-forced log-prob ----------
    @torch.no_grad()
    def logprob(
        self,
        messages: List[Dict[str, str]],
        completion: str,
        enable_thinking: bool = False
    ) -> Tuple[float, List[float]]:
        """
        Return (sum_logprob, per_token_logprobs) for p(completion | messages).
        """
        prompt_text = self.build_prompt_text(messages, enable_thinking=enable_thinking, add_generation_prompt=True)
        full_text = prompt_text + completion

        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(self.model.device).input_ids[0]
        full_ids   = self.tokenizer(full_text,   return_tensors="pt", add_special_tokens=False).to(self.model.device).input_ids[0]
        comp_ids   = full_ids[len(prompt_ids):]
        if comp_ids.numel() == 0:
            return 0.0, []

        outputs = self.model(full_ids.unsqueeze(0))
        logits = outputs.logits.squeeze(0)  # [T, V]

        # For each completion token, use logits at the immediately preceding position
        start = max(len(prompt_ids) - 1, 0)
        end   = start + len(comp_ids)
        step_logits = logits[start:end, :]                    # [len(comp_ids), V]
        logprobs = F.log_softmax(step_logits, dim=-1)
        token_lps = logprobs[torch.arange(len(comp_ids)), comp_ids]
        # Cast to float32 before summing to avoid bfloat16 accumulation errors
        token_lps_f32 = token_lps.float()
        return float(token_lps_f32.sum().item()), [float(x) for x in token_lps.tolist()]
        