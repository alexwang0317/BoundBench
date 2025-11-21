# models/hf_causal.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from vllm import LLM, SamplingParams

    _VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore
    _VLLM_AVAILABLE = False

@dataclass
class GenerateResult:
    texts: List[str]
    gen_token_ids: List[List[int]]  # only the newly generated tokens

class HFCausalLM:
    """
    Generic wrapper for HF causal LMs:
      - chat templating via tokenizer.apply_chat_telate (Qwen3, etc.)
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


class HFCausalLMvLLM:
    """
    Drop-in alternative to `HFCausalLM` backed by vLLM for faster inference.

    Mirrors the `sample` and `logprob` APIs using vLLM's high-throughput engine.
    """

    def __init__(
        self,
        model_name: str,
        *,
        tokenizer_name: Optional[str] = None,
        tensor_parallel_size: Optional[int] = None,
        dtype: Optional[str] = None,
        **llm_kwargs,
    ) -> None:
        if not _VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. Install with `pip install vllm` to use HFCausalLMvLLM."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        init_kwargs = dict(model=model_name, **llm_kwargs)
        if tensor_parallel_size is not None:
            init_kwargs["tensor_parallel_size"] = tensor_parallel_size
        if dtype is not None:
            init_kwargs["dtype"] = dtype
        if tokenizer_name is not None:
            init_kwargs["tokenizer"] = tokenizer_name

        self.llm = LLM(**init_kwargs)

    # ---------- Chat templating ----------
    def build_prompt_text(
        self,
        messages: List[Dict[str, str]],
        enable_thinking: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
        text = ""
        for m in messages:
            role = m.get("role", "user")
            content = m["content"]
            text += f"<<{role}>>: {content}\n"
        if add_generation_prompt:
            text += "<<assistant>>:"
        return text

    # ---------- Generation ----------
    def sample(
        self,
        messages: List[Dict[str, str]],
        n: int = 1,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        enable_thinking: bool = False,
    ) -> GenerateResult:
        prompt = self.build_prompt_text(messages, enable_thinking=enable_thinking, add_generation_prompt=True)
        if not do_sample:
            temperature = 0.0
            top_p = 1.0
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            logprobs=0,
        )
        outputs = self.llm.generate([prompt], sampling_params)
        request_output = outputs[0]
        texts: List[str] = []
        gen_token_ids: List[List[int]] = []
        for seq in request_output.outputs:
            texts.append(seq.text)
            gen_token_ids.append(seq.token_ids)
        return GenerateResult(texts=texts, gen_token_ids=gen_token_ids)

    # ---------- Teacher-forced log-prob ----------
    def logprob(
        self,
        messages: List[Dict[str, str]],
        completion: str,
        enable_thinking: bool = False,
    ) -> Tuple[float, List[float]]:
        prompt_text = self.build_prompt_text(messages, enable_thinking=enable_thinking, add_generation_prompt=True)
        full_text = prompt_text + completion

        prompt_ids = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0].tolist()
        full_ids = self.tokenizer(
            full_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0].tolist()
        comp_ids = full_ids[len(prompt_ids) :]
        if not comp_ids:
            return 0.0, []

        sampling_params = SamplingParams(
            n=1,
            max_tokens=0,
            temperature=0.0,
            top_p=1.0,
            prompt_logprobs=1,
            logprobs=0,
        )
        outputs = self.llm.generate([full_text], sampling_params)
        request_output = outputs[0]
        prompt_token_ids = request_output.prompt_token_ids  
        prompt_logprobs = request_output.prompt_logprobs    

        start_idx = len(prompt_ids)
        token_logps: List[float] = []
        for idx in range(start_idx, len(prompt_token_ids)):
            entry = prompt_logprobs[idx]
            logp = self._extract_token_logprob(entry, prompt_token_ids[idx])
            if logp is None:
                raise RuntimeError(
                    f"Token {prompt_token_ids[idx]} not found in prompt_logprobs at position {idx}. "
                    f"This may indicate a tokenization mismatch or vLLM API issue."
                )
            token_logps.append(logp)
        return float(sum(token_logps)), token_logps

    @staticmethod
    def _extract_token_logprob(entry, target_token_id: int) -> Optional[float]:
        """
        Handle the different logprob entry shapes returned by vLLM (dict or object).
        """
        if entry is None:
            return None
        if isinstance(entry, dict):
            for candidate in entry.values():
                cand_id = getattr(candidate, "token_id", None)
                if cand_id == target_token_id:
                    return float(candidate.logprob)
        elif hasattr(entry, "token_id"):
            if getattr(entry, "token_id") == target_token_id:
                return float(entry.logprob)
        return None