"""LLMs used in the paper and their first-token output distribution (Eq. 1)."""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "qwen2": "Qwen/Qwen2-7B-Instruct",
    "qwen3": "Qwen/Qwen3-8B",
    "gemma": "google/gemma-7b-it",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}


def load_tokenizer(name):
    return AutoTokenizer.from_pretrained(MODELS.get(name, name))


def load_model(name):
    return AutoModelForCausalLM.from_pretrained(MODELS.get(name, name), torch_dtype="auto", device_map="cuda")


@torch.no_grad()
def first_token_probs(model, tok, prompt):
    chat = tok.chat_template is not None
    if chat:
        msgs = [{"role": "user", "content": prompt}]
        kw = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt = tok.apply_chat_template(msgs, **kw)
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=not chat).to(model.device)
    return torch.softmax(model(**inputs).logits[0, -1].float(), -1).cpu().numpy().astype(np.float64)
