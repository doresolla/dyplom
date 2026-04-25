from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None


BASE_DIR = Path(__file__).resolve().parent

# Локальные директории моделей.
# Можно переопределить через переменные окружения QWEN_MODEL_DIR и MISTRAL_MODEL_DIR.
MODELS: Dict[str, Path] = {
    "qwen": Path(os.getenv("QWEN_MODEL_DIR", str(BASE_DIR / "models" / "Qwen2.5-7B-Instruct"))),
    "mistral": Path(os.getenv("MISTRAL_MODEL_DIR", str(BASE_DIR / "models" / "Mistral-7B-Instruct-v0.3"))),
}
DEFAULT_MODEL = "qwen"

USE_4BIT = os.getenv("LLM_USE_4BIT", "1") == "1"
MAP_MAX_NEW_TOKENS = 320
REDUCE_MAX_NEW_TOKENS = 700
CHUNK_TOKENS = 2200
OVERLAP_TOKENS = 200
REDUCE_GROUP_SIZE = 4

SYSTEM_PROMPT = (
    "Ты помощник для создания учебных конспектов. "
    "Пиши по-русски. Не добавляй факты, которых нет в исходном тексте."
)

_MODEL_CACHE = {
    "key": None,
    "tokenizer": None,
    "model": None,
}


def _resolve_model_key(model: str | None) -> str:
    if not model:
        return DEFAULT_MODEL

    key = model.strip().lower()
    if key in MODELS:
        return key

    lowered = model.lower()
    if "qwen" in lowered:
        return "qwen"
    if "mistral" in lowered:
        return "mistral"

    return DEFAULT_MODEL


def _resolve_model_path(model: str | None) -> Path:
    return MODELS[_resolve_model_key(model)]


def unload_model() -> None:
    model_obj = _MODEL_CACHE.get("model")
    tokenizer = _MODEL_CACHE.get("tokenizer")

    if model_obj is not None:
        del model_obj
    if tokenizer is not None:
        del tokenizer

    _MODEL_CACHE["key"] = None
    _MODEL_CACHE["tokenizer"] = None
    _MODEL_CACHE["model"] = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _load_model(model: str | None) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    key = _resolve_model_key(model)
    model_path = _resolve_model_path(model)

    if _MODEL_CACHE["key"] == key and _MODEL_CACHE["tokenizer"] is not None and _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["tokenizer"], _MODEL_CACHE["model"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Локальная директория модели не найдена: {model_path}\n"
            f"Проверьте MODELS или переменные окружения для модели '{key}'."
        )

    unload_model()

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    load_kwargs = {
        "local_files_only": True,
        "trust_remote_code": True,
    }

    if torch.cuda.is_available():
        if USE_4BIT and BitsAndBytesConfig is not None:
            load_kwargs["device_map"] = "auto"
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            load_kwargs["device_map"] = "auto"
            load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model_obj = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)

    if not torch.cuda.is_available():
        model_obj.to("cpu")

    model_obj.eval()

    _MODEL_CACHE["key"] = key
    _MODEL_CACHE["tokenizer"] = tokenizer
    _MODEL_CACHE["model"] = model_obj
    return tokenizer, model_obj


def _get_input_device(model_obj: AutoModelForCausalLM) -> torch.device:
    if hasattr(model_obj, "device"):
        return model_obj.device
    return next(model_obj.parameters()).device


def _build_chat_prompt(tokenizer: AutoTokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    return f"[SYSTEM]\n{system}\n\n[USER]\n{user}\n\n[ASSISTANT]\n"


@torch.inference_mode()
def _generate(
    tokenizer: AutoTokenizer,
    model_obj: AutoModelForCausalLM,
    prompt_text: str,
    max_new_tokens: int,
) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=False)
    input_device = _get_input_device(model_obj)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    output = model_obj.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_len = inputs["input_ids"].shape[-1]
    generated_ids = output[0][prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _split_by_tokens(
    tokenizer: AutoTokenizer,
    text: str,
    chunk_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> List[str]:
    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    if not token_ids:
        return []

    chunk_tokens = max(256, int(chunk_tokens))
    overlap_tokens = max(0, int(overlap_tokens))
    if overlap_tokens >= chunk_tokens:
        overlap_tokens = chunk_tokens // 4

    chunks: List[str] = []
    start = 0
    total = len(token_ids)

    while start < total:
        end = min(start + chunk_tokens, total)
        chunk = tokenizer.decode(token_ids[start:end], skip_special_tokens=True)
        chunks.append(chunk)
        if end >= total:
            break
        start = end - overlap_tokens

    return chunks


def _make_map_prompt(chunk_text: str) -> str:
    return f"""Сделай краткий, но содержательный конспект фрагмента лекции.

Требования:
- 6-10 пунктов;
- сохрани ключевые термины, числа, формулы и обозначения;
- убери повторы и мусор распознавания;
- не добавляй информацию от себя.

Формат ответа: Markdown-список.

ФРАГМЕНТ ТРАНСКРИПТА:
{chunk_text}
"""


def _make_reduce_prompt(partials_text: str) -> str:
    return f"""На основе частичных конспектов сделай единый итоговый конспект лекции.

Требования:
- в начале укажи тему или основную идею;
- далее дай 10-16 содержательных пунктов;
- отдельным блоком выдели важные определения, формулы или обозначения, если они есть;
- не добавляй факты, которых нет в исходном тексте.

Формат ответа: Markdown.

ЧАСТИЧНЫЕ КОНСПЕКТЫ:
{partials_text}
"""


def _reduce_recursive(
    tokenizer: AutoTokenizer,
    model_obj: AutoModelForCausalLM,
    partials: List[str],
) -> str:
    current = [p.strip() for p in partials if p and p.strip()]
    if not current:
        return ""

    while len(current) > 1:
        next_level: List[str] = []
        for i in range(0, len(current), REDUCE_GROUP_SIZE):
            group = current[i:i + REDUCE_GROUP_SIZE]
            group_text = "\n\n".join(
                f"Фрагмент {idx + 1}:\n{item}" for idx, item in enumerate(group)
            )
            prompt = _build_chat_prompt(tokenizer, SYSTEM_PROMPT, _make_reduce_prompt(group_text))
            reduced = _generate(
                tokenizer=tokenizer,
                model_obj=model_obj,
                prompt_text=prompt,
                max_new_tokens=REDUCE_MAX_NEW_TOKENS,
            )
            next_level.append(reduced)
        current = next_level

    return current[0].strip()


def summarize_with_llm(text: str, model: str | None = None) -> str:
    if not text or not text.strip():
        return ""

    tokenizer, model_obj = _load_model(model)
    clean_text = preprocess_text(text)
    chunks = _split_by_tokens(tokenizer, clean_text)

    if not chunks:
        return ""

    partials: List[str] = []
    for chunk in chunks:
        prompt = _build_chat_prompt(tokenizer, SYSTEM_PROMPT, _make_map_prompt(chunk))
        summary_part = _generate(
            tokenizer=tokenizer,
            model_obj=model_obj,
            prompt_text=prompt,
            max_new_tokens=MAP_MAX_NEW_TOKENS,
        )
        partials.append(summary_part)

    if len(partials) == 1:
        return partials[0].strip()

    return _reduce_recursive(tokenizer, model_obj, partials)


def preprocess_text(text: str) -> str:
    if text is None:
        return ""

    text = text.replace("\xa0", " ")
    text = "\n".join(line.strip() for line in text.splitlines())
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def fallback_summary(text: str, max_lines: int = 12) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    top = lines[:max_lines]
    bullets = "\n".join(f"- {line}" for line in top)
    return "## Черновик конспекта\n\n" + bullets


def save_summary(summary: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(summary, encoding="utf-8")
    return out_path