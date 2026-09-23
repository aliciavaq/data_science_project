"""
Loads Qwen3-4B-Instruct-2507, the model required for all LLM tasks in the milestone.

Requires the `llm` extra:

    uv sync --extra llm

Quick check that everything works (downloads ~8 GB of weights on first run):

    uv run python -m ift6758.llm.qwen

See the model card for usage and recommended generation settings:
https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"


def load_model(model_name: str = MODEL_NAME):
    """
    Loads the tokenizer and model from the Hugging Face Hub (cached after the first download).

    The weights are stored in BF16 and loaded in FP16 on GPU (the Colab T4 does not support BF16).
    On CPU the model is loaded in FP32.

    Returns:
        (model, tokenizer)
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    dtype = torch.float32 if device == "cpu" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_model()
    print(f"Loaded {MODEL_NAME} on {model.device} ({model.dtype})")

    messages = [{"role": "user", "content": "In one sentence, what is a hat trick in hockey?"}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=64)
    print(tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
