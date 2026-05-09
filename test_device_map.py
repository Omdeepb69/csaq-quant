import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "Qwen/Qwen1.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# simulate 2 GPUs with device_map
device_map = {"model.embed_tokens": 0, "model.layers.0": 0, "model.layers.1": 1, "lm_head": 1}
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

# input on cpu
enc = tokenizer("hello world", return_tensors="pt")
print("Input device:", enc.input_ids.device)

try:
    out = model(**enc)
    print("Success! Logits device:", out.logits.device)
except Exception as e:
    print("Error:", e)
