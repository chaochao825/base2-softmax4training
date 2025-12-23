import os
from transformers import GPT2LMHeadModel, AutoTokenizer

os.environ['HF_HOME'] = "/home/spco/base-2-bitnet/.hf_cache"
model_name = "gpt2-xl"

print(f"Attempting to download {model_name}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    print(f"Successfully downloaded {model_name} to {os.environ['HF_HOME']}")
except Exception as e:
    print(f"Failed to download {model_name}: {e}")


