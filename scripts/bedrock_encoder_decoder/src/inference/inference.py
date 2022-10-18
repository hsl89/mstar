from mstar import AutoModel, AutoTokenizer

# from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

model_name = "mstar-t5-1-9B-bedrock"
revision = "stage_2_alexatm"

model = AutoModel.from_pretrained(model_name, revision=revision).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
input_string = "<extra_id_0> All mainstream forms of Judaism today"
tok = tokenizer(input_string, return_tensors="pt")
print("TOKENIZED")
print(tok)
ids = model.generate(
    input_ids=tok["input_ids"].cuda(),
    attention_mask=tok["attention_mask"].cuda(),
    max_length=50,
)
print("GENERATED")
output = tokenizer.batch_decode(ids)
print("Prompt:", input_string)
print("Continuation:", output)
