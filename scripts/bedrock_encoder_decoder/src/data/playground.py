from p3_datamodule import T50Data
import torch
import pytorch_lightning as pl
import transformers as hf

P3_BATCH = 8
SAMPLE_SIZE = 512
torch.distributed.init_process_group(backend="gloo")
pl.seed_everything(1)
torch.manual_seed(1)
print(
    "World size ",
    torch.distributed.get_world_size(),
    " Rank ",
    torch.distributed.get_rank(),
)

P3_BATCH = 8
PILE_BATCH = 0
TOKENIZER = "t5-base"

tokenizer = hf.AutoTokenizer.from_pretrained(TOKENIZER)
data_module = T50Data(tokenizer, P3_BATCH, PILE_BATCH)
data_module.setup()
p3_loader = data_module.train_dataloader()
for i, out in enumerate(p3_loader):
    if i == 0:
        break

for key, val in out.items():
    print(key, val.shape)
    assert list(val.shape) == [P3_BATCH, SAMPLE_SIZE]
if torch.distributed.get_rank() == 0:
    for idx in range(P3_BATCH):
        to_decode = out["input_ids"][idx].long()
        print("Encoder input ids\n", to_decode)
        # to_decode = torch.where(to_decode==-100,tokenizer.pad_token_id,to_decode)
        print("Encoder text \n", tokenizer.decode(to_decode))
        # list(out['input_ids'][idx])))
        to_decode = out["labels"][idx].long()
        print("Decoder target labels", to_decode)
        # print("Decoder labels before padding conversion\n",to_decode)
        to_decode = torch.where(to_decode == -100, tokenizer.pad_token_id, to_decode)
        print("Decoder target text \n", tokenizer.decode(list(to_decode)))
        # deal with padding tokens conversion to -100
        to_decode = out["decoder_input_ids"][idx].long()
        # to_decode = torch.where(to_decode==-100,tokenizer.pad_token_id,to_decode)
        print("Decoder input text \n", tokenizer.decode(list(to_decode)))

        try:
            print(out["attention_mask"][idx])
        except:
            print("No encoder attention mask")
        try:
            print(out["decoder_attention_mask"][idx])
        except:
            print("No decoder attention mask")
