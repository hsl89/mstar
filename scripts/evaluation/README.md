# mstar evaluation
Source code for evaluating pretrained models

## Sample command:
### for huggingface GPT models
```
python3 mstar/compute_perplexity.py --model_name gpt2-large --dataset wikitext --split test --gpu_id 0 --stride 512
```

### for GPT-neo models
```
python3 mstar/compute_perplexity.py --model_name EleutherAI/gpt-neo-1.3B --dataset ptb --split test --gpu_id 0 --stride 512
```

### for mstar models
```
python3 mstar/compute_perplexity.py --mstar --package_path $packaged_model_path --model_size 672M --dataset $dataset --split test --gpu_id 0 --stride 512
```
