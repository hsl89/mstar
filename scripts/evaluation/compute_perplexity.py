from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer, GPTNeoForCausalLM
from datasets import load_dataset
import torch
from tqdm import tqdm
from torch import package
import argparse
import time

# load huggingface/gpt-neo style models
def load_model_hf(model_name, device):
    if "neo" in model_name:
        model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)
        max_length = model.config.max_position_embeddings
    else:
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        max_length = model.config.n_positions
    return model, max_length

# load models from mstar
def load_model_mstar(package_path, model_size, device):

    package_name = "gpu2"
    resource_name = 'model-{}.pkl'.format(model_size)
    imp = package.PackageImporter(package_path)
    model = imp.load_pickle(package_name, resource_name).to(device)
    print('model loaded ...')
    #print('max length: {}'.format(model.config.n_ctx))
    max_length = model.config.n_positions
    return model, max_length

def load_tokenizer(model_id):
    #tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


def load_data_samples(dataset, split):
    if dataset == "wikitext":
        samples = load_dataset(dataset, "wikitext-2-raw-v1", split=split)
        return samples["text"]
    elif dataset == "lambada":
        samples = load_dataset(dataset, split=split)
        return samples["text"]
    elif dataset == "ptb":
        samples = load_dataset("ptb_text_only", split=split)
        return samples["sentence"]
    else:
        raise NotImplementedError("The current script is not implemented for {} dataset.".format(dataset))

def compute_perplexity(model, tokenizer, samples, max_length, stride=512):
    # https://huggingface.co/docs/transformers/perplexity
    # https://sjmielke.com/comparing-perplexities.htm
    encodings = tokenizer("\n\n".join(samples), return_tensors="pt")
    #print('max_length: {}, num_inputs: {}'.format(max_length, encodings.input_ids.size(1)))

    nlls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        #print('i: {}, begin_loc: {}, end_loc: {}, trg_len: {}'.format(i, begin_loc, end_loc, trg_len))
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)

    n_byte = len("\n\n".join(samples).encode("utf-8"))
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    ppl_byte = torch.exp(torch.stack(nlls).sum() / n_byte)
    #print(f'computed perplexity: {ppl:.2f}')
    print('n_token: {}, n_byte: {}'.format(end_loc, n_byte))
    print(f'token perplexity: {ppl:.2f}, byte perplexity: {ppl_byte:.2f}')
    return ppl

def add_args(parser):
    """
    Adds logs arguments to the passed parser
    """
    parser.add_argument("--mstar", action='store_true', required=False,
                        help="use when processing single file, not a list of files")
    parser.add_argument("--package_path", default="/home/ubuntu/data", type=str, required=False,
                        help="Path to the packaged model evaluation.")
    parser.add_argument("--model_size", default="672M", type=str, required=False,
                        help="size of the model being evaluated.")
    parser.add_argument("--model_name", default="gpt2", type=str, required=False,
                        help="the huggingface/gpt-neo style model name for evaluation.")
    parser.add_argument("--dataset", default="wikitext", type=str, required=False,
                        help="The dataset on which to evaluate the model")
    parser.add_argument("--split", default="test", type=str, required=False,
                        help="The split of the dataset to use for evaluation")
    parser.add_argument("--gpu_id", default=0, type=int, required=False,
                        help="id of the gpu to run evaluation")
    parser.add_argument("--stride", default=512, type=int, required=False,
                        help="stride for sliding window evaluation")
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    start = time.time()
    # get the device name
    device = 'cuda:{}'.format(args.gpu_id)
    # load model
    if args.mstar:
        model, max_length = load_model_mstar(args.package_path, args.model_size, device)
    else:
        model, max_length = load_model_hf(args.model_name, device)

    # load tokenizer, hard-coding 'gpt2' tokenizer
    tokenizer = load_tokenizer('gpt2')
    # load dataset
    samples = load_data_samples(args.dataset, args.split)
    # compute perplexity
    ppl = compute_perplexity(model, tokenizer, samples, max_length, stride=args.stride)

    end = time.time()
    print(f'time to process: {(end - start):.2f} seconds')