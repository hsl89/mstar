import argparse
from mstar import AutoModel, AutoConfig
from mstar.models import model_factory
import gc
import copy
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model_files", "-f", nargs="+", type=str, default=[], help="local model paths. Use `--model` and `--revisions` if model is in `s3://mstar-models`.")
parser.add_argument("--model", "-m", type=str, default=None, help="model ID in `s3://mstar-models`")
parser.add_argument("--revisions", "-r", nargs="+", type=str, default=[], help="revision IDs in `s3://mstar-models`. Multiple IDs are allowed.")
parser.add_argument("--output", "-o", type=str, help="output directory for averaged model.")
args = parser.parse_args()

assert args.model_files or args.revisions, "No models specified"

model_args = [{"pretrained_model_name_or_path": path} for path in args.model_files] + [{"pretrained_model_name_or_path": args.model, "revision": rev} for rev in args.revisions]

num_models = float(len(model_args))
first_model = AutoModel.from_pretrained(**model_args[0]).state_dict()

avg_state_dict = copy.deepcopy(first_model)

old_avg_state_dict = {}

print_first = True

print("\nCommencing averaging...")

# we first sum the weights so we can load the models lazily
for model_arg in model_args[1:]:
    model = AutoModel.from_pretrained(**model_arg).state_dict()
    
    # check that keys are identical across models
    assert list(first_model.keys()) == list(model.keys()), "Layers are not identical. Model must have exactly the same layers to be averaged."

    for key in first_model:
        avg_state_dict[key] = sum([avg_state_dict[key], model[key]])
        if print_first:
            print("Key:", key, "Weights:", model[key])
            print_first = False

    del(model)
    gc.collect()
    print_first = True


for key in avg_state_dict:
    avg_state_dict[key] /= num_models
    if print_first:
        print("Averaged:", avg_state_dict[key])
        print_first = False

print("Done.")


if args.model and args.revisions:
    config = AutoConfig.from_pretrained(args.model, revision=args.revisions[-1])
elif args.model_files:
    config = AutoConfig.from_pretrained(args.model_files[-1])


model = getattr(model_factory, config.architectures[0])(config=config).cpu()
model.load_state_dict(avg_state_dict)

print()
print("Saving to", args.output)
# one shard since M* autodownload will fail otherwise
model.save_pretrained(args.output, max_shard_size="999GB")
print("Done.")
