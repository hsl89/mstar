import subprocess
import os
import argparse
import yaml
import warnings


def node_type_conversion(x):
    if x == "p4":
        return "p4d.24xlarge"
    elif x == "p3":
        return "p3dn.24xlarge"
    elif x == "g4":
        return "g4dn.12xlarge"
    else:
        raise NotImplementedError()


# take region
parser = argparse.ArgumentParser()
parser.add_argument(
    "--region",
    type=str,
    choices=["us-east-1", "us-east-2", "us-west-2", "ap-northeast-2"],
    default="us-east-1",
)
parser.add_argument("--node-type", type=str, choices=["p3", "p4", "g4"], default="p4")
parser.add_argument("--node-num", type=int, default=0)
parser.add_argument(
    "--gpu", type=int, default=0, help="How many gpus, only override if >0"
)
parser.add_argument(
    "--name", type=str, required=False, help="override base config name"
)
parser.add_argument(
    "--type",
    type=str,
    choices=["workspace", "tier2"],
    required=True,
    help="Workspace or tier2 launch",
)
parser.add_argument(
    "--model-path",
    type=str,
    required=False,
    help="Model path to automodel for Tier 2 evaluation",
)
args = parser.parse_args()


if args.type == "workspace":
    if args.node_type == "p4":
        BASE_FILE = "configs/p4_dev_workspace.yaml"
    elif args.node_type == "g4":
        BASE_FILE = "configs/g4_dev_workspace.yaml"
    else:
        raise NotImplementedError()
elif args.type == "tier2":
    BASE_FILE = "configs/tier2/base.yaml"
    assert args.node_num <= 1, f"Tier 2 is single GPU, requested {args.node_num}"
else:
    raise NotImplementedError()

# for simpler cmd line args
args.node_type = node_type_conversion(args.node_type)

if args.node_type == "g4":
    assert args.gpu <= 4
else:
    assert args.gpu <= 8
"""
#need mstarx cluster environment to be set up
#attach to right region
attach_cmd='mstarx --profile gluonnlp config --cluster mstar-eks --region {}'.format(args.region)
print("Running ",attach_cmd)
os.system(attach_cmd)
"""
# load base config file, then overwrite
base_config = yaml.load(open(BASE_FILE, "r"), Loader=yaml.FullLoader)
base_config["node_type"] = args.node_type

# only override if >0
if args.gpu:
    base_config["gpu"] = args.gpu

# only overwrite node number if nonzero number probided
if args.node_num:
    base_config["node_num"] = args.node_num

# image region
if args.type == "tier2":
    # assumption is that only us-east-2 region pullis supported
    pass
else:
    beg, end = base_config["image"].split("REGION_TAG")
    base_config["image"] = beg + args.region + end


if args.type == "tier2":
    # override MODEL_TAG with path to an automodel
    beg, end = base_config["name"].split("MODEL_TAG")
    base_config["name"] = beg + args.model_path + end
    new_args = []
    for entry in base_config["args"]:
        if "MODEL_TAG" in entry:
            """
            beg,end = entry.split('MODEL_TAG')
            if new_args[-1]=='--model_args':
                tokenizer_args='tokenizer'
                end = '/hf_version/'+','+tokenizer_args
            else:
                end=''
            new_entry=beg+args.model_path+end
            """
            new_entry = args.model_path.join(entry.split("MODEL_TAG"))
            new_args.append(new_entry)
        else:
            new_args.append(entry)

    base_config["args"] = new_args

if args.type == "workspace":
    # image name based on user name
    my_name = subprocess.check_output("whoami", text=True).strip("\n")
    beg, end = base_config["image"].split("USER_TAG")
    base_config["image"] = beg + my_name + end
    # image name based on additional tag
    beg, end = base_config["image"].split("NAME_TAG")
    new_name_tag = args.name_tag + "_dev" if args.type == "workspace" else args.name_tag
    base_config["image"] = beg + new_name_tag + end

with open("configs/tmp_config.yaml", "w") as file:
    documents = yaml.dump(base_config, file)


# raise ValueError
# need mstarx cluster environment to be set up
# attach to right region
attach_cmd = "mstarx --profile gluonnlp config --cluster mstar-eks --region {}".format(
    args.region
)
print("Running ", attach_cmd)
os.system(attach_cmd)

if args.type == "workspace":
    launch_cmd = (
        "mstarx --profile gluonnlp  workspace create -f configs/tmp_config.yaml"
    )
elif args.type in ["tier2"]:
    launch_cmd = "mstarx submit -f configs/tmp_config.yaml"
else:
    raise NotImplementedError()

print("Running ", launch_cmd)
os.system(launch_cmd)
