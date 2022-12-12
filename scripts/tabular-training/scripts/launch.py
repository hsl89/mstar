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
    "--rebuild-docker",
    type=int,
    default=False,
    help="Rerun/Skip docker build by default. Necessary to rebuild docker if type==job since we cant git pull inside container.",
)
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
    choices=["workspace", "job", "resume_job"],
    required=True,
    help="Workspace or job launch?",
)
parser.add_argument(
    "--name-tag",
    type=str,
    default="easel",
    required=False,
    help="Tag for dockerfile, build in scripts/docker.sh",
)
args = parser.parse_args()

if args.type == "workspace":
    if args.node_type == "p4":
        BASE_FILE = "configs/p4_dev_workspace.yaml"
    elif args.node_type == "g4":
        BASE_FILE = "configs/g4_dev_workspace.yaml"
    else:
        raise NotImplementedError()
elif args.type == "job":
    BASE_FILE = "configs/pretrain_launch.yaml"
elif args.type == "resume_job":
    BASE_FILE = "configs/pretrain_resume.yaml"
    # args.name_tag=args.name_tag+'_dev'
else:
    raise NotImplementedError()

# for simpler cmd line args
args.node_type = node_type_conversion(args.node_type)

if args.node_type == "g4":
    assert args.gpu <= 4
else:
    assert args.gpu <= 8

# need mstarx cluster environment to be set up
# attach to right region
attach_cmd = "mstarx --profile gluonnlp config --cluster mstar-eks --region {}".format(
    args.region
)
print("Running ", attach_cmd)
os.system(attach_cmd)

if args.rebuild_docker:
    # build research dockerfile and push to region with arg
    docker_cmd = "bash scripts/docker.sh {} {} {}".format(
        args.region, args.name_tag, int(args.type == "workspace")
    )
    print("Running ", docker_cmd)
    os.system(docker_cmd)
else:
    if args.type == "job":
        warnings.warn("Skipping docker build may lead to bad job runs")


# load base config file, then overwrite
base_config = yaml.safe_load(open(BASE_FILE, "r"))
base_config["node_type"] = args.node_type

# only override if >0
if args.gpu:
    base_config["gpu"] = args.gpu

# only overwrite node number if nonzero number probided
if args.node_num:
    base_config["node_num"] = args.node_num

# image region
beg, end = base_config["image"].split("REGION_TAG")
base_config["image"] = beg + args.region + end

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

if args.type == "workspace":
    launch_cmd = (
        "mstarx --profile gluonnlp  workspace create -f configs/tmp_config.yaml"
    )
elif args.type in ["job", "resume_job"]:
    launch_cmd = "mstarx submit -f configs/tmp_config.yaml"
else:
    raise NotImplementedError()

print("Running ", launch_cmd)
os.system(launch_cmd)
