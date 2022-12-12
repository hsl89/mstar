import subprocess
import os
import argparse
import yaml

def dump_yaml(base_config):
    with open('./tmp_config.yaml', 'w') as file:
        documents = yaml.dump(base_config, file)

def run(args):

    BASE_FILE = "./exps/eks_sftpv2_t5small.yaml"
    sweep_arg = args.sweep_arg
    sweep_value = [1,2,3]

    #load base config file, then overwrite
    base_config = yaml.safe_load(open(BASE_FILE,'r'))

    commands = base_config['command']


    for value in sweep_value:
        for i in range(len(commands)):
            if sweep_arg in commands[i]:
                commands[i] = "{}={}".format(sweep_arg, value)
        base_config['command'] = commands
        dump_yaml(base_config)
        launch_cmd = 'mstarx --profile gluonnlp submit -f ./tmp_config.yaml'
        print("Running ",launch_cmd)
        os.system(launch_cmd)

def cancel_job(args):
    job_lists = args.joblist.split(" ")
    for jobs in job_lists:        
        launch_cmd = f'mstarx --profile gluonnlp cancel -j {jobs}'
        print("Running ",launch_cmd)
        os.system(launch_cmd)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_type',type=str,choices=['run','cancel'],default='run')
    parser.add_argument('--region',type=str,choices=['us-east-1','us-east-2','us-west-2','ap-northeast-2'],default='us-east-2')
    parser.add_argument('--node-type',type=str,choices=['p3dn.24xlarge','p4d.24xlarge'],default='p4d.24xlarge')
    parser.add_argument('--node-num',type=int,default=-1)
    parser.add_argument('--base-file',type=str,default=-1)
    parser.add_argument('--sweep-arg',type=str, default="model.OptimizerArgs.seed")
    parser.add_argument('--joblist',type=str, default="rlfh-60m-t5small-0f0f6e7b27884716badb572 rlfh-60m-t5small-35e3fdfdaeb143c7ac70ef2")
    args = parser.parse_args()
    if args.run_type == "run":
        run(args)
    elif args.run_type == "cancel":
        cancel_job(args)
    else:
        raise("job types not supported")
