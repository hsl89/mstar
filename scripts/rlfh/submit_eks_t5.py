import subprocess
import os
import argparse
import yaml

def dump_yaml(base_config):
    with open('./tmp_config.yaml', 'w') as file:
        documents = yaml.dump(base_config, file)

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--region',type=str,choices=['us-east-1','us-east-2','us-west-2','ap-northeast-2'],default='us-east-2')
    parser.add_argument('--node-type',type=str,choices=['p3dn.24xlarge','p4d.24xlarge'],default='p4d.24xlarge')
    parser.add_argument('--node-num',type=int,default=-1)
    parser.add_argument('--base-file',type=str,default=-1)
    parser.add_argument('--sweep-arg',type=str, default="model.OptimizerArgs.seed")
    args = parser.parse_args()


    BASE_FILE = "./conf/eks_sftp_t5_small.yaml"
    sweep_arg = "model.OptimizerArgs.seed"
    sweep_value = [1,2,3]


    #load base config file, then overwrite
    base_config = yaml.load(open(BASE_FILE,'r'),Loader=yaml.FullLoader)

    commands = base_config['command']


    for value in sweep_value:
        for i in range(len(commands)):
            if sweep_arg in commands[i]:
                commands[i] = "{}={}".format(sweep_arg, value)
        base_config['command'] = commands
        dump_yaml(base_config)
        # print(base_config)
        launch_cmd = 'mstarx --profile gluonnlp submit -f ./tmp_config.yaml'
        print("Running ",launch_cmd)
        os.system(launch_cmd)


if __name__=="__main__":
    run()