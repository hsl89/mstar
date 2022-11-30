import os
import argparse
import yaml

def node_type_conversion(x):
    if x == 'p4':
        return 'p4d.24xlarge'
    if x == 'p4de':
        return 'p4de.24xlarge'
    elif x == 'p3':
        return 'p3dn.24xlarge'
    elif x == 'g4':
        return 'g4dn.12xlarge'
    elif x == 'g5':
        return 'g5.48xlarge'
    else:
        raise NotImplementedError()



#take region
parser = argparse.ArgumentParser()
parser.add_argument('--region', '-rgn', type=str, choices=['us-east-1','us-east-2','us-west-2','ap-northeast-2'], default='us-east-1', help="Region to launch the jobs in.")
parser.add_argument('--node-type', '-n', type=str,choices=['p3','p4','g4','g5','p4de'], default='p4', help="Node types to use.")
parser.add_argument('--node-num', type=int, default=1)
parser.add_argument('--setting', '-s', type=str, nargs="+", choices=['a','o', 'i', 'p'], default=['a'], help="Perturbation settings to launch evaluations for: [o]riginal (no perturbations), [i]nput perturbers, [p]rompt perturbers, or [a]ll.")
parser.add_argument('--perturb_prob', '-p', type=float, nargs="+", default=[0.25], help="Perturbation probabilities to launch evaluations for.")
parser.add_argument('--base_yaml', '-f', type=str, default='configs/eval_robustness_eks.yaml', help="Path to config file")
args = parser.parse_args()

BASE_FILE = args.base_yaml

#for simpler cmd line args
args.node_type = node_type_conversion(args.node_type)

if args.region == 'us-east-1':
    attach_cmd='mstarx --profile gluonnlp config --cluster mstar-eks-v2 --region {}'.format(args.region)
else:
    attach_cmd='mstarx --profile gluonnlp config --cluster mstar-eks --region {}'.format(args.region)
print("Running ", attach_cmd)
os.system(attach_cmd)


def launch_job(revision, dataset, num_fewshot, perturbers, perturb_prob, perturb_seed):
    #load base config file, then overwrite
    print("Loading from", BASE_FILE)

    base_config = yaml.load(open(BASE_FILE,'r'), Loader=yaml.FullLoader)
    base_config['node_type'] = args.node_type

    #only overwrite node number if nonzero number provided
    if args.node_num:
        base_config['node_num'] = args.node_num

    #image region
    beg, end = base_config['image'].split('REGION_TAG')
    base_config['image'] = beg + args.region + end

    trunc_model_name = base_config['model'].split('/')[-1] if '/' in base_config['model'] else base_config['model'][-10:]

    model_args = f"pretrained={base_config['model']},use_accelerate=True"
    
    if revision != None:
        model_args += f",revision={revision}"
        trunc_model_name = revision

    if base_config.get('add_special_tokens', None) != None:
        model_args += f",add_special_tokens={base_config['add_special_tokens']}"

    if base_config.get('special_prefix_token_id', None) != None and base_config.get('special_prefix_token', None) != None:
        model_args += f",special_prefix_token_id={base_config['special_prefix_token_id']}"
        model_args += f",special_prefix_token={base_config['special_prefix_token']}"

    if base_config.get('mode_prefix_token', None) != None:
        model_args += f",mode_prefix_token={base_config['mode_prefix_token']}"

    if base_config.get('add_extra_id_0', None):
        model_args += f",add_extra_id_0={base_config['add_extra_id_0']}"

    num_templates = '-1' if perturbers == 'all_prompt' or perturbers == 'Original' else '5'

    base_config['name'] += '-'.join([dataset, str(num_fewshot), str(perturb_prob), str(perturb_seed), str(base_config['run_no']), trunc_model_name, ])

    base_config['args'] += [
        "--model_api_name", base_config['model_api'],
        "--model_args", model_args,
        "--num_fewshot", str(num_fewshot),
        "--batch_size", "auto",
        "--run_no", str(base_config['run_no']),
        '--perturbers', perturbers, 
        '--task_name', dataset, 
        '--template_names', 'original_templates',
        '--num_sampled_templates', num_templates,
        '--perturb_seed', str(perturb_seed),
        '--perturb_prob', str(perturb_prob),
    ]

    if 'subsample_factor' in base_config:
        if isinstance(base_config['subsample_factor'], int):
            base_config['args'] += ['--example_subsample_factor', str(base_config['subsample_factor'])]
        
        elif dataset in base_config['subsample_factor']:
            base_config['args'] += ['--example_subsample_factor', str(base_config['subsample_factor'][dataset])]
        
    print("Launch args:", base_config['args'])

    os.makedirs('configs', exist_ok=True)
    with open('configs/tmp_config.yaml', 'w') as file:
        yaml.dump(base_config, file)

    launch_cmd = 'mstarx --profile gluonnlp submit -f configs/tmp_config.yaml'

    if args.region == "us-east-1":
        launch_cmd = "DAG_NAME=pytorch_evaluation_job_dag " + launch_cmd

    if args.region == "us-east-2" or args.region == "us-east-1":
        launch_cmd = "ENVIRONMENT_NAME=MStarAirflowEnvironment " + launch_cmd

    os.system(launch_cmd)

base_config = yaml.load(open(BASE_FILE,'r'), Loader=yaml.FullLoader)

for dataset in base_config['datasets']:
    for revision in base_config.get('revisions', [None]):
        for num_fewshot in base_config['num_fewshot']:
            if 'a' in args.setting or 'o' in args.setting:
                launch_job(revision, dataset, num_fewshot, 'Original', 0, 42)
            
            if 'a' in args.setting or 'i' in args.setting:
                for perturb_prob in args.perturb_prob:
                    launch_job(revision, dataset, num_fewshot, 'all_input_no_rnd', perturb_prob, 42)

            if 'a' in args.setting or 'p' in args.setting:
                launch_job(revision, dataset, num_fewshot, 'all_prompt', 1, 42)