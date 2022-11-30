from plotting import json_to_df, get_slim_filename
import argparse
import pandas as pd
import sys

pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.3f}'.format

parser = argparse.ArgumentParser()
parser.add_argument("--folder", "-fdr", type=str, default=".", help="Folder where `slim-*` files are located.")
parser.add_argument("--models", "-mdl", type=str, nargs="+", default=[], help="Model name in the `slim-*` filename (after `model=`). You may pass a list of names.")
parser.add_argument("--datasets", "-ds", type=str, nargs="+", default=[], help="Dataset name in the `slim-*` filename (after `model=`). You may pass a list of names.")
parser.add_argument("--ehseed", "-es", type=int, default=1234, help="The seed used in the evaluation harness for shuffling and sampling in non-perturbation--related functions.")
parser.add_argument("--ptbseed", "-ps", type=int, default=42, help="The seed used for shuffling and sampling in perturbation-related functions.")
parser.add_argument("--num_shot", "-ns", type=int, default=0, help="Number of shots.")
parser.add_argument("--batch_size", "-bs", type=str, default="auto", help="The batch size argument used.")
parser.add_argument("--perturb_prob", "-pprob", type=str, default="0.25", help="The perturbation probability used in one run.")
parser.add_argument("--perturbers", "-ptbrs", type=str, nargs="+", default=["Original"], help="Manually specify the perturber results that should be loaded. 'Original' results will need to be loaded to calculate any deltas.")
parser.add_argument("--run", "-r", type=str, default="001", help="Run number as it appears in the filename.")

perturbers_group = parser.add_mutually_exclusive_group()
perturbers_group.add_argument("--input_perturbers", "-ip", action="store_true", help="Shortcut for loading files needed to calculate deltas for input perturber evaluations.")
perturbers_group.add_argument("--prompt_perturbers", "-pp", action="store_true", help="Shortcut for loading files needed to calculate deltas for prompt perturber evaluations.")
parser.add_argument("--aggregate_prompts", "-aprm", action="store_true", help="Aggregate results across prompts per dataset and model. Equivalent to Tier 2 results.")
parser.add_argument("--aggregate_prompts_perturbers", "-aptb", action="store_true", help="Aggregate results across prompts and perturbers per dataset and model, then displays the standard deviations accross prompts on the left and across perturbers on the right.")
parser.add_argument("--aggregate_abs_delta_percentage", "-adlt", action="store_true", help="Calculate the absolute \% delta across prompts and perturbers per dataset and model, then displays the standard deviations accross prompts on the left and across perturbers on the right.")
parser.add_argument("--metric", "-m", type=str, default="acc", help="The metric to display. `nlg` can be passed as a shortcut to display BLEU, ROUGE-2, and ROUGE-L, and `sum` to display ROUGE-2 and ROUGE-L.")

args = parser.parse_args()

if not args.aggregate_prompts and not args.aggregate_prompts_perturbers and not args.aggregate_abs_delta_percentage:
    print("Warning: No mode selected!")
    sys.exit()

def zip_sort_df(df, value_col='mean'):
    return sorted(list(zip(
        df['model'].tolist(), 
        df[value_col]['mean'].tolist(), 
        df[value_col]['std'].tolist())), key=lambda x: x[0]
    )

def x_y_agg(df, x, value_col='mean'):
    ywise_agg = df.groupby(['model', x], as_index=False).agg({value_col: ['mean']})
    ywise_agg.columns = pd.Series([i[0] for i in ywise_agg.columns.tolist()])
    x_ywise_agg = ywise_agg.groupby(['model'], as_index=False).agg({value_col: ['mean', 'std']})
    return x_ywise_agg

def print_easy_copy(printables):
    metrics = [f"{p['dataset']} ({p['metric']})" for p in printables]
    print("Metrics:", *metrics)

    for model, _, _ in printables[0]["zipped_dfs"][0]:
        print(model)

    for i, (model, _, _) in enumerate(printables[0]['zipped_dfs'][0]):
        print_str = ""
        for metric in printables:
            zipped_df_1 = metric['zipped_dfs'][0]
            assert model == zipped_df_1[i][0]
            mean = zipped_df_1[i][1]
            std = zipped_df_1[i][2]
            if len(printables[0]['zipped_dfs']) == 1:
                if mean < 1:
                    print_str += f"{mean:.3f} (±{std:.3f})"
                else:
                    print_str += f"{mean:.2f} (±{std:.3f})"
            else:
                zipped_df_2 = metric['zipped_dfs'][1]
                assert model == zipped_df_2[i][0]
                std2 = zipped_df_2[i][2]
                if mean < 1:
                    print_str += f"{mean:.3f} (±{std:.3f} | ±{std2:.3f})"
                else:
                    print_str += f"{mean:.2f} (±{std:.3f} | ±{std2:.3f})"
            print_str += "\t"
        print(print_str)


def generate_printables(df, dataset, metric):
    sub_df = df.loc[df['dataset'] == dataset].loc[df['metric'] == metric]

    if len(sub_df) == 0:
        return

    if args.aggregate_prompts:
        agg_df = sub_df.groupby(['model']).agg({'mean': ['mean', 'std']})
        
        zipped_df = sorted(list(zip(agg_df.index.tolist(), agg_df['mean']['mean'].tolist(), agg_df['mean']['std'].tolist())), key=lambda x: x[0])
        
        return (zipped_df, )

    if args.aggregate_prompts_perturbers:
        perturber_promptwise_agg = x_y_agg(sub_df, 'perturber')

        perturber_promptwise_agg_zipped = zip_sort_df(perturber_promptwise_agg)
        
        prompt_perturberwise_agg = x_y_agg(sub_df, 'prompt_name')

        prompt_perturberwise_agg_zipped = zip_sort_df(prompt_perturberwise_agg)

        return perturber_promptwise_agg_zipped, prompt_perturberwise_agg_zipped

    
    if args.aggregate_abs_delta_percentage:
        sub_df = sub_df[sub_df['perturber'] != 'Original']

        perturber_promptwise_agg = x_y_agg(sub_df, 'perturber', 'absolute_delta_percent')

        perturber_promptwise_agg_zipped = zip_sort_df(perturber_promptwise_agg, 'absolute_delta_percent')

        prompt_perturberwise_agg = x_y_agg(sub_df, 'prompt_name', 'absolute_delta_percent')

        prompt_perturberwise_agg_zipped = zip_sort_df(prompt_perturberwise_agg, 'absolute_delta_percent')
        
        return perturber_promptwise_agg_zipped, prompt_perturberwise_agg_zipped



if args.input_perturbers:
    args.perturbers = ['Original', 'all_input_no_rnd']
elif args.prompt_perturbers:
    args.perturbers = ['Original', 'all_prompt']

file_dict = {}

for dataset in args.datasets:
    for model in args.models:
        for perturber in args.perturbers:
            file = get_slim_filename(args.folder, model, dataset, perturber, args.num_shot, args.perturb_prob, args.ehseed, args.ptbseed, args.run)
            if model in file_dict:
                file_dict[model].append(file)
            else:
                file_dict[model] = [file]
        

print("Retrieved files:", file_dict)

df = json_to_df(file_dict)

if args.metric == 'sum':
    metric_list = ['rouge2_fmeasure', 'rougeL_fmeasure']
elif args.metric == 'nlg':
    metric_list = ['bleu', 'rouge2_fmeasure', 'rougeL_fmeasure']
else:
    metric_list = [args.metric]


df.loc[df.metric == 'acc', 'mean'] = df['mean'] * 100
df.loc[df['metric'].str.startswith('rouge'), 'mean'] = df['mean'] * 100

df_metric_list = [(dataset, metric) for dataset in args.datasets for metric in metric_list]

printables = [
    {
        "metric": metric,
        "dataset": dataset,
        "zipped_dfs": generate_printables(df, dataset, metric)
    } for dataset, metric in df_metric_list
]

printables = [p for p in printables if p['zipped_dfs'] != None]

print_easy_copy(printables)