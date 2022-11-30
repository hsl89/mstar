import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from glob import glob

def get_original_results_dict(json_file):
    results = {}
    original_result_objs = [result for result in json_file["results"] if result["perturber"] == "Original"][0]["results"]
    for result_obj in original_result_objs:
        metric = [k[:-7] for k in result_obj.keys() if "_stderr" in k][0]
        if result_obj["prompt_name"] not in results:
            results[result_obj["prompt_name"]] = { metric: result_obj[metric] }
        else:
            results[result_obj["prompt_name"]][metric] = result_obj[metric]
    return results


def json_to_df(json_files):
    model_rows = []
    
    for model, file_paths in json_files.items():
        for path in file_paths:
            model_rows.append(load_json_file(path, model))
    
    df = pd.concat(model_rows, ignore_index=True).drop_duplicates(ignore_index=True)
    for c in ['model', 'prompt_name', 'perturber', 'metric']:
        df[c] = df[c].astype('string')
    df['delta'], df['absolute_delta'], df['absolute_delta_percent'] = calculate_delta(df)
    return df
    
def calculate_delta(df):
    deltas = []
    abs_deltas = []
    abs_deltas_percent = []
    for _, row in df.iterrows():
        # Isolate the column with the
        orig_mean = df[(df['dataset'] == row['dataset']) & (df['model'] == row['model']) & (df['prompt_name'] == row['prompt_name']) & (df['metric'] == row['metric']) & (df['perturber'] == 'Original')].drop_duplicates(subset=['mean'], ignore_index=True)['mean'].squeeze()
        delta = row['mean'] - orig_mean
        deltas.append(delta)
        abs_deltas.append(abs(delta))
        abs_deltas_percent.append(abs(delta)/orig_mean * 100)
    return pd.to_numeric(deltas), pd.to_numeric(abs_deltas), pd.to_numeric(abs_deltas_percent)

def load_json_file(file_path, model):
    json_file = json.load(open(file_path))
    
    if "perturber" in json_file["results"][0]:
        perturber_rows = []
        for perturber_results in json_file["results"]:
            perturber_rows.append(add_to_df(perturber_results, 
                                            model, 
                                            perturber_results["perturber"],
                                            perturber_results["perturb_prob"],
                                            perturber_results["perturb_seed"]))

        df = pd.concat(perturber_rows, ignore_index=True)
        return df
    else:
        return add_to_df(json_file, model)


def add_to_df(per_perturber_results, model, perturber_name="Original", perturb_prob=0, perturb_seed=-1):
    rows = []
    for r in per_perturber_results["results"]:

        metric = [k[:-7] for k in r.keys() if "_stderr" in k]
        if not metric:
            metric = [k for k in r.keys() if k not in {"task_name", "prompt_name", "dataset_path","dataset_name", "subset"}]
        metric = metric[0]
        row = {"model": model,
                "dataset": r['task_name'],
                "prompt_name": r["prompt_name"],
                "mean": r[metric],
                "stderr": r[metric + "_stderr"] if (metric + "_stderr") in r else None,
                "perturb_seed": perturb_seed if perturber_name != "Original" else -1,
                "perturber": perturber_name.replace("Perturber", ""),
                "perturb_prob": perturb_prob if perturber_name != "Original" else 0,
                "metric": metric}
        rows.append(pd.DataFrame([row]))
    return pd.concat(rows, ignore_index=True)


def plot_prompt_variation(title, json_files, random_baseline=None, plot_metric='acc', ylimit=None, legend_pos='upper left'):
    # load dataset
    df = json_to_df(json_files)
    #metrics = list(results.keys())
    plot_df = df[(df["perturber"]=="Original") & (df["metric"]==plot_metric)]
    #plot_metric = plot_metric if len(metrics) > 1 else metrics[0]

    plt.figure(figsize=(20, 12))
    plt.rc('font', size=20)
    plt.rc('legend', fontsize=15)
    palette = "tab20" if len(df.model.unique()) > 10 else "tab10"
    ax = sns.barplot(x="prompt_name", y="mean", hue="model", 
                        data=plot_df, palette=sns.color_palette(palette))
    
    if random_baseline is not None:
        ax.axhline(random_baseline, linestyle='--', color='red', label='random')
    plt.title(title, fontsize = 40)
    plt.legend(loc=legend_pos)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize='small')
    ax.set_ylim([None, ylimit])
    ax.set(ylabel=plot_metric)
    ax.set(xlabel="Prompt Name")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]

    ax.errorbar(x_coords, y_coords, yerr=plot_df["stderr"],
                        fmt=' ', barsabove=True, color='black')
    plt.tight_layout()
    plt.savefig(f"../mstar-robustness-figs/{title.lower()}.png",format="png")
    plt.show()

def get_name(path):
    return path.split("/")[-1][5:-5].replace('..-', '')

def plot_absolute_scores(dataset_name, model_name, json_files, num_shot=0, random_baseline=None, plot_metric='acc', ylimit=None, legend_pos='upper left'):
    # load dataset
    df = json_to_df({model_name: json_files})
    df = df[df['metric'] == plot_metric]
    plt.figure(figsize=(20, 12))
    plt.rc('font', size=20)
    plt.rc('legend', fontsize=15)
    palette = "tab20" if len(df.perturber.unique()) > 10 else "tab10"
    ax = sns.barplot(x="prompt_name", y="mean", hue="perturber", #style="perturber",
                        data=df, palette=sns.color_palette(palette))

    if random_baseline is not None:
        ax.axhline(random_baseline, linestyle='--', color='red', label='random')

    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    ax.errorbar(x_coords, y_coords, yerr=df["stderr"],
                        fmt=' ', barsabove=True, color='black')

    plt.title(f"{num_shot}-shot {dataset_name} ({model_name})", fontsize = 40)
    plt.legend(loc=legend_pos)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize='small')
    ax.set_ylim([0, ylimit])
    ax.set(xlabel="Prompt Name")
    ax.set(ylabel=plot_metric)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"../mstar-robustness-figs/raw-scores-{num_shot}-shot-{dataset_name}-{model_name}.png",format="png")

def plot_deltas(dataset_name, json_files, plot_metric='acc', xlimit=None, legend_pos='upper left', figure_height=12):
    # load dataset
    df = json_to_df(json_files)
    plt.figure(figsize=(20, figure_height))
    plt.rc('font', size=20)
    plt.rc('legend', fontsize=15)
    palette = "tab20" if len(df.prompt_name.unique()) > 10 else "tab10"
    ax = sns.barplot(x="delta", y="prompt_name", hue="perturber", #style="perturber",
                        data=df, palette=sns.color_palette(palette))
    
    plt.title(f"{dataset_name} ({list(json_files.keys())[0]})", fontsize = 40)
    plt.legend(loc=legend_pos)
    ax.set_xlim([xlimit, None])
    ax.set(xlabel=plot_metric)
    ax.set(ylabel="Perturber")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"../mstar-robustness-figs/deltas-{get_name(list(json_files.values())[0])}.png",format="png")

def plot_absolute_deltas_over_perturb_prob(dataset_name, json_files, num_shot=0, plot_metric='acc', ylimit=None, legend_pos='upper left', hue_col="prompt_name", figure_height=12):
    # load dataset
    model_name = list(json_files.keys())[0]
    df = json_to_df(json_files)
    plt.figure(figsize=(20, figure_height))
    plt.rc('font', size=20)
    plt.rc('legend', fontsize=15)
    n_colors = len(df[df["perturber"] != "ReplaceWithRandomCharacter"][hue_col].unique())
    print(n_colors)
    palette = "tab20" if n_colors > 10 else "tab10"
    ax = sns.lineplot(
        y="absolute_delta_percent",
        x="perturb_prob",
        hue=hue_col,
        data=df[df["perturber"] != "ReplaceWithRandomCharacter"],
        palette=sns.color_palette(palette, n_colors=n_colors)
    )
    
    plt.title(f"{num_shot}-shot {dataset_name} ({model_name})", fontsize = 40)
    plt.legend(loc=legend_pos)
    ax.set_ylim([0, ylimit])
    ax.set_xlim([0, 1])
    ax.set(xlabel='Perturbation Probability')
    ax.set(ylabel=f"Relative % Change in {plot_metric}")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"../mstar-robustness-figs/absolute-deltas-{num_shot}-shot-{dataset_name}-by-{hue_col}-{model_name}.png",format="png")


def get_slim_filename(
    file_dir, model, dataset, perturbers, num_shot=0, perturb_prob=0.25, seed=1234, perturb_seed=42, run_no='001'
    ):
    if perturbers == "Original":
        perturb_prob = "0.0"
    elif perturbers == "all_prompt":
        perturb_prob = "1.0"

    filenames = glob(os.path.join(file_dir, f"slim*model={model}.task={dataset}*fewshot={num_shot}*ehseed={seed}.ptbrs={perturbers}.ptbseed={perturb_seed}.p={perturb_prob}.run={run_no}.json"))
    
    if len(filenames) > 1:
        print(filenames)
        filenames = [f for f in filenames if "bs=auto" in f]
        print(filenames)

    assert len(filenames) == 1, f"slim*model={model}.task={dataset}*fewshot={num_shot}*ehseed={seed}.ptbrs={perturbers}.ptbseed={perturb_seed}.p={perturb_prob}.run={run_no}.json"

    filename = filenames[0]

    print("Retrieved", filename)
    return filename
