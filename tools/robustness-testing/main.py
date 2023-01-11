"""
Copied and modified from lm-evaluation-harness
"""
import argparse
import datetime
import time
import json
import logging
import os
from lm_eval import evaluator
from lm_eval.api import utils
import monkey_patch
from patched_functions import perturb_evaluate
from perturbers import get_all_perturber_keys, get_all_input_perturber_keys, get_all_prompt_perturber_keys, get_all_training_input_perturber_keys, get_all_zeroshot_prompt_perturber_keys

logger = logging.getLogger("main")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_api_name", required=True, 
        help="Name of the model API to use. See `lm_eval.list_model_apis()` for available APIs."
    )
    parser.add_argument(
        "--model_args", default="", 
        help="Model constructor args that you'd pass into a model of type, `--model_api_name`. These must be comma-separated keyword args, e.g. `key1=value1,key2=value2`, with no spaces."
    )
    parser.add_argument(
        "--task_name", required=True,
        help="Name of the task to use as found in the lm_eval registry. See: `lm_eval.list_tasks()`."
    )
    parser.add_argument(
        "--run_no", type=int, default=1,
        help="Used to differentiate runs if the filenames are otherwise the same, e.g., the step number can be passed to this parameter for multiple checkpoints of the same model."
    )
    parser.add_argument(
        "--template_names", default="original_templates",
        help="""Comma-separated list of template names for the specified
        task. Example:
        `> python main.py ... --task_name rte --template_names imply,mean`
        - Default: `all_templates`
        - General Selectors:
            - `"all_templates"`: Selects all templates for the task
            - `"original_templates"`: Selects only templates that are designed to match the original task
        """
    )
    parser.add_argument(
        "--num_sampled_templates", type=int, default=-1,
        help="The number of prompt templates to use."
    )
    parser.add_argument(
        "--template_idx", type=int, default=None,
        help="Choose template by index from available templates."
    )
    parser.add_argument(
        "--bootstrap_iters", type=int, default=100000,
        help="Iters for stderr computation."
    )
    parser.add_argument(
        "--num_fewshot", type=int, default=0,
        help="Number of shots."
    )
    parser.add_argument(
        "--seed", type=int, default=1234,
        help="Seed for lm-evaluation-harness--related functions."
    )
    parser.add_argument(
        "--perturb_seed", type=int, default=42,
        help="Seed for perturbation-related functions."
    )
    parser.add_argument(
        "--example_subsample_factor", type=int, default=1, 
        help="The number of times to reduce the dataset by, e.g., a subsample factor of 10 reduces the dataset size by 10."
    )
    parser.add_argument("--batch_size", type=str, default="auto",
        help="The batch size to use. Use `auto` for the batch size to be automatically estimated."
    )
    parser.add_argument("--device", type=str, default=None,
        help="The device to place your model onto, e.g. cuda:0. For large models available through the HuggingFace Hub you should use `accelerate` by passing `use_accelerate=True` to `--model_args`.",
    )
    parser.add_argument("--output_dir", type=str, default='../robustness-outputs/', 
        help="The folder where the results will be saved."
    )
    parser.add_argument("--perturbers", type=str, default="Original",
        help="""Comma-separated list of perturber names for the specified task. Example:
        `> python main.py ... --perturbers Original,AddLetterPerturber`
        - Default: `Original`
        - General Selectors:
            - `all`: Selects all perturbers
            - `all_but_original`: Selects all perturbers but the original
            - `all_input`: Selects all input perturbers
            - `all_input_no_rnd: Selects all input perturbers but ReplaceWithRandomCharacterPerturber
            - `all_prompt`: Selects all prompt-specific perturbers
        """
    )
    parser.add_argument(
        "--perturb_prob", type=float, default=0,
        help="Probability of a perturbation being applied, used to control the perturbation strength."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit the number of examples to evaluate on; ONLY USE THIS FOR DEBUGGING PURPOSES."
    )
    parser.add_argument(
        "--output_path", default=None, 
        help="""Use output_path as `output_filename`. For example:
        `> python main.py ... --output_path blop`
        # saves files into `outputs/blop.json`
        Warning: You currently cannot change/add folder structure.
        """
    )
    return parser.parse_args()


def args_to_name(args, separator):
    """Map `args` to file name. If output_path is set, we use that instead."""
    def _fix_model_name(model, model_args):
        if model_args == "":
            return model
        elif "pretrained" not in model_args:
            logger.warning("WARNING: Unprepared for these model args.")
            return f"{model}={model_args}"

        model_name = None
        rev_name = None

        for arg in model_args.split(","):
            # Example:
            #   pretrained=google/t5-base-lm-adapt --> google-t5-base-lm-adapt
            if "pretrained" in arg:
                model_name = arg.split("=")[-1]
                if model_name.count("/") > 1:
                    model_name = model_name.split("/")[-1]
                else:
                    model_name = model_name.replace("/", "-")
                
                if len(model_name) > 20:
                    model_name = model_name[:10] + '...' + model_name[-10:]
                    model_name = model_name[:10] + '...' + model_name[-10:]
            
            if arg.split("=")[0] == "revision":
                rev_name = arg.split("=")[-1]
                
        if model_name:
            if rev_name:
                return model_name + "-" + rev_name
            return model_name

    fields = {
        "model": _fix_model_name(args.model_api_name, args.model_args),
        "task": args.task_name,
        "templates": args.template_names,
        "fewshot": str(args.num_fewshot),
        "bs": args.batch_size if args.batch_size.isnumeric() else "auto",
        "esf": str(args.example_subsample_factor) if args.example_subsample_factor > 1 else None,
        "ehseed": str(args.seed),
        "ptbrs": args.perturbers,
        "ptbseed": str(args.perturb_seed),
        "p": str(args.perturb_prob),
        "run": str(args.run_no).zfill(3),
    }
    if args.num_sampled_templates > 0:
        fields["templates"] += "C" + str(args.num_sampled_templates)
    fields = [f"{k}={v}" for k, v in fields.items() if v is not None]
    # Some prompts also have "/" in them!
    filename = f"{separator}".join(fields).replace("/", "-")
    if args.limit is not None:
        # Do not use limited files for final analysis.
        return f"limited={args.limit}{separator}" + filename

    return filename

def setup_example_logger(output_dir, output_path, separator):
    """Sets up a logger that will save each example and prediction."""
    logger = logging.getLogger("examples")
    filename = f"{output_dir}/examples{separator}{output_path}.jsonl"
    formatter = logging.Formatter("%(message)s")
    handler = logging.FileHandler(filename)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def format_results(args, results, config, slim=False):
    from scripts.agg2slim import agg2slim

    formatted_results = {"results": [], "config": config}
    for perturber, p_results in results.items():
        formatted_p_results = agg2slim(p_results) if slim else p_results.copy()
        del formatted_p_results["config"]
        formatted_results["results"].append(
            {
                "perturber": perturber,
                "perturb_seed": args.perturb_seed,
                "perturb_prob": args.perturb_prob,
                "results": formatted_p_results["results"],
            }
        )
            
    return formatted_results

def main():
    start_time = time.monotonic()
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.limit:
        logger.warning(
            "\n» WARNING: `--limit` SHOULD ONLY BE USED FOR TESTING. REAL METRICS "
            "SHOULD NOT BE COMPUTED USING LIMIT."
        )

    template_names = utils.cli_template_names(
        args.task_name, args.template_names, args.template_idx
    )

    path_separator = "."
    output_path = args_to_name(args, separator=path_separator)
    setup_example_logger(args.output_dir, output_path, path_separator)

    if args.perturbers.lower() == 'all':
        perturber_list = get_all_perturber_keys()
    elif args.perturbers.lower() == 'all_but_original':
        perturber_list = [p for p in get_all_perturber_keys() if p != 'Original']
    elif args.perturbers.lower() == 'all_input':
        perturber_list = get_all_input_perturber_keys()
    elif args.perturbers.lower() == 'all_input_no_rnd':
        perturber_list = get_all_training_input_perturber_keys()
    elif args.perturbers.lower() == 'all_prompt':
        if args.num_fewshot == 0:
            perturber_list = get_all_zeroshot_prompt_perturber_keys()
        else:
            perturber_list = get_all_prompt_perturber_keys()
    else:
        perturber_list = args.perturbers.split(',')

    if args.batch_size.isnumeric():
        args.batch_size = int(args.batch_size)

    results, config = perturb_evaluate(
        model_api_name=args.model_api_name,
        model_args=args.model_args,
        task_name=args.task_name,
        template_names=template_names,
        num_sampled_templates=args.num_sampled_templates,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        example_subsample_factor=args.example_subsample_factor,
        device=args.device,
        use_cache=False,
        limit=args.limit,
        seed=args.seed,
        perturber_list=perturber_list,
        perturb_seed=args.perturb_seed,
        perturb_prob=args.perturb_prob,
        bootstrap_iters=args.bootstrap_iters,
    )

    
    with open(f"{args.output_dir}/agg{path_separator}{output_path}.json", "w") as f:
        json.dump(
            format_results(args, results, config),
            f, indent=2
        )

    with open(f"{args.output_dir}/slim{path_separator}{output_path}.json", "w") as f:
        # We add `indent = 2` to help with quick readability.
        json.dump(
            format_results(args, results, config, slim=True),
            f, indent=2
        )
        
    for perturber in perturber_list:
        print(perturber, "|", "perturb_prob:", args.perturb_prob)
        print(evaluator.make_table(results[perturber]))
    
    end_time = time.monotonic()
    exec_time = datetime.timedelta(seconds=end_time - start_time)
    
    with open(f"{args.output_dir}/time{path_separator}{output_path}.txt", "w") as f:
        f.write(str(exec_time))
    
    print("Time taken:", exec_time)


if __name__ == "__main__":
    main()
