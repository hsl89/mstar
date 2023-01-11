import random
from lm_eval.api.model import CachingLM
from lm_eval.api.task import PromptSourceTask
from lm_eval.api.utils import set_seed
from lm_eval.evaluator import evaluate
from lm_eval.models import get_model_from_args_string
from lm_eval.tasks import get_task_list
from batch_size_estimator import calculate_batchsize
from tqdm import tqdm
from perturbers import get_perturber
from perturbers.perturber import InputPerturber
from perturbers.prompt_perturbers import ExampleOrderPerturber, ExampleSeparatorPerturber, PromptSuffixPerturber, TextTargetSeparatorPerturber

orig_fewshot_context = PromptSourceTask.fewshot_context
orig_doc_to_text = PromptSourceTask.doc_to_text
orig_fewshot_examples = PromptSourceTask.fewshot_examples

def perturbed_fewshot_examples(self, docs, k, rng, prompt=None):
    """Injects ExampleOrderPerturber into original `fewshot_examples` method."""

    fewshot_examples, fewshot_idx = orig_fewshot_examples(self, docs, k, rng,prompt)
    if isinstance(self.perturber, ExampleOrderPerturber):
        fewshot_idx = self.perturber.perturb_order(fewshot_idx)
    return fewshot_examples, fewshot_idx

def fewshot_context_w_perturber_logged(self, doc, num_fewshot, rng=None):
    """Injects perturbers into original `fewshot_context` method and adds perturbers to the logged example."""

    ctx, logging_info = orig_fewshot_context(self=self, doc=doc, num_fewshot=num_fewshot, rng=rng)
    logging_info['perturber'] = type(self.perturber).__name__
    logging_info['perturb_prob'] = self.perturber.perturb_prob
    logging_info['perturb_seed'] = int(self.perturber.seed)
    return ctx, logging_info

def doc_to_perturbed_text(self, doc) -> str:
    """Injects perturbers into original `doc_to_text` method."""

    text = orig_doc_to_text(self, doc)
    if isinstance(self.perturber, InputPerturber) or isinstance(self.perturber, PromptSuffixPerturber):
        return self.perturber.perturb(text)
    else:
        return text

def get_task_list_w_perturber(perturber, task_name, template_names):
    """Injects perturbers into retrieved Tasks."""

    task_list = get_task_list(task_name, template_names)
    for i in range(len(task_list)):
        task_list[i].perturber = perturber
        if issubclass(type(perturber), ExampleSeparatorPerturber):
            task_list[i].example_separator = perturber.get_example_separator()

        if issubclass(type(perturber), TextTargetSeparatorPerturber):
            task_list[i].text_target_separator = perturber.get_text_target_separator()

    return task_list

def get_subsampled_indices(subsample_factor, task_name, template_names):
    """Subsamples dataset and returns sampled indices for each split."""
    
    tasks = get_task_list(task_name, template_names)
    assert len(tasks) > 0

    task = tasks[0] # we assume all tasks use the same HF dataset

    splits = list(task.dataset.keys()) # task.dataset is a DatasetDict where the keys are splits, e.g, 'dev', 'test'.

    num_examples_per_split = { split: len(task.dataset[split]) for split in splits }
    
    # TODO: Switch to an independent random number generator instead of relying on `random.seed()``
    subsampled_indices_per_split = {
        split: random.sample(range(num_examples), k = num_examples // subsample_factor) \
        for split, num_examples in num_examples_per_split.items()
    }    

    return subsampled_indices_per_split

def get_sampled_task_indices(num_sampled_templates, task_name, template_names):
    """Samples prompt templates and returns sampled indices."""
    # TODO: Switch to an independent random number generator instead of relying on `random.seed()``
    task_indices = list(range(len(get_task_list(task_name, template_names))))
    return random.sample(task_indices, k=num_sampled_templates)

def subsample_dataset(dataset, subsampled_indices):
    """Retrieve the subsampled examples."""

    # we directly modify the datasets in task.dataset
    for split, indices in subsampled_indices.items():
        dataset[split] = dataset[split].select(indices)

def perturb_evaluate(
    model_api_name,
    model_args,
    task_name,
    template_names,
    num_sampled_templates=-1,
    example_subsample_factor=1, #modified
    num_fewshot=0,
    batch_size=None,
    device=None,
    use_cache=False,
    limit=None,
    bootstrap_iters=100000,
    seed=1234,
    perturber_list=None, #modified
    perturb_seed=42, #modified
    perturb_prob=0 #modified
):
    """Modified `cli_evaluate` from lm-evaluation-harness. Instantiate and evaluate a model on a list of tasks with perturbations.

    :param model: Union[str, LM]
        Name of model or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str]
        String arguments for each model class, see LM.create_from_arg_string.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int, optional
        Batch size for model
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param no_cache: bool
        Whether or not to cache
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param description_dict: dict[str, str]
        Dictionary of custom task descriptions of the form: `task_name: description`
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param seed: int
        Random seed.
    :param perturber_list: list[str]
        List of perturber keys
    :param perturb_seed: int
        Seed for perturbation randomness
    :param perturb_prob: float
        Probability of applying the perturbation
    :return
        Dictionary of results
    """
    # original code from evaluate()
    assert task_name != None, "No tasks specified"
    set_seed(seed)

    model = get_model_from_args_string(
        model_api_name, model_args, {
            "batch_size": int(batch_size) if batch_size.isnumeric() else 0, # modified for auto batchsize calculation
            "device": device
        }
    )

    if use_cache:
        cache_args = model_args.replace("=", "-").replace(",", "_").replace("/", "-")
        cache_location = f"lm_cache/{model_api_name}_{cache_args}.db"
        model = CachingLM(model, cache_location)
    # end original
    
    results = {}
    config = {
                "model": model_api_name,
                "model_args": model_args,
                "num_fewshot": num_fewshot,
                "batch_size": batch_size if batch_size else "auto",
                "example_subsample_factor": example_subsample_factor,
                "device": device,
                "no_cache": use_cache,
                "limit": limit,
                "bootstrap_iters": bootstrap_iters,
                "eh_seed": seed,
            }


    # get the indices first to ensure we have the same examples across prompts
    subsampled_indices = get_subsampled_indices(example_subsample_factor, task_name, template_names)

    # get the indices first to ensure we have the same prompt templates across perturbations
    if num_sampled_templates > 0 and num_sampled_templates < len(template_names):
        task_indices = get_sampled_task_indices(num_sampled_templates, task_name, template_names)

    for perturber_name in tqdm(perturber_list, desc="outer loop"):
        perturber = get_perturber(perturber_name)(perturb_seed, perturb_prob)

        tasks = get_task_list_w_perturber(perturber, task_name, template_names) # modified function from original
        
        if num_sampled_templates > 0 and num_sampled_templates < len(tasks):
            tasks = [tasks[s] for s in task_indices]

        for task in tasks:
            # added logic for subsampling datasets
            if example_subsample_factor > 1:
                subsample_dataset(task.dataset, subsampled_indices)

        if batch_size == "auto":
            # added logic for calculating batch size automatically
            found_bs = calculate_batchsize(model, tasks, num_fewshot, seed)
            print("Batch size found:", found_bs)
            model._batch_size = found_bs # set batch size for the model

        results[perturber_name] = evaluate(
            model=model,
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
            bootstrap_iters=bootstrap_iters,
            seed=seed,
        ) # original function call 
        
        results[perturber_name]["config"] = None # added for compatibility with agg2slim
        results[perturber_name]["perturber"] = perturber_name 
        results[perturber_name]["perturb_seed"] = perturber.seed
        results[perturber_name]["perturb_prob"] = perturber.perturb_prob
    return results, config
