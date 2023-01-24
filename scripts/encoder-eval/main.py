from mteb import MTEB
from copy import deepcopy
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import time
import json
import traceback

from mstar.utils.lightning import MStarEKSLogger

import load_models
import inference
import utils
import model_class
import hydra
from omegaconf import DictConfig, OmegaConf

import os
from loguru import logger
import datasets

# global configs
dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, "config.json"), "r") as f:
    config = json.load(f)

datasets.config.HF_DATASETS_CACHE = config["data_cache_dir"]
ERROR_LOG_PATH = config["error_log_path"]
LOG_PATH = config["log_path"]

logger = utils.get_logger()

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    return


def cleanup():
    dist.destroy_process_group()
    return


def evaluate(rank, world_size, cfg: DictConfig):
    if world_size > 1:
        setup(rank, world_size)

    load_model_fn = getattr(load_models, cfg.model.load_model_fn)
    inference_fn = getattr(inference, cfg.model.inference_fn)

    model = getattr(
        model_class,
        "FakeInferenceModel" if cfg.use_fake_inference else cfg.model.model_class,
    )(
        rank=rank,
        world_size=world_size,
        load_model_fn=load_model_fn,
        inference_fn=inference_fn,
        debug=cfg.debug,
        use_ddp=cfg.use_ddp,
        task_params=cfg.task,
        model_params=cfg.model.model_params,
        tokenizer_params=cfg.model.tokenizer_params,
        data_params=cfg.model.data_params,
    )

    evaluation = MTEB(
        tasks=cfg.task.task_name,
        err_logs_path=ERROR_LOG_PATH,
        task_langs=cfg.task.task_langs if "task_langs" in cfg.task else None,
        **cfg.model.evaluator_constructor_kwargs.get(cfg.task.task_type, {})
    )

    # MTEB delete the task once it is completed
    tasks = deepcopy(evaluation.tasks)


    try:
        results = evaluation.run(
            rank=rank,
            model=model,
            output=None,
            overwrite_results=True,
            eval_splits=cfg.split if "split" in cfg else None,
            **cfg.model.evaluator_run_kwargs.get(cfg.task.task_type, {})
        )
        if rank == 0:
            eval_splits = (
                cfg.split if "split" in cfg else task.description["eval_splits"]
            )
            for task in tasks:
                main_score = task.description["main_score"]
                for split in eval_splits:
                    task_name = task.description["name"]
                    print("*** result for %s of %s ***" % (split, task_name))
                    res = results.get(task_name, {}).get(split, {})
                    if res == {}:
                        print(
                            "Error encountered while evaluating %s; Check error_log.txt for detail"
                            % task_name
                        )

                    print(res)
                    evaluation_time = res["evaluation_time"]
                    if evaluation._task_langs:
                        res = utils.average_over_languages(res, evaluation._task_langs)
                        print("after averaging over languages")
                        print(res)
                    if "KUBERNETES_SERVICE_HOST" in os.environ:
                        eks_logger = MStarEKSLogger(
                            experiment_name=cfg.exp_name,
                            run_name="%s_%s_%s"
                            % (cfg.model.model_params.model_name, task_name, split),
                            tags={"mode": "inference"},
                        )

                        eks_logger.log_hyperparams(cfg)
                        if os.path.exists(ERROR_LOG_PATH):
                            eks_logger.log_artifact(ERROR_LOG_PATH)

                        if os.path.exists(LOG_PATH):
                            eks_logger.log_artifact(LOG_PATH)

                        eks_logger.log_metrics(
                            {
                                main_score: res[main_score]
                                if cfg.task.task_type != "sts"
                                else res["cos_sim"]["spearman"],
                                "evaluation_time": evaluation_time
                            }
                        )

                        eks_logger.close()

    except Exception as e:
        logger.error(
            "Rank: %s | Exception encountered while running evalution: %s" % (rank, e)
        )
        logger.error("Rank: %s | Full traceback:\n %s" % (rank, traceback.format_exc()))
        if world_size > 1:
            cleanup()
        raise e

    if world_size > 1:
        cleanup()
    return


@hydra.main(
    version_base=None, config_path=os.path.join(dir_path, "conf"), config_name="config"
)
def main(cfg):
    print("========= task configuration ============")
    print(OmegaConf.to_yaml(cfg))

    # download data in the main process
    tasks = (
        cfg.task.task_name
        if isinstance(cfg.task.task_name, list)
        else [cfg.task.task_name]
    )
    e = MTEB(tasks=tasks)
    print("=============== loading task data ==================")
    e.load_tasks_data()

    world_size = torch.cuda.device_count() if cfg.use_ddp else 1
    if cfg.use_ddp:
        ctx = mp.get_context("spawn")
        procs = []
        for i in range(world_size):
            p = ctx.Process(
                target=evaluate,
                kwargs={"rank": i, "world_size": world_size, "cfg": cfg},
            )
            p.start()
            procs.append((p, time.time()))

        for i, (p, s) in enumerate(procs):
            logger.info("***** waiting to process %s to finish *****\n\n" % i)
            p.join()
            e = time.time()
            logger.info(
                "***** process %s finished in %0.2f seconds *****\n\n" % (i, e - s)
            )
    else:
        evaluate(rank=0, world_size=1, cfg=cfg)

    return


if __name__ == "__main__":
    main()
