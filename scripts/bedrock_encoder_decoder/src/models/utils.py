"""
General utilities to accompany model creation
"""

import deepspeed
import mstar.models.t5
import torch as th
import pytorch_lightning as pl
import contextlib
import logging

logger = logging.getLogger(__name__)


def load_model(
    model_config,
    trainer,
    precision,
    state_dict_path=None,
    state_dict_attribute="auto",
    key_prefix="auto",
    meta_context=False,
):
    """
    Load a model taking deepspeed model sharding into account
    """
    assert precision in (32, 16, "bf16")

    unwrapped_state_dict = None
    if state_dict_path and trainer.is_global_zero:
        logger.info(
            f"loading the model parameters from the ckpt file {state_dict_path!r}"
        )
        state_dict = th.load(state_dict_path, map_location="cpu")
        if state_dict_attribute == "auto":
            if "state_dict" in state_dict:
                state_dict_attribute = "state_dict"
            elif "module" in state_dict:
                state_dict_attribute = "module"
            else:
                logger.warning(
                    "Unable to induce `state_dict_attribute`. Default to `None`."
                )
                state_dict_attribute = None
        if key_prefix == "auto":
            if "state_dict" in state_dict:
                key_prefix = "model."
            elif "module" in state_dict:
                key_prefix = "module.model."
            else:
                logger.warning("Unable to induce `key_prefix`. Default to `None`.")
                key_prefix = None
        if state_dict_attribute:
            logger.info(f"using state dict attribute {state_dict_attribute!r}")
            state_dict = state_dict[state_dict_attribute]
        if key_prefix:
            logger.info(f"using state dict key prefix {key_prefix!r}")
            unwrapped_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith(key_prefix):
                    new_key = key[len(key_prefix) :]
                    unwrapped_state_dict[new_key] = value
        else:
            unwrapped_state_dict = state_dict

    context = contextlib.nullcontext()
    zero_stage_3 = (
        isinstance(
            trainer.training_type_plugin,
            pl.plugins.training_type.deepspeed.DeepSpeedPlugin,
        )
        and trainer.training_type_plugin.zero_stage_3
        and not meta_context
    )
    if zero_stage_3:
        if trainer.training_type_plugin.precision in (16, "mixed"):
            dtype = th.float16
        elif trainer.training_type_plugin.precision == "bf16":
            dtype = th.bfloat16
        else:
            dtype = th.float32

        context = deepspeed.zero.Init(
            remote_device=trainer.training_type_plugin.remote_device,
            pin_memory=True,
            config=trainer.training_type_plugin.config,
            dtype=dtype,
        )

    with context:

        model = mstar.models.t5.MStarT5ForConditionalGeneration(model_config)

        if not zero_stage_3 and unwrapped_state_dict is not None:
            model.load_state_dict(unwrapped_state_dict)
            logger.info(
                f"loaded the model parameters from the state dict {state_dict_path!r}"
            )

            return model

    if zero_stage_3 and unwrapped_state_dict is not None:

        def load(module: th.nn.Module, prefix=""):
            nonlocal unwrapped_state_dict
            missing_keys = []
            unexpected_keys = []
            error_msgs = []
            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(unwrapped_state_dict, "_metadata", None)
            state_dict = unwrapped_state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            # because zero3 puts placeholders in model params, this context
            # manager gathers (unpartitions) the params of the current layer, then loads from
            # the state dict and then re-partitions them again
            with deepspeed.zero.GatheredParameters(
                list(module.parameters(recurse=False)), modifier_rank=0
            ):
                if trainer.is_global_zero:
                    module._old_load_from_state_dict(
                        state_dict=state_dict,
                        prefix=prefix,
                        local_metadata=local_metadata,
                        strict=True,
                        missing_keys=missing_keys,
                        unexpected_keys=unexpected_keys,
                        error_msgs=error_msgs,
                    )

            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(model, prefix="")

    return model
