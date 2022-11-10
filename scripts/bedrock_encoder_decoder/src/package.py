"""
Convert a Deepspeed zero checkpoint to M* automodel
Only supports MStarT5ForConditionalGeneration
"""
import os
import logging
import hydra
import mstar
import mstar.models.t5
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
import torch

# single shard for M* models due to download assumptions
MAX_SHARD_SIZE = "999GB"

logger = logging.getLogger()
logger.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path="config", config_name="package")
def main(cfg):
    """
    Convert deepspeed checkpoint to automodel
    """

    if cfg.save_location is None:
        cfg.save_location = cfg.ckpt_folder

    full_state_dict_save_path = os.path.join(
        cfg.save_location, cfg.full_state_dict_save_name
    )

    # utility function will save optimizer states, model states, etc
    logger.info(
        "Using pytorch lightning utility function to save full pytorch lightning state"
    )
    convert_zero_checkpoint_to_fp32_state_dict(
        cfg.ckpt_folder, full_state_dict_save_path
    )

    logger.info("Loading back full state to obtain only model state")
    fp32_state_dict = torch.load(full_state_dict_save_path)

    # strip out extra "model" added by pytorch lightning for M* loading via AutoModel
    logger.info(
        "Stripping out extra pytorch lightning keys and reformatting for M* automodel load"
    )
    stripped_state_dict = {
        key[len("model.") :]: val for key, val in fp32_state_dict["state_dict"].items()
    }

    # to save RAM, don't need optimizer states
    del fp32_state_dict

    # create the model config for testing/packaging
    hf_model_config = mstar.models.t5.MStarT5Config(**cfg.model)

    # make sure state dict shapes match model config
    logger.info("Loading state dict into model to check parameters match")

    model = mstar.models.t5.MStarT5ForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=None,
        config=hf_model_config,
        state_dict=stripped_state_dict,
    )

    logger.info("Parameter load succeeded, model parameter shapes match state dict")

    automodel_save_location = os.path.join(cfg.save_location, "automodel")
    logger.info("Saving final automodel to %s", automodel_save_location)
    # single shard for M* models due to download assumptions
    model.save_pretrained(
        os.path.join(cfg.save_location, "automodel"), max_shard_size=MAX_SHARD_SIZE
    )


if __name__ == "__main__":
    main()
