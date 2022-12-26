from hydra import compose, initialize
import mstar.models.t5
import models.utils


def test_model_size_counting():
    with initialize("../config"):
        cfg = compose(config_name="base", overrides=["model=tiny"])

    hf_model_config = mstar.models.t5.MStarT5Config(**cfg.model)
    model = mstar.models.t5.MStarT5ForConditionalGeneration(hf_model_config)

    # only do the count with a small model size
    model_params_billion = sum(x.numel() for x in model.parameters()) / 1000000000

    # compute number of params in billions
    num_params_billion = models.utils.count_model_parameters(hf_model_config)
    assert model_params_billion == num_params_billion
