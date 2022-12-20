"""
Get model size (in B parameters) and tflops
Save model tracer bullet if necessary

USAGE:
To run for a given model size, such as 11B

#get model size and tflops from a yaml config in config/models/
python get_model_info.py model=11B

#create and save the tracer bullet
python get_model_info model=11B ++save_tracer_bullet=1

#create and save the tracer bullet in a user-specified location, i.e. FSX
python get_model_info model=11B ++save_tracer_bullet=1 ++automodel_save_path=/mnt_out/colehawk/tmp/
"""

import hydra
import logging
import mstar.models.t5
import models.utils

@hydra.main(version_base=None, config_path="config", config_name="base.yaml")
def main(cfg):

    logging.info(cfg.model)
    hf_model_config = mstar.models.t5.MStarT5Config(**cfg.model)

    num_params = models.utils.count_model_parameters(hf_model_config)
    logging.info(f"{num_params:.2f}B parameters")

    #count model TFLOPS
    tflops_per_train_step = mstar.utils.flops_calc.compute_tflops_per_gpu(
                model_type="encoder_decoder",
                sec_per_step=1.0,  # will get actual time during each train-step
                micro_batchsize=1,
                activation_checkpointing=cfg.model.gradient_checkpointing,
                vocab_size=cfg.model.vocab_size,
                hidden_size=cfg.model.d_model,
                decoder_num_layers=cfg.model.num_decoder_layers,
                encoder_num_layers=cfg.model.num_layers,
                decoder_seq_len=1024,
                encoder_seq_len=2048,
                use_gated_mlp=getattr(hf_model_config, "is_gated_act", False),
            )
    logging.info(f"{tflops_per_train_step:.2f} TFLOPs per step")
    
    #instantiate the model and save a tracer bullet
    #may consume a lot of local storage
    if getattr(cfg,'save_tracer_bullet',False):
        logging.info("Creating tracer bullet model, random init")
        model = mstar.models.t5.MStarT5ForConditionalGeneration(config=hf_model_config)
        logging.info("Saving tracer bullet model")
        #mstar model api assumes 1 shard
        model.save_pretrained(getattr(cfg,"automodel_save_path","automodel"),max_shard_size='999GB')


if __name__=='__main__':
    main()
