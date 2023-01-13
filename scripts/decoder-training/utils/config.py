import hydra
import logging

def validate_config(cfg, hf_model_config):
    validate_config_softmax(cfg, hf_model_config)
    validate_config_checkpointing(cfg)

def validate_config_checkpointing(cfg):
    """
    Make sure checkpointing occurs after a 
    val step not right before. Leads to better
    checkpoint name formatting
    """

    #user can choose to skip override and decouple val_check and checkpointing
    if not getattr(cfg,"skip_val_check_override",0):
        assert cfg.trainer.val_check_interval is None, "Should not set val_check_interval, only use checkpointing frequency"
        logging.info("Overriding val check interval {cfg.trainer.val_check_interval} for better ckpt formatting")
        #override value 
        # -1 is sufficient for proper checkpoint naming
        # -2 sufficient to (usually) avoid validation starting immediately on checkpoint resume
        # validation end step on checkpoint resume can cause failure on startup 
        # since on_validation_epoch_end gets empty list
        cfg.trainer.val_check_interval=cfg.lightning.callbacks.checkpoint.every_n_train_steps-2

def validate_config_softmax(cfg, hf_model_config):
    """
    Checks that model config matches trainer precision type
    """
    
    if cfg.model.fused_scaled_masked_softmax:
        # make sure trainer/softmax precision match
        assert cfg.trainer.precision in {16, 'bf16'}, f'Trainer precision {cfg.trainer.precision} is not supported in Fused Softmax.'
        if cfg.trainer.precision == 16:
            expected_precision = 'fp16'
        else:
            expected_precision = 'bf16'

        assert hf_model_config.softmax_precision == expected_precision, f'Trainer precision "{cfg.trainer.precision}" should match softmax precision "{hf_model_config.softmax_precision}".'
