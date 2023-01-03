import torch
import hydra

if __name__=="__main__":
    with hydra.initialize("config/optimization/optimizer"):
        cfg = hydra.compose(config_name="adam")

    print(cfg)
    
    module = torch.nn.Linear(10,10).cuda()

    opt = hydra.utils.instantiate(cfg, params=module.parameters())

    print(opt)
