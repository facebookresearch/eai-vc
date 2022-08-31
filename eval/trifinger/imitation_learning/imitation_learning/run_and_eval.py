import hydra
from omegaconf import OmegaConf

from imitation_learning.run import main


@hydra.main(config_path="config", config_name="default")
def run_and_eval(cfg):
    print("Training reward function")
    run_result = main(cfg)

    eval_cfg = OmegaConf.merge(cfg, cfg.eval_args)
    eval_cfg.logger.run_name = "eval_" + run_result["run_name"]
    eval_cfg.load_checkpoint = run_result["last_ckpt"]
    eval_cfg.load_policy = False

    print("Evaluating reward function")
    main(eval_cfg)


if __name__ == "__main__":
    run_and_eval()
