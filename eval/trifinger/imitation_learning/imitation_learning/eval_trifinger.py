import hydra
from omegaconf import OmegaConf

from imitation_learning.run_mirl_trifinger import main


@hydra.main(config_path="config/meta_irl", config_name="trifinger")
def run_and_eval(cfg):
    print("Training reward function")

    ### EVAL reward fun
    run_name = "515-0-lCPjxW"
    eval_cfg = OmegaConf.merge(cfg, cfg.eval_args)
    eval_cfg.load_checkpoint = (
        f"/Users/fmeier/projects/imitation-learning/imitation_learning/multirun/2022-05-15"
        f"/14-17-38/0/data/checkpoints/{run_name}/ckpt.11000.pth"
    )

    # eval on train
    start_state_noise = 0.01
    eval_cfg.logger.run_name = f"eval0_{run_name}_start_noise_{start_state_noise}"
    eval_cfg.load_policy = False
    eval_cfg.env_settings.start_state_noise = start_state_noise
    eval_cfg.train_or_eval = "eval0"

    print("Evaluating reward function")
    main(eval_cfg)
    #
    # # eval on test distr 1
    # start_state_noise = 0.02
    # run_name = run_result["run_name"]
    # eval_cfg.logger.run_name = f"eval1_{run_name}_start_noise_{start_state_noise}"
    # eval_cfg.env_settings.start_state_noise = start_state_noise
    # eval_cfg.train_or_eval = "eval1"
    #
    # print("Evaluating reward function")
    # main(eval_cfg)
    #
    # # eval on test distr 2
    # start_state_noise = 0.03
    # eval_cfg.logger.run_name = f"eval2_{run_name}_start_noise_{start_state_noise}"
    # eval_cfg.env_settings.start_state_noise = start_state_noise
    # eval_cfg.train_or_eval = "eval2"
    #
    # print("Evaluating reward function")
    # main(eval_cfg)
    #
    # # eval on test distr 3
    # start_state_noise = 0.04
    # eval_cfg.logger.run_name = f"eval3_{run_name}_start_noise_{start_state_noise}"
    # eval_cfg.env_settings.start_state_noise = start_state_noise
    # eval_cfg.train_or_eval = "eval3"
    #
    # print("Evaluating reward function")
    # main(eval_cfg)
    #
    # # eval on test distr 4
    # start_state_noise = 0.05
    # eval_cfg.logger.run_name = f"eval4_{run_name}_start_noise_{start_state_noise}"
    # eval_cfg.load_checkpoint = run_result["last_ckpt"]
    # eval_cfg.load_policy = False
    # eval_cfg.env_settings.start_state_noise = start_state_noise
    # eval_cfg.train_or_eval = "eval4"
    #
    # print("Evaluating reward function")
    # main(eval_cfg)


if __name__ == "__main__":
    run_and_eval()
