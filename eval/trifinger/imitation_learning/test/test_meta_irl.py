from hydra import compose, initialize
from imitation_learning.run import main

BASE_CONFIG_DIRECTORY = "../imitation_learning/config"


def test_with_initialize() -> None:
    with initialize(config_path=BASE_CONFIG_DIRECTORY):
        cfg = compose(
            config_name="default",
            overrides=[
                "+meta_irl=pointmass",
                "logger=cli",
                "num_env_steps=129280",
                "seed=0",
            ],
        )
        eval_result = main(cfg)

        print(eval_result)

        assert (
            eval_result["dist_to_goal"] == 0.092275844886899
        ), f"distance to goal was {eval_result['dist_to_goal']}"


if __name__ == "__main__":
    test_with_initialize()
