from cgi import test
from hydra import compose, initialize
from imitation_learning.run import main

BASE_CONFIG_DIRECTORY = "../imitation_learning/config"


def test_with_initialize() -> None:
    with initialize(config_path=BASE_CONFIG_DIRECTORY):
        cfg = compose(config_name="default", overrides=["+ppo=pointmass", "logger=cli"])
        eval_result = main(cfg)
        assert (
            eval_result["dist_to_goal"] < 0.03
        ), f"Final distance to goal was {eval_result['dist_to_goal']}"


if __name__ == "__main__":
    test_with_initialize()
