from rl_utils.logging import Logger


def test_base_logger():
    logger = Logger(
        "test_run", 31, "./data/logs", "./data/vids", "./data/save_dir", 5, {}
    )
    logger.collect_info("test_val", 2.3)
    logger.collect_info("test_val2", 2.9, no_rolling_window=True)

    logger.interval_log(0, 0)

    logger.close()
