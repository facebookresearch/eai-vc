import logging
import omegaconf


log = logging.getLogger(__name__)

def setup_wandb(config):
    if isinstance(config, omegaconf.DictConfig):
        config = omegaconf.OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True
        )
    wandb_cfg_dict = config["wandb"]

    try:
        log.info(f"wand initializing...")
        import wandb

        wandb.require("service")
        wandb_run = wandb.init(config=config, **wandb_cfg_dict)

        log.info(f"wandb initialized")

        return wandb_run
    except Exception as e:
        log.warning(f"Cannot initialize wandb: {e}")
        return
