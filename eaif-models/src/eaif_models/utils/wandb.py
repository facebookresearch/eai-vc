import logging


log = logging.getLogger(__name__)

def setup_wandb(project_name="eaif", entity="eai-foundations", resume=True):
    try:
        import wandb
    except ImportError as e:
        log.warning(f"Cannot import wandb: {e}")
        return

    log.info(f"wand initializing...")

    wandb.require("service")
    wandb.init(project=project_name, entity=entity, resume=resume)

    log.info(f"wandb initialized")

    return wandb
