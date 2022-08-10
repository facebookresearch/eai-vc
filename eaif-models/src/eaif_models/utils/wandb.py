import logging


log = logging.getLogger(__name__)

def setup_wandb(project_name="eaif", entity="eai-foundations", resume=True):
    try:
        log.info(f"wand initializing...")
        import wandb

        wandb.require("service")
        wandb.init(project=project_name, entity=entity, resume=resume)

        log.info(f"wandb initialized")

        return wandb
    except Exception as e:
        log.warning(f"Cannot initialize wandb: {e}")
        return
