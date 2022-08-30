from omegaconf import DictConfig, OmegaConf

from rl_utils.logging.base_logger import Logger, LoggerCfgType

try:
    import wandb
except ImportError:
    wandb = None


class WbLogger(Logger):
    """
    Logger for logging to the weights and W&B online service.
    """

    def __init__(
        self,
        wb_proj_name: str,
        wb_entity: str,
        run_name: str,
        seed: int,
        log_dir: str,
        vid_dir: str,
        save_dir: str,
        smooth_len: int,
        full_cfg: LoggerCfgType,
        group_name: str = "",
        **kwargs,
    ):
        """
        :parameter run_name: If empty string then a random run name is assigned.
        :parameter group_name: If empty string then no group name is used.
        """
        if wandb is None:
            raise ImportError("Wandb is not installed")

        super().__init__(
            run_name, seed, log_dir, vid_dir, save_dir, smooth_len, full_cfg
        )
        if wb_proj_name == "" or wb_entity == "":
            raise ValueError(
                f"Must specify W&B project and entity name {wb_proj_name}, {wb_entity}"
            )

        self.wb_proj_name = wb_proj_name
        self.wb_entity = wb_entity
        self.wandb = self._create_wandb(full_cfg, group_name)

    def log_vals(self, key_vals, step_count):
        wandb.log(key_vals, step=int(step_count))

    def watch_model(self, model):
        wandb.watch(model)

    def _create_wandb(self, full_cfg: LoggerCfgType, group_name: str):
        if group_name == "":
            group_name = None
        if isinstance(full_cfg, DictConfig):
            full_cfg = OmegaConf.to_container(full_cfg, resolve=True)

        self.run = wandb.init(
            project=self.wb_proj_name,
            name=self.run_name,
            entity=self.wb_entity,
            group=group_name,
            config=full_cfg,
        )
        return wandb

    def collect_img(self, k: str, img_path: str, prefix: str = ""):
        use_k = prefix + k
        self._step_log_info[use_k] = wandb.Image(img_path)
        self._clear_keys.add(use_k)

    def close(self):
        self.run.finish()
