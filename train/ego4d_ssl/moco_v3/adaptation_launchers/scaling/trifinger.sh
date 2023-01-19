# wandb login --host=https://fairwandb.org/ --relogin

PYTHONPATH=. python main_moco.py environment.slurm=True \
    logging.wandb_project="habitat_adapt_rep" logging.name="moco-v3_vitb_trifinger_adapt_try2" \
    model.arch=vit_base environment.ngpu=4 environment.world_size=1 \
    optim.batch_size=200 optim.epochs=3000 optim.warmup_epochs=400 optim.lr=5e-4 \
    data.train_filelist="/checkpoint/maksymets/eaif/datasets/trifinger/manifest.txt" \
    model.load_path="/checkpoint/maksymets/eaif/models/scaling_hypothesis_mae/mae_vit_base_ego_inav_233_epochs.pth"
