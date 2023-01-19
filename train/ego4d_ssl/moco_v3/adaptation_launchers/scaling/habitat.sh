# wandb login --host=https://fairwandb.org/ --relogin

# 100% in-domain data (oracle variant)
# PYTHONPATH=. python main_moco.py environment.slurm=True \
#     logging.wandb_project="habitat_adapt_rep" logging.name="moco-v3_vitb_hgsp_100percent_adapt_try2" \
#     model.arch=vit_base environment.ngpu=8 environment.world_size=8 \
#     optim.batch_size=3200 optim.epochs=300 optim.warmup_epochs=40 optim.lr=5e-5 \
#     data.train_filelist="/checkpoint/maksymets/eaif/datasets/hm3d+gibson.npy" \
#     model.load_path="/checkpoint/maksymets/eaif/models/scaling_hypothesis_mae/mae_vit_base_ego_inav_233_epochs.pth"

# 5 percent data (practical variant)
PYTHONPATH=. python main_moco.py environment.slurm=True \
    logging.wandb_project="habitat_adapt_rep" logging.name="moco-v3_vitb_hgsp_5percent_adapt_try2" \
    model.arch=vit_base environment.ngpu=8 environment.world_size=8 \
    optim.batch_size=3200 optim.epochs=300 optim.warmup_epochs=40 optim.lr=5e-5 \
    data.train_filelist="/checkpoint/maksymets/eaif/datasets/5percent_hm3d_gibson.npy" \
    model.load_path="/checkpoint/maksymets/eaif/models/scaling_hypothesis_mae/mae_vit_base_ego_inav_233_epochs.pth"

# 5 percent data (in-domain SSL)
# PYTHONPATH=. python main_moco.py environment.slurm=True \
#     logging.wandb_project="habitat_adapt_rep" logging.name="moco-v3_vitb_hgsp_5percent_IDP_try1" \
#     model.arch=vit_base environment.ngpu=8 environment.world_size=1 \
#     data.train_filelist="/checkpoint/maksymets/eaif/datasets/5percent_hm3d_gibson.npy" \
