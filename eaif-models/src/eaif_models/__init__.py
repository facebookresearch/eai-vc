import os

eaif_models_dir_path = os.path.dirname(os.path.abspath(__file__))
eaif_models_config_files = os.listdir(eaif_models_dir_path + "/conf/model")
eaif_model_zoo = [f.split(".")[0] for f in eaif_models_config_files]
