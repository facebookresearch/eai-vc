# EAI Foundation Models

This package contains a minimal-dependency set of model loading code. Model definitions are defined under `src/eaif_models/models`, with configurations (including reference checkpoint filepaths) under `src/eaif_models/conf`.

## Installation

Within the environment defined in [environment.yml](../environment.yml):

`pip install -e ./eaif-models`

## Adding a model

1. We try to minimze the number of unique model architectures under [models](src/eaif_models/models). If you are using a common backbone, simply specify how the checkpoint maps to this common backbone (for example, see how this works for [mapping MoCo, R3M to ResNet](src/eaif_models/models/resnet/resnet.py)).
1. Add a config file to [src/eaif_models/conf/model](src/eaif_models/conf/model) following the naming convention defined in [`get_model_tag`](src/eaif_models/utils/__init__.py).
    1. The Hydra configuration is instantiated by calling [load_model](eaif-models/src/eaif_models/models/__init__.py), which defines the input / output of the model-loading API.
1. Run `pytest eaif-models/tests/test_model_loading.py` to ensure the model is properly added.