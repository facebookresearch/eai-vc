# EAI Foundation Models

This package contains a minimal-dependency set of model loading code. Model definitions are defined under `src/vc_models/models`, with configurations (including reference checkpoint filepaths) under `src/vc_models/conf`.

## Installation

Within the environment defined in [environment.yml](../environment.yml):

`pip install -e ./vc_models`


## How to Add a Model

Follow these steps to add a new model to the repository:

1. We aim to minimize the number of unique model architectures under the [models](src/vc_models/models) directory. If you are using a common backbone, you can specify how the checkpoint maps to this common backbone. For example, you can see how this is done for [ResNet](src/vc_models/models/resnet/resnet.py).
1. Add a configuration file to [src/vc_models/conf/model](src/vc_models/conf/model) using the naming convention `{algorithm}_{architecture}_{data}{comment}` defined in [`get_model_tag`](src/vc_models/utils/__init__.py).
1. To ensure that the model has been properly added, run `pytest vc_models/tests/test_model_loading.py`.

By following these steps, you can add a new model to the repository and ensure that it is properly integrated with the existing codebase.