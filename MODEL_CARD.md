# Model Card: VC-1 (Visual Cortex)

Last updated: 2023-03-28

Version: 1.0

- This doc: https://github.com/facebookresearch/eai-vc/blob/main/MODEL_CARD.md
- Other Links:
  [VC-1 Website](https://eai-vc.github.io/),
  [VC-1 Blogpost](https://ai.facebook.com/blog/robots-learning-video-simulation-artificial-visual-cortex-vc-1),
  [VC-1 Paper](https://arxiv.org/abs/2303.18240),
  [VC-1 Demo](https://github.com/facebookresearch/eai-vc/blob/main/tutorial/tutorial_vc.ipynb)

The VC-1 model is a vision transformer (ViT) pre-trained on over 4,000 hours of egocentric videos from 7 different sources, together with ImageNet. The model is trained using Masked Auto-Encoding (MAE) and is available in two sizes: ViT-B and ViT-L. The model is intended for use for EmbodiedAI tasks, such as object manipulation and indoor navigation.
* VC-1 (ViT-L): Our best model, uses a ViT-L backbone, also known simply as `VC-1` | [Download](https://dl.fbaipublicfiles.com/eai-vc/vc1_vitl.pth)
* VC-1-base (VIT-B): pre-trained on the same data as VC-1 but with a smaller backbone (ViT-B) | [Download](https://dl.fbaipublicfiles.com/eai-vc/vc1_vitb.pth)

## Model Details

- Model Name: VC-1 (Vision Transformer-based model)
- Architecture:
  - Patch size: 16x16
  - Embedding dimension: 1024
  - Number of layers: 24
  - Number of heads: 16
  - MLP ratio: 4
  - QKV bias: True
  - Layer normalization: eps=1e-6
- Inputs: Images presented in 224x224x3.
- Outputs: 1024x1 embedding.
- Image Size: 224
- Use of Classification Token: True
- Dropout Rate: 0.0
- Algorithm: MAE
- Epochs trained: 182
- Model authors: Arjun Majumdar, Karmesh Yadav, Sergio Arnaud, Yecheng Jason Ma, Claire Chen, Sneha Silwal, Aryan Jain, Vincent-Pierre Berges, Pieter Abbeel, Jitendra Malik, Dhruv Batra, Yixin Lin, Oleksandr Maksymets, Aravind Rajeswaran, and Franziska Meier.
- Person of Contact: Oleksandr Maksymets (FAIR)


## Citation

If you use this model, please cite:

```bibtex
@inproceedings{vc2023,
      title={Where are we in the search for an Artificial Visual Cortex for Embodied Intelligence?},
      author={Arjun Majumdar and Karmesh Yadav and Sergio Arnaud and Yecheng Jason Ma and Claire Chen and Sneha Silwal and Aryan Jain and Vincent-Pierre Berges and Pieter Abbeel and Jitendra Malik and Dhruv Batra and Yixin Lin and Oleksandr Maksymets and Aravind Rajeswaran and Franziska Meier},
      year={2023},
      eprint={2303.18240},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Model Data

Training data:
The VC-1 model was trained on a large-scale dataset of egocentric videos, consisting of over 5.6 million frames. The dataset includes three modalities: manipulation, navigation, and object recognition. The manipulation modality includes videos of people performing various manipulations, such as cooking, cleaning, and tool use. The navigation modality includes videos of people moving around in indoor environments, such as homes and offices. The object recognition modality includes images from the ImageNet dataset, which contains over 1.2 million images of objects in various categories.


This table provides an overview of the assembled datasets used for scaling hypothesis experiments, including the total number of frames and the frames used for each dataset:

| Dataset                   | Contains     | Total Frames  | Frames used |
| ----------------------|:-----------:|:-------------:|:-----------:|
| Ego4D                  | Ego4D       | 418,578,043   | 2,790,520   |
|                        |             |               |             |
| EgoM (Manipulation)    | Ego4D       | 418,578,043   | 2,790,520   |
|                        | 100DOH      | 99,899        | 99,899      |
|                        | SS-v2       | 25,209,271    | 315,115     |
|                        | Epic Kitchens| 19,965,439    | 332,757     |
|                        |             | Total         | 3,538,291   |
|                        |             |               |             |
| EgoO (OpenHouse24)     | Ego4D       | 418,578,043   | 2,790,520   |
|                        | OpenHouse24 | 27,806,971    | 499,442     |
|                        |             | Total         | 3,289,962   |
|                        |             |               |             |
| EgoN (Navigation)      | Ego4D       | 418,578,043   | 2,790,520   |
|                        | OpenHouse24 | 27,806,971    | 499,442     |
|                        | RealEstate10K| 10,000,000    | 303,087     |
|                        |             | Total         | 3,289,962   |
|                        |             |               |             |
| EgoMN (Manipulation, Navigation) | Ego4D+M   | 3,538,291    | 3,538,291   |
|                        | OpenHouse24 | 27,806,971    | 499,442     |
|                        | RealEstate10K| 10,000,000    | 303,087     |
|                        |             | Total         | 4,340,820   |
|                        |             |               |             |
| EgoMNI (Manipulation, Navigation, ImageNet) | Ego4D+MN | 4,340,820 | 4,340,820 |
|                        | ImageNet    | 1,281,167     | 1,281,167   |
|                        |             | Total         | 5,621,987   |

The VC-1 models were trained on EgoMNI (Manipulation, Navigation, ImageNet) assembled dataset.



Evaluation data (see also section [Evaluation Results](#performance)
below):
The mode was evaluated on CortexBench that includes 17 tasks from 7 benchmarks and described below:

| Benchmark | Tasks |
|-----------|-------|
| Adroit | Relocate, Reorient-Pen |
| MetaWorld | Assembly, Bin-Picking, Button-Press, Drawer-Open, Hammer |
| DeepMind Control | Finger-Spin, Reacher-Hard, Cheetah-Run, Walker-Stand, Walker-Walk |
| TriFinger | Reach-Cube, Push-Cube |
| Habitat | Image-Goal Navigation (ImageNav), Object-Goal Navigation (ObjectNav) |
| Habitat 2.0 | Mobile Pick |


## Model Creation & Maintenance

The VC-1 model was created by pre-training ViT-B and ViT-L on a combination of egocentric videos and ImageNet using Masked Auto-Encoding (MAE). The model is maintained by the authors and is available for open-source use.

## Model Usage

The VC-1 model is intended for EmbodiedAI tasks, such as object manipulation and indoor navigation.. The model outputs embeddings for image frame, which can be used as features for downstream tasks:

```
from vc_models.models.vit import model_utils

model,embd_size,model_transforms,model_info = model_utils.load_model(model_utils.VC1_BASE_NAME)

#the img loaded should be Bx3x250x250
img = your_function_here ...

#output will be of size Bx3x224x224
transformed_img = model_transforms(img)
#img will be 1x768
embedding = model(transformed_img)
```

## Performance

The performance of the models on the CortexBench:

| Model               | Adroit       |  Meta-World           |  DMControl          |  Trifinger           |  ObjectNav  | ImageNav   |    Mobile Pick | Mean Rank | Mean Success  |
| ------------------------ | ------------ | ---------------------- | ------------------- | --------------------- | ------------ | ------------ | ----------------- | --------- | -------------- |
| Ego4D (VIT-B) |  48.7 ± 1.3 |  86.1 ± 2.1 |  64.1 ± 2.3 |  68.3 ± 1.1 |    46.8 ± 1.1 |   64.0 ± 0.7 |        57.4 ± 2.2 | 8.6 | 62.2 |
| Ego4D (VIT-L) |  50.0 ± 1.2 |  92.9 ± 2.4 |  60.8 ± 3.3 |  69.7 ± 0.5 |    47.6 ± 1.1 |   55.8 ± 0.8 |        67.6 ± 2.1 | 5.9 | 63.5 |
| Ego4D+N (VIT-B) |  50.0 ± 2.4 |  86.4 ± 2.9 |  59.5 ± 2.4 |  67.8 ± 1.3 |    54.7 ± 1.1 |   68.7 ± 0.7 |        59.4 ± 2.2 | 7.2 | 63.8 |
| Ego4D+N (VIT-L) |  54.0 ± 1.2 |  89.1 ± 2.9 |  66.4 ± 1.7 |  66.9 ± 0.4 |    57.4 ± 1.1 |   70.5 ± 0.7 |        65.2 ± 2.1 | 3.5 | 67.1 |
| Ego4D+M (VIT-B) |  51.3 ± 2.4 |  83.5 ± 2.6 |  64.3 ± 1.8 |  69.1 ± 0.4 |    47.3 ± 1.1 |   65.8 ± 0.7 |        59.8 ± 2.2 | 7.0 | 63.0 |
| Ego4D+M (VIT-L) |  52.0 ± 1.3 |  88.3 ± 3.2 |  64.7 ± 2.4 |  64.7 ± 0.9 |    47.3 ± 1.1 |   65.5 ± 0.7 |        68.6 ± 2.1 | 6.0 | 64.4 |
| VC-1: Ego4D+MN (VIT-B) |  48.7 ± 2.4 |  85.3 ± 5.2 |  64.2 ± 1.9 |  70.3 ± 0.5 |    52.8 ± 1.1 |   68.9 ± 0.7 |        58.6 ± 2.2 | 6.9 | 64.1 |
| VC-1: Ego4D + MNI (VIT-L)  |   59.3 ± 5.2 |  88.8 ± 2.2 |  66.9 ± 1.4 |  71.7 ± 0.4 |    60.3 ± 1.1 |   70.3 ± 0.7 |        63.2 ± 2.2 | 2.4 | 68.7 |

## Limitations

The VC-1 model has been evaluated on a limited set of benchmarks and may not perform as well on other tasks. While we have focused on masked auto-encoders as the pre-training objective and ViT as the architecture in our study, there may be other SSL algorithms that exhibit different scaling behaviors or superior performance on the proposed datasets in our benchmark.

Additionally, the VC-1 model is computationally expensive to train and may not be practical for all use cases. The large size of the model may also pose challenges for deployment on resource-constrained devices.

It is important to note that although we utilize real-world images and videos for pre-training our visual representation models (PVRs), the evaluation benchmarks used in this study serve as proxies for actual robotic tasks. Therefore, the performance of the PVR models on real robots may differ from the rankings established in this study. Further research is necessary to fully evaluate the effectiveness of these models in real-world scenarios.

Finally, while we have made efforts to ensure fairness and avoid bias in our benchmark selection, it is possible that certain demographics or use cases may not be adequately represented in our evaluation tasks. Future work could explore additional benchmarks that address a wider range of scenarios and demographics.
