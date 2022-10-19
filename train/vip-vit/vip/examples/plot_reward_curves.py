# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import cv2
import glob
from matplotlib import pyplot as plt
import numpy as np
import os 

import torch 
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image  

from vip import load_vip

def load_embedding(rep='vip'):
    if rep == "vip":
        model = load_vip()
        transform = T.Compose([T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor()])
    elif rep == "r3m":
        from r3m import load_r3m
        model = load_r3m("resnet50")
        transform = T.Compose([T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor()])
    elif rep == "resnet":
        model = models.resnet50(pretrained=True, progress=False)
        transform = T.Compose([T.Resize(256),
                            T.CenterCrop(224),
                            T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return model, transform 

def main(reps):
    model_transform_all = {}
    for rep in reps:
        model, transform = load_embedding(rep)
        model.to('cuda')
        model.eval()
        model_transform_all[rep] = (model, transform)

    embedding_names = {'vip': 'VIP', 'resnet': 'ResNet', 'r3m': 'R3M'}
    colors = {'vip': 'tab:blue', 'resnet': 'tab:orange', 'r3m':'tab:red'}
   
    os.makedirs('embedding_curves', exist_ok=True)
    videos = glob.glob("demo_realrobot/*")
    for video_id, vid in enumerate(videos):
        task_name = vid.split('/')[-1].split('.')[0]
        print(task_name)
        vidcap = cv2.VideoCapture(vid)
        count = 0
        imgs = []
        while True:
            success,image = vidcap.read()
            if not success:
                break
            imgs.append(image)
            count += 1

        # get correct rgb channels
        for i in range(len(imgs)):
            imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR)

        for key in model_transform_all:
            model, transform = model_transform_all[key]

            # transform images based on choice of representation 
            imgs_cur = []
            for i in range(len(imgs)):
                imgs_cur.append(transform(Image.fromarray(imgs[i].astype(np.uint8))))
            imgs_cur = torch.stack(imgs_cur)
            if key in ['vip', 'r3m']:
                imgs_cur = imgs_cur * 255

            with torch.no_grad():
                embeddings = model(imgs_cur.cuda())
                embeddings = embeddings.cpu().numpy()

            # get goal embedding
            goal_embedding = embeddings[-1]

            # compute goal embedding distance
            distances = [] 
            for t in range(embeddings.shape[0]):
                cur_embedding = embeddings[t]
                cur_distance = np.linalg.norm(goal_embedding-cur_embedding)
                distances.append(cur_distance)
            distances = np.array(distances) / distances[0] # normalize to [0,1]

            # plot embedding distance curves
            alpha = 0.45 if key != 'vip' else 1.0
            plt.plot(np.arange(len(distances)), distances, color=colors[key], label=embedding_names[key], linewidth=3, alpha=alpha)

        plt.xlabel("Frame", fontsize=15)
        plt.ylabel("Normalized Distance", fontsize=15)
        plt.title(f"Embedding Distance Curves Comparison", fontsize=15)
        plt.legend(loc="upper right")
        newax = plt.axes([0.13,0.03,0.35,0.35], anchor='NE', zorder=1)
        newax.set_xticks([])
        newax.set_yticks([])
        newax.imshow(imgs[-1])
        plt.savefig(f"embedding_curves/{task_name}.png",bbox_inches='tight')
        plt.close()

           
if __name__ == '__main__':
    reps = ['vip', 'resnet']
    # reps = ['vip', 'r3m', 'resnet'] # requires installing r3m
    main(reps)